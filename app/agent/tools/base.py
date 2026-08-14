"""
Tool 基类和辅助类

Tool 是 Agent 可以调用的外部功能，如搜索、计算、API 调用等。

结构化错误设计（供 Agent 自主决策恢复路径）:
- 每个失败返回 ToolResult 携带 error_code / retryable / suggestion
- error_code: TIMEOUT | TOOL_ERROR | VALIDATION_ERROR | AUTH_ERROR |
              NOT_FOUND | RATE_LIMITED | UNKNOWN
- retryable: True 表示 Agent 可稍后重试（如限流/超时/瞬时故障）
- suggestion: 给 LLM 的自然语言恢复建议
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel

from app.agent.types import ToolResult

logger = logging.getLogger(__name__)


def classify_tool_error(exc: BaseException) -> tuple[str, bool, Optional[str]]:
    """
    将异常分类为结构化错误信息。

    Returns:
        (error_code, retryable, suggestion)
    """
    exc_name = type(exc).__name__
    exc_msg = str(exc).lower()

    if isinstance(exc, asyncio.TimeoutError) or "timeout" in exc_name.lower():
        return "TIMEOUT", True, "工具执行超时，可稍后重试或拆分请求"
    if "rate" in exc_msg or "429" in exc_msg:
        return "RATE_LIMITED", True, "触发限流，请稍后重试或降低调用频率"
    if any(k in exc_name.lower() for k in ("auth", "unauthorized", "forbidden", "apikey", "api_key")):
        return "AUTH_ERROR", False, "认证失败，请检查 API Key 或权限配置"
    if "not found" in exc_msg or "404" in exc_msg or exc_name == "KeyError":
        return "NOT_FOUND", False, "目标资源不存在，请检查参数或改用其他工具"
    if any(k in exc_name.lower() for k in ("validation", "valueerror", "typeerror", "jsondecode")):
        return "VALIDATION_ERROR", False, "参数或返回格式不合法，请修正调用参数"
    if any(k in exc_name.lower() for k in ("connection", "serviceunavailable", "badgateway", "internalserver")):
        return "TOOL_ERROR", True, "服务暂时不可用，可稍后重试"
    return "TOOL_ERROR", False, "工具执行失败，请检查参数或改用其他工具"


class BaseTool(ABC):
    """
    Tool 基类。

    所有 Tool 必须继承此类并实现 execute 方法。
    """

    # Tool 基本信息
    name: str
    description: str

    # JSON Schema 格式的参数定义
    parameters: dict = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self):
        """初始化 Tool。"""
        if not hasattr(self, "name") or not self.name:
            raise ValueError("Tool must have a name")
        if not hasattr(self, "description") or not self.description:
            raise ValueError("Tool must have a description")

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        执行 Tool。

        Args:
            **kwargs: Tool 参数

        Returns:
            ToolResult: 执行结果
        """
        pass

    async def safe_execute(self, **kwargs) -> ToolResult:
        """
        安全执行 Tool，捕获异常并返回结构化错误。

        Args:
            **kwargs: Tool 参数

        Returns:
            ToolResult: 执行结果（失败时携带 error_code/retryable/suggestion）
        """
        try:
            result = await self.execute(**kwargs)
            # 归一化：execute 可能返回 dict / ToolResult
            if isinstance(result, ToolResult):
                if not result.success and result.error_code is None:
                    code, retryable, suggestion = classify_tool_error(
                        RuntimeError(result.error or "Tool failed")
                    )
                    result.error_code = code
                    result.retryable = retryable
                    result.suggestion = suggestion
                return result
            if isinstance(result, dict):
                success = result.get("success", True)
                if not success and result.get("error"):
                    code, retryable, suggestion = classify_tool_error(
                        RuntimeError(result["error"])
                    )
                    result.setdefault("error_code", code)
                    result.setdefault("retryable", retryable)
                    result.setdefault("suggestion", suggestion)
                return ToolResult(**result)
            return ToolResult(success=True, data=result)
        except asyncio.TimeoutError as e:
            logger.exception(f"Tool {self.name} timed out: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool {self.name} timed out",
                error_code="TIMEOUT",
                retryable=True,
                suggestion="工具执行超时，可稍后重试或简化请求",
            )
        except Exception as e:
            logger.exception(f"Tool {self.name} execution failed: {e}")
            code, retryable, suggestion = classify_tool_error(e)
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                error_code=code,
                retryable=retryable,
                suggestion=suggestion,
            )

    def to_openai_schema(self) -> dict:
        """
        转换为 OpenAI function calling 格式。

        Returns:
            OpenAI tool schema
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def parse_arguments(self, arguments: str | dict) -> dict:
        """
        解析 Tool 调用参数。

        Args:
            arguments: JSON 字符串或字典

        Returns:
            解析后的参数字典
        """
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return {}
        return arguments

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"


class MCPTool(BaseTool):
    """
    MCP (Model Context Protocol) Tool 封装。

    用于调用外部 MCP 服务。
    """

    mcp_endpoint: str
    mcp_tool_name: str

    def __init__(
        self,
        name: str,
        description: str,
        mcp_endpoint: str,
        mcp_tool_name: str,
        mcp_headers: Optional[dict[str, str]] = None,
        parameters: Optional[dict] = None,
    ):
        self.name = name
        self.description = description
        self.mcp_endpoint = mcp_endpoint
        self.mcp_tool_name = mcp_tool_name
        self.mcp_headers = mcp_headers or {}
        if parameters:
            self.parameters = parameters
        super().__init__()

    async def execute(self, **kwargs) -> ToolResult:
        """
        调用 MCP 服务。

        Uses MCPClient to execute the tool on the remote MCP server.
        """
        try:
            from app.agent.tools.mcp.client import MCPClient

            client = MCPClient(self.mcp_endpoint, headers=self.mcp_headers)

            if kwargs and "user_id" in kwargs:
                kwargs.pop("user_id")

            return await client.call_tool(self.mcp_tool_name, kwargs)

        except Exception as e:
            logger.exception(f"MCP Tool {self.name} execution failed: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"MCP tool execution failed: {str(e)}",
            )


class ToolExecutor:
    """
    Tool 执行器。

    负责执行 Tool 调用并返回结果。
    支持每个工具独立超时（默认/覆盖配置，见 config.yml resilience.tools）。
    """

    def __init__(self, tools: dict[str, BaseTool], user_id: Optional[str] = None):
        """
        初始化执行器。

        Args:
            tools: Tool 名称到实例的映射
            user_id: 用户 ID（用于自动注入工具调用）
        """
        self.tools = tools
        self.user_id = user_id

    def _timeout_for(self, tool_name: str) -> Optional[float]:
        """获取工具超时（秒）；0 或 None 表示不限制。"""
        try:
            from app.config import settings

            cfg = settings.resilience.tools
            override = (cfg.timeout_overrides or {}).get(tool_name)
            if override is not None:
                return float(override) if float(override) > 0 else None
            default = cfg.default_timeout_seconds
            return float(default) if default and default > 0 else None
        except Exception:
            return None

    async def execute(
        self,
        tool_name: str,
        arguments: str | dict,
        event_handler: Optional[Callable[[Any], Awaitable[None]]] = None,
        bypass_approval: bool = False,
    ) -> ToolResult:
        """
        执行指定的 Tool（权限检查 + 审批 + 独立超时 + 注入净化）。

        Args:
            tool_name: Tool 名称
            arguments: Tool 参数
            event_handler: 子代理事件处理器
            bypass_approval: 审批通过后的重执行标记（跳过审批检查）

        Returns:
            ToolResult: 执行结果（失败时携带 error_code/retryable/suggestion）
        """
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found",
                error_code="NOT_FOUND",
                retryable=False,
                suggestion="请检查工具名称是否正确，或改用其他可用工具",
            )

        # ==========================================================================
        # P0 安全 1/3: 工具权限矩阵
        # ==========================================================================
        try:
            from app.security.permissions import permission_matrix

            decision = permission_matrix.check(tool_name, self.user_id)
            if not decision.allowed:
                logger.warning(
                    "Permission denied: tool=%s user=%s reason=%s",
                    tool_name,
                    self.user_id,
                    decision.reason,
                )
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Permission denied: {decision.reason}",
                    error_code="PERMISSION_DENIED",
                    retryable=False,
                    suggestion="你没有权限调用该工具，请告知用户或改用其他工具",
                )
        except Exception as e:
            logger.debug("Permission check skipped: %s", e)

        # ==========================================================================
        # P0 安全 2/3: 人工介入审批（危险工具）
        # ==========================================================================
        if not bypass_approval:
            try:
                from app.security.approval import approval_manager

                if approval_manager.requires_approval(tool_name):
                    if approval_manager.auto_approve(self.user_id):
                        logger.info(
                            "Tool %s auto-approved for admin %s",
                            tool_name,
                            self.user_id,
                        )
                    else:
                        pre_args = tool.parse_arguments(arguments) or {}
                        req = approval_manager.request(
                            tool_name,
                            dict(pre_args),
                            user_id=self.user_id,
                        )
                        logger.info(
                            "Tool %s requires approval: approval_id=%s",
                            tool_name,
                            req.approval_id,
                        )
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"Tool '{tool_name}' requires user approval",
                            error_code="APPROVAL_PENDING",
                            retryable=True,
                            suggestion="等待用户审批该工具调用",
                            approval_id=req.approval_id,
                        )
            except Exception as e:
                logger.debug("Approval check skipped: %s", e)

        parsed_args = tool.parse_arguments(arguments) or {}
        parsed_args = dict(parsed_args)
        if self.user_id and "user_id" not in parsed_args:
            parsed_args["user_id"] = self.user_id
        if event_handler and tool_name.startswith("subagent_"):
            parsed_args["event_handler"] = event_handler

        timeout = self._timeout_for(tool_name)
        if timeout is None:
            result = await tool.safe_execute(**parsed_args)
        else:
            try:
                result = await asyncio.wait_for(
                    tool.safe_execute(**parsed_args),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Tool %s timed out after %.1fs (user_id=%s)",
                    tool_name,
                    timeout,
                    self.user_id,
                )
                result = ToolResult(
                    success=False,
                    data=None,
                    error=f"Tool '{tool_name}' timed out after {timeout:.1f}s",
                    error_code="TIMEOUT",
                    retryable=True,
                    suggestion="工具执行超时，可稍后重试、简化参数或改用其他工具",
                )

        # ==========================================================================
        # P0 安全 3/3: 工具返回内容注入检测（纵深防御第二层）
        # ==========================================================================
        if result.success and result.data is not None:
            try:
                from app.security.injection import injection_detector

                scan = injection_detector.scan_tool_result(
                    tool_name, str(result.data)
                )
                if scan.blocked and scan.sanitized is not None:
                    result.data = scan.sanitized
            except Exception as e:
                logger.debug("Injection scan skipped: %s", e)

        return result

    def get_schemas(self, tool_names: Optional[list[str]] = None) -> list[dict]:
        """
        获取 Tool schemas。

        Args:
            tool_names: 要获取的 Tool 名称列表，None 表示全部

        Returns:
            OpenAI tool schema 列表
        """
        if tool_names is None:
            return [t.to_openai_schema() for t in self.tools.values()]
        return [
            self.tools[name].to_openai_schema()
            for name in tool_names
            if name in self.tools
        ]
