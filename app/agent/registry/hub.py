"""AgentHub: single entrypoint for Agent + Tool + Provider (MCP, custom).

Design goals:
- One import path for all registration/lookup APIs.
- Providers are first-class: builtin, mcp, and future user-defined.
- No backwards compatibility layer.

Public API intentionally mirrors what the rest of the codebase needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, Type, runtime_checkable

from app.agent.types import AgentConfig
from app.agent.tools.base import BaseTool, ToolExecutor

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolProvider(Protocol):
    """Tool source provider.

    Examples:
    - builtin provider: registers python-implemented tools
    - mcp provider: loads & registers tools from MCP servers
    - custom provider: user-defined tools from DB/config
    """

    name: str

    def get_tool(self, name: str) -> Optional[BaseTool]:
        raise NotImplementedError

    def get_tool_schema(self, name: str) -> Optional[dict]:
        raise NotImplementedError

    def get_tool_schemas(self, names: Optional[list[str]] = None) -> list[dict]:
        raise NotImplementedError

    def list_tool_names(self) -> list[str]:
        raise NotImplementedError

    def register_tool(self, tool: BaseTool) -> None:
        raise NotImplementedError

    def unregister_tool(self, name: str) -> bool:
        raise NotImplementedError

    def list_servers_with_tools(self) -> list[dict]:
        """Return tools grouped by server.

        Returns:
            List of server info dicts, each containing:
            - name: server name
            - type: "local" or "mcp"
            - tools: list of tool info dicts
        """
        raise NotImplementedError


@dataclass(frozen=True)
class _AgentEntry:
    cls: Type["BaseAgent"]
    config: AgentConfig


class AgentHub:
    """Unified module hub."""

    _agents: dict[str, _AgentEntry] = {}
    _providers: dict[str, ToolProvider] = {}
    # P3 多租户隔离：会话级有状态工具实例缓存
    # key = (session_id, tool_name)，仅缓存 is_stateful 工具的克隆实例
    # 进程内缓存（工具实例含连接，跨进程共享无意义）；大小上限防泄漏
    _session_tools: dict[tuple[str, str], "BaseTool"] = {}
    _session_tools_max: int = 2000

    # ==================== Agent ====================

    @classmethod
    def register_agent(cls, agent_cls: Type["BaseAgent"], config: AgentConfig) -> None:
        cls._agents[config.name] = _AgentEntry(cls=agent_cls, config=config)
        logger.info(f"Registered agent: {config.name}")

    @classmethod
    def get_agent(cls, name: str) -> "BaseAgent":
        entry = cls._agents.get(name)
        if not entry:
            raise KeyError(f"Agent '{name}' not found")
        return entry.cls(entry.config)

    @classmethod
    def get_agent_config(cls, name: str) -> AgentConfig:
        entry = cls._agents.get(name)
        if not entry:
            raise KeyError(f"Agent '{name}' not found")
        return entry.config

    @classmethod
    def list_agents(cls) -> list[str]:
        return list(cls._agents.keys())

    @classmethod
    def clear_agents(cls) -> None:
        cls._agents.clear()

    # ==================== Providers ====================

    @classmethod
    def register_provider(cls, provider: ToolProvider) -> None:
        if provider.name in cls._providers:
            raise ValueError(f"Provider already registered: {provider.name}")
        cls._providers[provider.name] = provider
        logger.info(f"Registered tool provider: {provider.name}")

    @classmethod
    def get_provider(cls, name: str) -> ToolProvider:
        provider = cls._providers.get(name)
        if not provider:
            raise KeyError(f"Provider '{name}' not found")
        return provider

    @classmethod
    def list_providers(cls) -> list[str]:
        return list(cls._providers.keys())

    @classmethod
    def clear_providers(cls) -> None:
        cls._providers.clear()

    # ==================== Tool surface (aggregated) ====================

    @classmethod
    def register_tool(cls, tool: BaseTool, provider: str = "local") -> None:
        cls.get_provider(provider).register_tool(tool)

    @classmethod
    def unregister_tool(cls, name: str) -> bool:
        for p in cls._providers.values():
            if p.get_tool(name):
                return p.unregister_tool(name)
        return False

    @classmethod
    def get_tool_for_session(
        cls,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[BaseTool]:
        """
        获取工具实例（P3 多租户隔离）。

        - 无 session_id：返回全局实例（schema 用途）
        - 有 session_id 且有状态工具：返回该会话的独立克隆实例（缓存复用）
        - 无状态工具：返回全局实例（安全共享）
        """
        if not session_id:
            return cls.get_tool(name, user_id)

        key = (session_id, name)
        cached = cls._session_tools.get(key)
        if cached is not None:
            return cached

        tool = cls.get_tool(name, user_id)
        if tool is None:
            return None
        if getattr(tool, "is_stateful", False):
            # 上限治理：超限时清理最旧的会话实例（LRU 近似，dict 保序）
            if len(cls._session_tools) >= cls._session_tools_max:
                try:
                    oldest = next(iter(cls._session_tools))
                    cls._session_tools.pop(oldest, None)
                except StopIteration:
                    pass
            cloned = tool.clone_for_session(user_id=user_id, session_id=session_id)
            cls._session_tools[key] = cloned
            return cloned
        return tool

    @classmethod
    def clear_session_tools(cls, session_id: Optional[str] = None) -> None:
        """清理会话级工具实例缓存（会话结束/内存治理时调用）。"""
        if session_id is None:
            cls._session_tools.clear()
            return
        for key in [k for k in cls._session_tools if k[0] == session_id]:
            cls._session_tools.pop(key, None)

    @classmethod
    def get_tool(cls, name: str, user_id: Optional[str] = None) -> Optional[BaseTool]:
        for p in cls._providers.values():
            # SubagentToolProvider 需要 user_id
            if p.name == "subagent" and user_id:
                tool = p.get_tool(name, user_id)  # type: ignore
            else:
                tool = p.get_tool(name)
            if tool:
                return tool
        return None

    @classmethod
    def get_tool_schemas(
        cls,
        names: Optional[list[str]] = None,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        if names is None:
            schemas: list[dict] = []
            for p in cls._providers.values():
                if p.name == "subagent" and user_id:
                    schemas.extend(p.get_tool_schemas(None, user_id))  # type: ignore
                else:
                    schemas.extend(p.get_tool_schemas(None))
            return schemas

        # keep order per names
        result: list[dict] = []
        for n in names:
            for p in cls._providers.values():
                if p.name == "subagent" and user_id:
                    schema = p.get_tool_schema(n, user_id)  # type: ignore
                else:
                    schema = p.get_tool_schema(n)
                if schema:
                    result.append(schema)
                    break
        return result

    @classmethod
    def list_tools(cls, user_id: Optional[str] = None) -> list[str]:
        names: list[str] = []
        for p in cls._providers.values():
            if p.name == "subagent" and user_id:
                names.extend(p.list_tool_names(user_id))  # type: ignore
            else:
                names.extend(p.list_tool_names())
        return names

    @classmethod
    def list_all_servers(cls, user_id: Optional[str] = None) -> list[dict]:
        """Aggregate all servers with tools from all providers.

        Returns:
            List of server dicts with unified structure:
            [
                { "name": "builtin", "type": "local", "tools": [...] },
                { "name": "amap", "type": "mcp", "tools": [...] },
                { "name": "subagents", "type": "subagent", "tools": [...] },
            ]
        """
        servers: list[dict] = []
        for p in cls._providers.values():
            if p.name == "subagent" and user_id:
                servers.extend(p.list_servers_with_tools(user_id))  # type: ignore
            else:
                servers.extend(p.list_servers_with_tools())
        return servers

    @classmethod
    def create_tool_executor(
        cls,
        tool_names: Optional[list[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ToolExecutor:
        if tool_names is None:
            tools: dict[str, BaseTool] = {}
            for p in cls._providers.values():
                if p.name == "subagent" and user_id:
                    tool_list = p.list_tool_names(user_id)  # type: ignore
                else:
                    tool_list = p.list_tool_names()
                for name in tool_list:
                    tool = cls.get_tool_for_session(name, user_id, session_id)
                    if tool:
                        tools[name] = tool
            return ToolExecutor(tools, user_id=user_id)

        tools = {}
        for n in tool_names:
            tool = cls.get_tool_for_session(n, user_id, session_id)
            if tool:
                tools[n] = tool
        return ToolExecutor(tools, user_id=user_id)

    # ==================== Cleanup ====================

    @classmethod
    def clear_all(cls) -> None:
        cls.clear_agents()
        cls.clear_providers()


# Imported only for type checking; avoid runtime circular import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agent.agents import BaseAgent  # pragma: no cover
