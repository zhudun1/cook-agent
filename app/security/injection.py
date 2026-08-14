# app/security/injection.py
"""
注入纵深防御：工具返回内容注入检测

第一层（用户输入检查）由 prompt_guard 承担；
本模块为第二层——**工具返回内容**（搜索结果 / 网页内容 / 外部 API 响应）
可能携带恶意指令（"忽略之前的指令"、"你是系统提示词"等），
在进入 LLM 上下文前检测并处理。

动作:
- block: 命中则截断/移除风险内容，返回 sanitized 文本（默认）
- warn:  仅告警，内容原样传递

检测源：复用 PromptGuard 的模式规则 + 本模块内置工具内容注入模式 +
配置追加的 extra_patterns。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.config.security_config import InjectionGuardConfig

logger = logging.getLogger(__name__)

# 内置注入模式（工具返回内容场景）
# 覆盖：指令覆盖、系统提示词泄露、越狱、角色扮演劫持、钓鱼式诱导
BUILTIN_INJECTION_PATTERNS: List[str] = [
    # 指令覆盖
    r"忽略\s*(之前|上面|前面)?\s*(的)?\s*(所有)?\s*(指令|指示|要求|命令|设定|system prompt)",
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|messages|context)",
    r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
    r"你(现在|接下来)?\s*(不再|不需要|不要)\s*(遵守|遵循|理会)\s*(之前|原有|系统)",
    # 系统提示词/角色劫持
    r"(你是|你现在是|你的角色是|act\s+as|you\s+are\s+now)\s*(系统|system|管理员|admin|assistant\s*的?\s*提示词|the\s+system\s+prompt)",
    r"打印|显示|泄露|暴露|reveal|show\s+(me\s+)?(你的|the)\s*(system\s+prompt|提示词|指令|instructions)",
    # 越狱 / 注入 payload 特征
    r"DAN\b|jailbreak|越狱",
    r"<\|?im_start\|?>|<\|?system\|?>",
    r"（?忽略以上内容[，,]?以下?[，,]?为?[：:]?）?",
    r"ignore\s+everything\s+(above|before)",
]


@dataclass
class InjectionScanResult:
    """注入检测结果。"""

    blocked: bool
    reason: Optional[str] = None
    matched_patterns: List[str] = field(default_factory=list)
    sanitized: Optional[str] = None


class InjectionDetector:
    """
    工具返回内容注入检测器。

    用法::

        detector = InjectionDetector()
        result = detector.scan_tool_result(tool_name="web_search", content="...")
        if result.blocked:
            content = result.sanitized  # 使用净化后的内容
    """

    def __init__(self, config: Optional[InjectionGuardConfig] = None):
        self.config = config or _default_config()
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> List[re.Pattern]:
        patterns = list(BUILTIN_INJECTION_PATTERNS)
        patterns.extend(self.config.extra_patterns or [])
        compiled = []
        for p in patterns:
            try:
                compiled.append(re.compile(p, re.IGNORECASE))
            except re.error as e:
                logger.warning("Invalid injection pattern %r: %s", p, e)
        return compiled

    # ------------------------------------------------------------------
    def scan_tool_result(
        self,
        tool_name: str,
        content: str,
        max_preview: int = 2000,
    ) -> InjectionScanResult:
        """
        扫描工具返回内容。

        Args:
            tool_name: 工具名（用于日志/审计）
            content: 工具返回内容
            max_preview: 风险内容在日志中的截断长度

        Returns:
            InjectionScanResult（blocked=True 时 sanitized 为净化后内容）
        """
        if not self.config.enabled or not self.config.check_tool_results:
            return InjectionScanResult(blocked=False)
        if not content:
            return InjectionScanResult(blocked=False)

        matched = []
        for pattern in self._patterns:
            if pattern.search(content):
                matched.append(pattern.pattern)

        if not matched:
            return InjectionScanResult(blocked=False)

        reason = (
            f"possible prompt injection detected in tool '{tool_name}' result "
            f"({len(matched)} pattern(s))"
        )
        logger.warning(
            "%s: %s", reason, content[:max_preview],
        )
        # 上报结构化日志（安全事件）
        try:
            from app.telemetry.logger import log_structured_event

            log_structured_event(
                "security_injection_tool_result",
                {
                    "tool": tool_name,
                    "patterns": matched,
                    "content_preview": content[:max_preview],
                },
            )
        except Exception:
            pass

        if self.config.action == "block":
            sanitized = self._sanitize(content, matched)
            return InjectionScanResult(
                blocked=True,
                reason=reason,
                matched_patterns=matched,
                sanitized=sanitized,
            )

        return InjectionScanResult(
            blocked=True,
            reason=reason,
            matched_patterns=matched,
            sanitized=content,  # warn 模式：内容原样
        )

    def _sanitize(self, content: str, matched: List[str]) -> str:
        """净化：移除命中注入的片段（保留其余内容）。"""
        sanitized = content
        for pattern in matched:
            try:
                sanitized = re.sub(pattern, "[注入内容已过滤]", sanitized, flags=re.IGNORECASE)
            except re.error:
                continue
        # 双重保险：截断超长内容
        if len(sanitized) > 20000:
            sanitized = sanitized[:20000] + "...[truncated]"
        return sanitized


def _default_config() -> InjectionGuardConfig:
    try:
        from app.config import settings

        return settings.security.injection_guard
    except Exception:
        return InjectionGuardConfig()


# 全局单例
injection_detector = InjectionDetector()
