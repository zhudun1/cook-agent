# app/security/permissions.py
"""
工具权限矩阵（Tool Permission Matrix）

工具 × 用户 授权模型：
1. admin 用户拥有全部工具权限
2. 用户显式 deny 优先于 allow（黑名单优先）
3. 敏感工具（default_denied_tools）默认拒绝，除非显式 allow
4. 其余工具默认允许

规则匹配支持 glob（如 "delete_*"）。

接入点：ToolExecutor.execute 执行前检查，
越权返回结构化错误（PERMISSION_DENIED, retryable=False）供 Agent 决策。
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass
from typing import List, Optional

from app.config.security_config import PermissionConfig

logger = logging.getLogger(__name__)


@dataclass
class PermissionDecision:
    """权限决策结果。"""

    allowed: bool
    reason: str
    matched_rule: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "matched_rule": self.matched_rule,
        }


class PermissionMatrix:
    """
    工具权限矩阵。

    用法::

        matrix = PermissionMatrix()
        decision = matrix.check("web_search", user_id="u1")
        if not decision.allowed:
            return ToolResult(success=False, error_code="PERMISSION_DENIED", ...)
    """

    def __init__(self, config: Optional[PermissionConfig] = None):
        self.config = config or _default_config()

    # ------------------------------------------------------------------
    def check(self, tool_name: str, user_id: Optional[str] = None) -> PermissionDecision:
        """
        检查用户是否有权调用指定工具。

        Args:
            tool_name: 工具名
            user_id: 用户 ID（None 视为匿名）

        Returns:
            PermissionDecision
        """
        if not self.config.enabled:
            return PermissionDecision(True, "permission check disabled")

        # 1. admin 全放行
        if user_id and user_id in (self.config.admin_users or []):
            return PermissionDecision(
                True, "admin user", matched_rule=f"admin:{user_id}"
            )

        # 2. 显式 deny（黑名单优先）
        if user_id:
            denied = self.config.deny.get(user_id) or []
            if self._match_any(tool_name, denied):
                return PermissionDecision(
                    False,
                    f"tool '{tool_name}' denied for user {user_id}",
                    matched_rule=f"deny:{user_id}",
                )

        # 3. 显式 allow（覆盖默认拒绝）
        if user_id:
            allowed = self.config.allow.get(user_id) or []
            if self._match_any(tool_name, allowed):
                return PermissionDecision(
                    True,
                    f"tool '{tool_name}' explicitly allowed for {user_id}",
                    matched_rule=f"allow:{user_id}",
                )

        # 4. 敏感工具默认拒绝
        if self._match_any(tool_name, self.config.default_denied_tools or []):
            return PermissionDecision(
                False,
                f"tool '{tool_name}' is sensitive and denied by default",
                matched_rule="default_denied",
            )

        # 5. 默认允许
        return PermissionDecision(True, "allowed by default")

    # ------------------------------------------------------------------
    @staticmethod
    def _match_any(tool_name: str, patterns: List[str]) -> bool:
        """glob 模式匹配（含精确名）。"""
        return any(fnmatch.fnmatch(tool_name, p) for p in patterns)


def _default_config() -> PermissionConfig:
    try:
        from app.config import settings

        return settings.security.permissions
    except Exception:
        return PermissionConfig()


# 全局单例
permission_matrix = PermissionMatrix()
