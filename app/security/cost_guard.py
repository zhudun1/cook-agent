# app/security/cost_guard.py
"""
成本熔断（Cost Guard）

防止单会话 / 单轮 LLM token 用量失控：
- 按会话（conversation_id / session_id）累计输入+输出 token
- 调用前检查预算，超过按配置动作处理：
  - warn:    仅告警（结构化日志）
  - refuse:  拒绝本次调用（结构化错误）
  - degrade: 提示降级到低成本层级
- 达到 warn_threshold_ratio 比例时先发出告警

当前实现为进程内计数器（单实例部署足够）；
多实例部署时应替换为 Redis 计数器（接口已隔离，见注释）。

线程安全：asyncio 单事件循环内使用 dict + 无 await 的更新，天然安全；
若从多线程调用（如 LLM usage 后台回调）需加锁，此处提供锁。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional

from app.config.security_config import CostGuardConfig

logger = logging.getLogger(__name__)


@dataclass
class CostCheckResult:
    """成本检查结果。"""

    allowed: bool
    action: str  # ok | warn | refuse | degrade
    reason: str
    session_used: int = 0
    session_budget: int = 0

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "reason": self.reason,
            "session_used": self.session_used,
            "session_budget": self.session_budget,
        }


class CostGuard:
    """
    成本熔断器。

    用法::

        guard = CostGuard()
        check = guard.check("session-1")       # 调用前
        if not check.allowed: ...
        guard.record("session-1", in_tokens, out_tokens)  # 调用后
    """

    def __init__(self, config: Optional[CostGuardConfig] = None):
        self.config = config or _default_config()
        # 告警标记保持进程内（仅影响日志频率，无需共享）
        self._warned: Dict[str, bool] = {}
        self._lock = threading.Lock()
        # 用量计数走统一存储后端（memory / redis），支持多实例
        self._backend = None

    def _b(self):
        """懒加载存储后端（避免 import 循环）。"""
        if self._backend is None:
            from app.storage.backend import get_storage_backend

            self._backend = get_storage_backend()
        return self._backend

    @staticmethod
    def _usage_key(session_id: str) -> str:
        return f"cost:usage:{session_id}"

    # ------------------------------------------------------------------
    async def check(
        self,
        session_id: str,
        estimated_turn_tokens: int = 0,
    ) -> CostCheckResult:
        """
        调用前检查是否允许继续消耗 token。

        Args:
            session_id: 会话标识（conversation_id / session_id）
            estimated_turn_tokens: 本次调用预估 token（0 表示未知）

        Returns:
            CostCheckResult
        """
        if not self.config.enabled:
            return CostCheckResult(True, "ok", "cost guard disabled")

        try:
            used = await self._b().hgetall(self._usage_key(session_id))
            used_total = int(used.get("input", 0)) + int(used.get("output", 0))
        except Exception as e:
            logger.debug("Cost guard backend read failed, allow: %s", e)
            return CostCheckResult(True, "ok", "cost guard backend unavailable")

        budget = self.config.session_token_budget
        if budget <= 0:
            return CostCheckResult(True, "ok", "unlimited budget")

        # 触发告警阈值
        warn_at = int(budget * self.config.warn_threshold_ratio)
        if used_total >= warn_at and not self._warned.get(session_id):
            self._warned[session_id] = True
            logger.warning(
                "Cost guard warn: session %s used %d/%d tokens",
                session_id,
                used_total,
                budget,
            )

        # 单轮预算检查
        turn_budget = self.config.per_turn_token_budget
        if turn_budget > 0 and estimated_turn_tokens > turn_budget:
            return CostCheckResult(
                False,
                "refuse",
                f"per-turn token budget exceeded ({estimated_turn_tokens} > {turn_budget})",
                used_total,
                budget,
            )

        # 会话预算检查
        if used_total >= budget:
            action = self.config.action
            allowed = action != "refuse"
            return CostCheckResult(
                allowed,
                action,
                f"session token budget exceeded ({used_total} >= {budget})",
                used_total,
                budget,
            )

        return CostCheckResult(True, "ok", "within budget", used_total, budget)

    async def record(self, session_id: str, input_tokens: int, output_tokens: int) -> None:
        """调用后累计 token 用量（hash 自增）。"""
        if not self.config.enabled or not session_id:
            return
        try:
            b = self._b()
            key = self._usage_key(session_id)
            if int(input_tokens or 0):
                await b.hincrby(key, "input", int(input_tokens))
            if int(output_tokens or 0):
                await b.hincrby(key, "output", int(output_tokens))
        except Exception as e:
            logger.debug("Cost guard record failed: %s", e)

    async def get_usage(self, session_id: str) -> dict:
        """查询会话累计用量。"""
        try:
            used = await self._b().hgetall(self._usage_key(session_id))
        except Exception as e:
            logger.debug("Cost guard usage read failed: %s", e)
            used = {}
        input_tokens = int(used.get("input", 0))
        output_tokens = int(used.get("output", 0))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    async def reset(self, session_id: Optional[str] = None) -> None:
        """重置计数（测试用；生产可做会话过期清理）。"""
        try:
            b = self._b()
            if session_id:
                await b.delete(self._usage_key(session_id))
                self._warned.pop(session_id, None)
            else:
                await b.delete(self._usage_key(""))  # noop guard
        except Exception as e:
            logger.debug("Cost guard reset failed: %s", e)


def _default_config() -> CostGuardConfig:
    try:
        from app.config import settings

        return settings.security.cost_guard
    except Exception:
        return CostGuardConfig()


# 全局单例
cost_guard = CostGuard()
