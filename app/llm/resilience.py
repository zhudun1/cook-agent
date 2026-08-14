from __future__ import annotations

# app/llm/resilience.py
"""
LLM 调用韧性层：指数退避重试 + 模型降级切换。

设计：
1. `is_retryable_error` - 根据异常类型/消息判断是否可重试
2. `async_retry` - 指数退避重试循环（带抖动），返回最后一次异常
3. `call_with_fallback` - 模型降级链：
   - 同一 profile 内 model_names 逐个重试（先负载均衡后降级）
   - 跨层降级链（如 fast -> normal -> vision），每层独立重试
4. 所有失败均抛出 `LLMResilienceError`，携带 attempts / models_tried /
   error_type / retryable 结构化信息，供 Agent 自主决策恢复路径

与 telemetry 解耦：通过可选的 `on_attempt` / `on_fallback` 回调上报事件，
由调用方决定是否写入结构化日志/轨迹。
"""


import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, List, Optional

if TYPE_CHECKING:
    from app.config.resilience_config import (
        FallbackConfig,
        ResilienceConfig,
        RetryConfig,
    )

logger = logging.getLogger(__name__)


class LLMResilienceError(RuntimeError):
    """LLM 调用经重试与降级后仍失败的结构化错误。"""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        retryable: bool,
        attempts: int,
        models_tried: List[str],
        cause: Optional[BaseException] = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable
        self.attempts = attempts
        self.models_tried = models_tried
        self.cause = cause

    def to_dict(self) -> dict:
        return {
            "error": self.args[0],
            "error_type": self.error_type,
            "retryable": self.retryable,
            "attempts": self.attempts,
            "models_tried": self.models_tried,
        }


@dataclass
class ResilienceEvent:
    """韧性层事件（供 telemetry 消费）"""

    kind: str  # "attempt" | "retry" | "model_fallback" | "llm_type_fallback" | "give_up"
    llm_type: str
    model: str
    attempt: int
    error: str | None = None
    error_type: str | None = None
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "llm_type": self.llm_type,
            "model": self.model,
            "attempt": self.attempt,
            "error": self.error,
            "error_type": self.error_type,
            **self.detail,
        }


# ---------------------------------------------------------------------------
# 可重试判断
# ---------------------------------------------------------------------------

def is_retryable_error(exc: BaseException, cfg: RetryConfig) -> bool:
    """
    判断异常是否可重试。

    规则：异常类名或异常消息命中 retryable_errors 任一关键字即可重试；
    未命中则默认不可重试（避免掩盖确定性错误）。
    """
    exc_name = type(exc).__name__
    exc_msg = str(exc)
    for keyword in cfg.retryable_errors:
        keyword_lower = keyword.lower()
        if keyword_lower in exc_name.lower() or keyword_lower in exc_msg.lower():
            return True
    # 常见网络层错误兜底
    if isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return True
    return False


# ---------------------------------------------------------------------------
# 指数退避重试
# ---------------------------------------------------------------------------

async def async_retry(
    call: Callable[[], Awaitable[Any]],
    cfg: RetryConfig,
    *,
    on_event: Optional[Callable[[ResilienceEvent], None]] = None,
    llm_type: str = "normal",
    model: str = "",
) -> Any:
    """
    带指数退避的异步重试循环。

    Args:
        call: 无参异步调用（已绑定模型）
        cfg: 重试配置
        on_event: 事件回调（attempt/retry/give_up）
        llm_type / model: 事件上下文信息

    Returns:
        调用结果

    Raises:
        LLMResilienceError: 重试耗尽后抛出（带结构化信息）
    """
    max_retries = cfg.max_retries
    delay = cfg.base_delay_seconds
    last_exc: Optional[BaseException] = None

    for attempt in range(max_retries + 1):
        if on_event:
            on_event(
                ResilienceEvent(
                    kind="attempt",
                    llm_type=llm_type,
                    model=model,
                    attempt=attempt + 1,
                )
            )
        try:
            return await call()
        except Exception as e:
            last_exc = e
            retryable = is_retryable_error(e, cfg)
            if attempt >= max_retries or not retryable:
                if on_event:
                    on_event(
                        ResilienceEvent(
                            kind="give_up",
                            llm_type=llm_type,
                            model=model,
                            attempt=attempt + 1,
                            error=str(e),
                            error_type=type(e).__name__,
                            detail={"retryable": retryable},
                        )
                    )
                raise LLMResilienceError(
                    f"LLM call failed after {attempt + 1} attempt(s): {e}",
                    error_type=type(e).__name__,
                    retryable=retryable,
                    attempts=attempt + 1,
                    models_tried=[model],
                    cause=e,
                ) from e

            # 指数退避 + 抖动
            backoff = min(delay * (cfg.exponential_factor ** attempt), cfg.max_delay_seconds)
            if cfg.jitter:
                backoff *= random.uniform(0.8, 1.2)
            if on_event:
                on_event(
                    ResilienceEvent(
                        kind="retry",
                        llm_type=llm_type,
                        model=model,
                        attempt=attempt + 1,
                        error=str(e),
                        error_type=type(e).__name__,
                        detail={"backoff_seconds": round(backoff, 2)},
                    )
                )
            logger.warning(
                "LLM retry %d/%d for %s/%s in %.2fs: %s",
                attempt + 1,
                max_retries,
                llm_type,
                model,
                backoff,
                e,
            )
            await asyncio.sleep(backoff)

    # 不可达（循环必然 return 或 raise）
    raise LLMResilienceError(
        "LLM retry loop exhausted",
        error_type=type(last_exc).__name__ if last_exc else "Unknown",
        retryable=False,
        attempts=max_retries + 1,
        models_tried=[model],
        cause=last_exc,
    )  # pragma: no cover


# ---------------------------------------------------------------------------
# 模型降级链
# ---------------------------------------------------------------------------

async def call_with_fallback(
    call_builder: Callable[[str, str], Awaitable[Any]],
    llm_types: List[str],
    model_names: dict[str, List[str]],
    config: ResilienceConfig,
    *,
    on_event: Optional[Callable[[ResilienceEvent], None]] = None,
) -> Any:
    """
    带模型降级链的调用。

    Args:
        call_builder: 异步函数 builder(llm_type, model_name) -> 已绑定模型的调用
        llm_types: 降级链（如 ["fast", "normal", "vision"]）
        model_names: llm_type -> 模型名列表
        config: 韧性配置
        on_event: 事件回调

    Returns:
        调用结果

    Raises:
        LLMResilienceError: 全部模型/层级均失败
    """
    fallback_cfg = config.fallback
    retry_cfg = config.retry

    models_tried: List[str] = []
    last_error: Optional[LLMResilienceError] = None
    total_attempts = 0

    # 最多处理 1（主层）+ max_fallback_steps（降级步数）个层级
    allowed_tiers = 1 + fallback_cfg.max_fallback_steps
    for llm_type in llm_types[:allowed_tiers]:
        models = model_names.get(llm_type) or []
        if not models:
            continue

        for model in models:
            try:
                return await async_retry(
                    lambda t=llm_type, m=model: call_builder(t, m),
                    retry_cfg,
                    on_event=on_event,
                    llm_type=llm_type,
                    model=model,
                )
            except LLMResilienceError as e:
                last_error = e
                models_tried.extend(e.models_tried)
                total_attempts += e.attempts
                # 记录模型降级事件
                if on_event:
                    on_event(
                        ResilienceEvent(
                            kind="model_fallback",
                            llm_type=llm_type,
                            model=model,
                            attempt=e.attempts,
                            error=str(e),
                            error_type=e.error_type,
                            detail={"models_tried": e.models_tried},
                        )
                    )

        # 记录跨层降级事件
        if on_event:
            on_event(
                ResilienceEvent(
                    kind="llm_type_fallback",
                    llm_type=llm_type,
                    model="",
                    attempt=0,
                    error="all models failed for this tier",
                    detail={"next_tier": llm_types[1] if len(llm_types) > 1 else None},
                )
            )

    raise LLMResilienceError(
        f"All LLM fallbacks failed: {models_tried}",
        error_type=last_error.error_type if last_error else "Unknown",
        retryable=last_error.retryable if last_error else False,
        attempts=total_attempts or len(models_tried),
        models_tried=models_tried,
        cause=last_error.cause if last_error else None,
    )


def build_llm_type_chain(
    primary: str,
    config: ResilienceConfig,
) -> List[str]:
    """
    构建降级链：从 primary 类型出发，按 fallback_llm_types 配置生成有序链。

    例如 primary=fast, fallback_llm_types=[fast, normal, vision]
    -> [fast, normal]（默认最多降级 2 步）
    """
    fallback_cfg = config.fallback
    ordered = list(fallback_cfg.fallback_llm_types)
    if primary not in ordered:
        ordered = [primary] + [t for t in ordered if t != primary]

    # 从 primary 开始取 max_fallback_steps + 1 层
    start = ordered.index(primary)
    chain = ordered[start : start + fallback_cfg.max_fallback_steps + 1]
    return chain
