from __future__ import annotations

"""
LLM Provider - 统一的 LLM 初始化和调用入口

核心概念:
1. LLMProvider - 全局 LLM 提供者，管理配置和创建 LLM 实例
2. LLMInvoker - LLM 调用器，封装了调用逻辑和 usage tracking
"""


import random
from typing import Any, AsyncIterator, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from app.config.llm_config import LLMConfig, LLMProfileConfig, LLMType


class LLMProvider:
    """
    LLM 提供者 - 统一的 LLM 创建和管理入口

    使用方式:
        provider = LLMProvider(settings.llm)

        # 创建带 usage tracking 的 invoker（推荐）
        invoker = provider.create_invoker("fast")
        response = await invoker.ainvoke(messages)

        # 直接创建 ChatOpenAI 实例（不推荐，除非有特殊需求）
        llm = provider.create_llm("normal", streaming=True)
    """

    def __init__(self, config: LLMConfig):
        """
        初始化 LLM Provider

        Args:
            config: LLM 配置（通常来自 settings.llm）
        """
        self._config = config

    def get_profile(self, llm_type: LLMType | str | None = None) -> LLMProfileConfig:
        """获取指定类型的 LLM 配置 profile"""
        return self._config.get_profile(llm_type)

    def pick_model(self, llm_type: LLMType | str | None = None) -> str:
        """
        随机选择一个模型名称（用于负载均衡）

        Args:
            llm_type: LLM 类型 (fast/normal)

        Returns:
            选中的模型名称
        """
        profile = self.get_profile(llm_type)
        if not profile.model_names:
            raise ValueError("model_names cannot be empty")
        return random.choice(profile.model_names)

    def create_llm(
        self,
        llm_type: LLMType | str | None = None,
        *,
        streaming: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> ChatOpenAI:
        """
        创建原生 ChatOpenAI 实例

        注意: 此方法创建的实例不包含 usage tracking，
        推荐使用 create_invoker() 方法获取带 tracking 的调用器

        Args:
            llm_type: LLM 类型 (fast/normal/vision)
            streaming: 是否启用流式输出
            temperature: 温度参数（可选覆盖配置）
            max_tokens: 最大 tokens（可选覆盖配置）
            timeout: 请求超时时间（可选，vision profile 默认 120s）
            **kwargs: 其他 ChatOpenAI 参数

        Returns:
            ChatOpenAI 实例
        """
        profile = self.get_profile(llm_type)

        # Handle timeout - use profile's request_timeout if available
        effective_timeout = timeout
        if effective_timeout is None and hasattr(profile, "request_timeout"):
            effective_timeout = profile.request_timeout # type: ignore

        llm_kwargs = {
            "model": profile.pick_default_model(),
            "api_key": profile.api_key,
            "base_url": profile.base_url,
            "temperature": temperature if temperature is not None else profile.temperature,
            "max_completion_tokens": max_tokens if max_tokens is not None else profile.max_tokens,
            "streaming": streaming,
            "stream_usage": True,
            **kwargs,
        }

        if effective_timeout is not None:
            llm_kwargs["timeout"] = effective_timeout

        return ChatOpenAI(**llm_kwargs)  # type: ignore

    def create_invoker(
        self,
        llm_type: LLMType | str | None = None,
        *,
        streaming: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> "LLMInvoker":
        """
        创建带 usage tracking 的 LLM 调用器（推荐使用）

        Args:
            llm_type: LLM 类型 (fast/normal)
            streaming: 是否启用流式输出
            temperature: 温度参数（可选覆盖配置）
            max_tokens: 最大 tokens（可选覆盖配置）
            **kwargs: 其他 ChatOpenAI 参数

        Returns:
            LLMInvoker 实例
        """
        from app.llm.callbacks import get_usage_callbacks

        base_llm = self.create_llm(
            llm_type,
            streaming=streaming,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return LLMInvoker(
            provider=self,
            llm_type=llm_type,
            base_llm=base_llm,
            callbacks=get_usage_callbacks(),
            streaming=streaming,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=kwargs,
        )



class LLMInvoker:
    """
    LLM 调用器 - 封装 LLM 调用逻辑

    特性:
    - 自动附加 usage tracking callbacks
    - 支持动态模型选择（每次调用可选择不同模型）
    - 支持 tool calling
    - 支持流式输出
    - 韧性调用：指数退避重试 + 模型降级切换（配置见 config.yml resilience 段）

    使用方式:
        # 普通调用
        response = await invoker.ainvoke(messages)

        # 流式调用
        async for chunk in invoker.astream(messages):
            print(chunk.content)

        # 带 tools 调用
        response = await invoker.ainvoke_with_tools(messages, tools)
    """

    def __init__(
        self,
        provider: LLMProvider,
        llm_type: LLMType | str | None,
        base_llm: ChatOpenAI,
        callbacks: List[BaseCallbackHandler] | None = None,
        *,
        streaming: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_kwargs: dict | None = None,
    ):
        self._provider = provider
        self._llm_type = llm_type
        self._base_llm = base_llm
        self._callbacks = callbacks or []
        # 记录创建参数，用于模型降级时重建 LLM 实例
        self._streaming = streaming
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra_kwargs = dict(extra_kwargs or {})

    # ------------------------------------------------------------------
    # LLM 实例构建
    # ------------------------------------------------------------------
    def _make_llm(
        self,
        model: str,
        llm_type: LLMType | str | None = None,
        tools: list[Any] | None = None,
    ) -> ChatOpenAI:
        """按指定模型/层级构建 LLM 实例（支持降级重建）。"""
        llm = self._provider.create_llm(
            llm_type or self._llm_type,
            streaming=self._streaming,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **self._extra_kwargs,
        )
        llm = llm.bind(model=model)  # type: ignore
        if tools:
            llm = llm.bind(tools=tools)  # type: ignore
        return llm  # type: ignore

    def _get_llm_with_model(self, tools: list[Any] | None = None) -> ChatOpenAI:
        """
        获取绑定了随机模型的 LLM 实例

        每次调用都会随机选择一个模型，实现负载均衡
        """
        model = self._provider.pick_model(self._llm_type)
        return self._make_llm(model, self._llm_type, tools)

    def _prepare_config(self, kwargs: dict) -> dict:
        """准备调用配置，合并 callbacks"""
        # 提取已有的 callbacks
        call_callbacks = kwargs.pop("callbacks", None) or []
        config = kwargs.pop("config", None) or {}

        if isinstance(config, dict):
            config_callbacks = config.get("callbacks", []) or []
        else:
            config_callbacks = []

        # 合并所有 callbacks
        merged = list(call_callbacks) + list(config_callbacks) + self._callbacks
        if merged:
            kwargs["config"] = {"callbacks": merged}

        return kwargs

    # ------------------------------------------------------------------
    # 韧性事件上报（接入 telemetry）
    # ------------------------------------------------------------------
    def _on_resilience_event(self, event: Any) -> None:
        """韧性层事件回调：写入结构化日志。"""
        try:
            from app.telemetry.logger import log_structured_event

            log_structured_event(
                "llm_resilience",
                event.to_dict(),
            )
        except Exception:
            pass

    def _resilience_config(self):
        from app.config import settings

        return settings.resilience

    def _build_fallback_plan(self):
        """构建降级链与各层模型名。"""
        from app.llm.resilience import build_llm_type_chain

        cfg = self._resilience_config()
        chain = build_llm_type_chain(self._llm_type, cfg)
        model_names = {
            t: self._provider.get_profile(t).model_names or []
            for t in chain
        }
        return cfg, chain, model_names

    # ------------------------------------------------------------------
    # 非流式调用（带重试 + 降级）
    # ------------------------------------------------------------------
    async def ainvoke(self, messages: list, **kwargs: Any) -> Any:
        """异步调用 LLM（指数退避重试 + 模型降级）。"""
        kwargs = self._prepare_config(kwargs)
        return await self._invoke_resilient(messages, tools=None, **kwargs)

    async def ainvoke_with_tools(
        self, messages: list, tools: list, **kwargs: Any
    ) -> Any:
        """异步调用 LLM（带 tools，含重试 + 降级）。"""
        kwargs = self._prepare_config(kwargs)
        return await self._invoke_resilient(messages, tools=tools, **kwargs)

    async def _invoke_resilient(
        self,
        messages: list,
        tools: list | None = None,
        **kwargs: Any,
    ) -> Any:
        """韧性调用的统一实现（含成本熔断检查与记账）。"""
        from app.llm.resilience import call_with_fallback

        cfg, chain, model_names = self._build_fallback_plan()

        # ==========================================================================
        # P0 安全: 成本熔断（调用前检查）
        # ==========================================================================
        session_id = self._session_id()
        cost_check = await self._cost_check(session_id)
        if not cost_check.allowed:
            from app.llm.resilience import LLMResilienceError

            raise LLMResilienceError(
                f"Cost guard: {cost_check.reason}",
                error_type="COST_LIMIT",
                retryable=False,
                attempts=0,
                models_tried=[],
            )

        # 未启用韧性：直接调用（随机模型负载均衡）
        if not cfg.fallback.enabled and cfg.retry.max_retries == 0:
            response = await self._get_llm_with_model(tools=tools).ainvoke(
                messages, **kwargs
            )
            await self._record_usage(session_id, response)
            return response

        async def call(llm_type: str, model: str) -> Any:
            llm = self._make_llm(model, llm_type, tools)
            return await llm.ainvoke(messages, **kwargs)

        response = await call_with_fallback(
            call,
            chain,
            model_names,
            cfg,
            on_event=self._on_resilience_event,
        )
        await self._record_usage(session_id, response)
        return response

    # ------------------------------------------------------------------
    # 成本熔断辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _session_id() -> Optional[str]:
        """从 LLM 上下文取会话 ID（conversation_id 优先）。"""
        try:
            from app.llm.context import get_llm_context

            ctx = get_llm_context()
            if ctx:
                return ctx.conversation_id or ctx.request_id
        except Exception:
            pass
        return None

    async def _cost_check(self, session_id: Optional[str]):
        """成本熔断检查（未启用/无会话时放行）。"""
        try:
            from app.security.cost_guard import cost_guard

            if session_id:
                return await cost_guard.check(session_id)
        except Exception:
            pass
        from app.security.cost_guard import CostCheckResult

        return CostCheckResult(True, "ok", "cost guard unavailable")

    async def _record_usage(self, session_id: Optional[str], response: Any) -> None:
        """调用后按 usage_metadata 记账。"""
        if not session_id:
            return
        try:
            input_tokens = output_tokens = None
            metadata = getattr(response, "usage_metadata", None)
            if metadata:
                if isinstance(metadata, dict):
                    input_tokens = metadata.get("input_tokens")
                    output_tokens = metadata.get("output_tokens")
                else:
                    input_tokens = getattr(metadata, "input_tokens", None)
                    output_tokens = getattr(metadata, "output_tokens", None)
            if input_tokens is None and output_tokens is None:
                return
            from app.security.cost_guard import cost_guard

            await cost_guard.record(session_id, input_tokens or 0, output_tokens or 0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 流式调用（首块前重试，出流后不重试避免内容重复）
    # ------------------------------------------------------------------
    def astream(self, messages: list, **kwargs: Any) -> AsyncIterator[Any]:
        """流式调用 LLM（仅首块前失败会重试）。"""
        kwargs = self._prepare_config(kwargs)
        return self._astream_resilient(messages, tools=None, **kwargs)

    async def astream_with_tools(
        self, messages: list, tools: list, **kwargs: Any
    ) -> AsyncIterator[Any]:
        """流式调用 LLM（带 tools，仅首块前失败会重试）。"""
        kwargs = self._prepare_config(kwargs)
        async for chunk in self._astream_resilient(messages, tools=tools, **kwargs):
            yield chunk

    async def _astream_resilient(
        self,
        messages: list,
        tools: list | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        from app.llm.resilience import async_retry, LLMResilienceError

        cfg, chain, model_names = self._build_fallback_plan()

        # P0 安全: 成本熔断（流式调用前检查）
        session_id = self._session_id()
        cost_check = await self._cost_check(session_id)
        if not cost_check.allowed:
            raise LLMResilienceError(
                f"Cost guard: {cost_check.reason}",
                error_type="COST_LIMIT",
                retryable=False,
                attempts=0,
                models_tried=[],
            )

        if not cfg.fallback.enabled and cfg.retry.max_retries == 0:
            async for chunk in self._get_llm_with_model(tools=tools).astream(
                messages, **kwargs
            ):
                yield chunk
            return

        # 打开流并消费首个 chunk（HTTP 请求在此触发，失败可重试/降级）
        # 首块之后不再重试，避免 SSE 内容重复
        async def open_and_first(llm_type: str, model: str):
            stream = self._make_llm(model, llm_type, tools).astream(messages, **kwargs)
            iterator = stream.__aiter__()
            try:
                first = await iterator.__anext__()
            except StopAsyncIteration:
                return iterator, None
            return iterator, first

        fallback_steps = 0
        last_err: BaseException | None = None

        for llm_type in chain:
            models = model_names.get(llm_type) or []
            for model in models:
                try:
                    iterator, first = await async_retry(
                        lambda t=llm_type, m=model: open_and_first(t, m),
                        cfg.retry,
                        on_event=self._on_resilience_event,
                        llm_type=llm_type,
                        model=model,
                    )
                    if first is not None:
                        yield first
                    async for chunk in iterator:
                        yield chunk
                    return
                except LLMResilienceError as e:
                    last_err = e
                    # 模型降级
                except Exception as e:
                    # 流中途失败：不重试（已产生部分输出）
                    raise

            fallback_steps += 1
            if fallback_steps >= cfg.fallback.max_fallback_steps:
                break

        if isinstance(last_err, LLMResilienceError):
            raise last_err
        raise LLMResilienceError(
            "Streaming LLM call failed after all fallbacks",
            error_type=type(last_err).__name__ if last_err else "Unknown",
            retryable=False,
            attempts=1,
            models_tried=[],
            cause=last_err,
        )