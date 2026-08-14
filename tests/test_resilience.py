# tests/test_resilience.py
"""
LLM 韧性层单元测试：指数退避重试、模型降级、非重试错误传播。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 预注册 app.llm 包：跳过其 __init__ 对 langchain 的顶层导入，
# 使本测试在未安装 langchain 的轻量环境与完整环境均可运行
import types

_llm_pkg = types.ModuleType("app.llm")
_llm_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "app", "llm")]
sys.modules["app.llm"] = _llm_pkg

import asyncio

import pytest

from app.llm.resilience import (
    LLMResilienceError,
    async_retry,
    build_llm_type_chain,
    call_with_fallback,
    is_retryable_error,
)
from app.config.resilience_config import FallbackConfig, ResilienceConfig, RetryConfig


def _fast_cfg():
    return RetryConfig(max_retries=3, base_delay_seconds=0.01, jitter=False)


class TestRetry:
    def test_retryable_timeout_success(self):
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("API timeout")
            return "ok"

        async def main():
            return await async_retry(flaky, _fast_cfg())

        assert asyncio.run(main()) == "ok"
        assert calls["n"] == 3

    def test_non_retryable_propagates_immediately(self):
        async def bad():
            raise ValueError("bad request")

        async def main():
            return await async_retry(bad, _fast_cfg())

        with pytest.raises(LLMResilienceError) as exc_info:
            asyncio.run(main())
        assert exc_info.value.retryable is False
        assert exc_info.value.attempts == 1

    def test_retries_exhausted(self):
        calls = {"n": 0}

        async def always_fail():
            calls["n"] += 1
            raise TimeoutError("still failing")

        async def main():
            return await async_retry(always_fail, _fast_cfg())

        with pytest.raises(LLMResilienceError) as exc_info:
            asyncio.run(main())
        assert exc_info.value.attempts == 4  # max_retries + 1
        assert exc_info.value.retryable is True


class TestFallback:
    def test_chain_order(self):
        chain = build_llm_type_chain("fast", ResilienceConfig())
        assert chain[0] == "fast"
        assert chain[1] == "normal"

    def test_degrades_to_next_tier(self):
        async def builder(llm_type, model):
            if llm_type == "fast":
                raise TimeoutError("rate limit")
            return f"ok-from-{model}"

        async def main():
            cfg = ResilienceConfig(
                retry=RetryConfig(max_retries=1, base_delay_seconds=0.01, jitter=False),
                fallback=FallbackConfig(
                    fallback_llm_types=["fast", "normal"], max_fallback_steps=1
                ),
            )
            return await call_with_fallback(
                builder,
                ["fast", "normal"],
                {"fast": ["m1"], "normal": ["m2"]},
                cfg,
            )

        assert asyncio.run(main()) == "ok-from-m2"

    def test_all_fail_raises_structured(self):
        async def builder(llm_type, model):
            raise TimeoutError("down")

        async def main():
            cfg = ResilienceConfig(
                retry=RetryConfig(max_retries=1, base_delay_seconds=0.01, jitter=False),
            )
            return await call_with_fallback(
                builder,
                ["fast"],
                {"fast": ["m1"]},
                cfg,
            )

        with pytest.raises(LLMResilienceError) as exc_info:
            asyncio.run(main())
        assert exc_info.value.models_tried == ["m1"]  # 每层记录一次模型
        assert exc_info.value.attempts == 2  # max_retries=1 -> 2 次尝试


class TestRetryableClassification:
    def test_keyword_match(self):
        cfg = RetryConfig(retryable_errors=["RateLimitError"])
        assert is_retryable_error(RuntimeError("RateLimitError: slow down"), cfg)
        assert not is_retryable_error(RuntimeError("something else"), cfg)

    def test_builtin_network(self):
        cfg = RetryConfig(retryable_errors=[])
        assert is_retryable_error(ConnectionError("refused"), cfg)
        assert is_retryable_error(asyncio.TimeoutError(), cfg)
