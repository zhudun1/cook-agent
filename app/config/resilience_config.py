from __future__ import annotations


# app/config/resilience_config.py
"""
LLM 调用韧性配置：指数退避重试、模型降级、超时。
"""

from typing import List

from pydantic import BaseModel, Field


class RetryConfig(BaseModel):
    """指数退避重试配置。"""

    max_retries: int = Field(default=3, ge=0)
    base_delay_seconds: float = Field(default=1.0, gt=0)
    max_delay_seconds: float = Field(default=30.0, gt=0)
    exponential_factor: float = Field(default=2.0, gt=1.0)
    jitter: bool = True
    # 可重试的异常类型关键字（匹配异常类名或消息）
    retryable_errors: List[str] = Field(
        default_factory=lambda: [
            "TimeoutError",
            "APITimeoutError",
            "RateLimitError",
            "ServiceUnavailableError",
            "InternalServerError",
            "BadGatewayError",
            "ConnectionError",
            "openai.APIConnectionError",
        ]
    )


class FallbackConfig(BaseModel):
    """
    模型降级配置。

    Attributes:
        enabled: 是否启用降级
        retry_models_within_profile: 同一 profile 内模型列表逐个重试（负载均衡 + 降级）
        fallback_llm_types: 跨层降级链（如 ["fast", "normal"]，
            表示 fast 全失败后降级到 normal）
        max_fallback_steps: 最大降级步数
    """

    enabled: bool = True
    retry_models_within_profile: bool = True
    fallback_llm_types: List[str] = Field(
        default_factory=lambda: ["fast", "normal", "vision"]
    )
    max_fallback_steps: int = Field(default=2, ge=1)


class ToolExecutionConfig(BaseModel):
    """工具调用超时与错误配置。"""

    # 单个工具的默认超时（秒），0 表示不限制
    default_timeout_seconds: float = Field(default=30.0, ge=0)
    # 工具级超时覆盖（tool_name -> seconds）
    timeout_overrides: dict = Field(default_factory=dict)


class ResilienceConfig(BaseModel):
    """LLM 韧性总配置。"""

    retry: RetryConfig = Field(default_factory=RetryConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    tools: ToolExecutionConfig = Field(default_factory=ToolExecutionConfig)
