from __future__ import annotations


# app/config/tokenizer_config.py
"""
Token 计数与滑动窗口配置。
"""

from typing import Optional

from pydantic import BaseModel, Field


class TokenizerConfig(BaseModel):
    """
    tiktoken 计数与滑动窗口配置。

    Attributes:
        model: 用于选择 tiktoken 编码器的模型名（仅影响编码器选择）
        encoding_name: 显式指定 tiktoken 编码器名（优先于 model）
        token_budget: 单次 LLM 请求的 token 预算（滑动窗口上限）
        compression_token_ratio: 触发摘要压缩的 token 阈值系数，
            当未压缩消息估算 token > token_budget * ratio 时触发压缩
        min_uncompressed_messages: 压缩后至少保留的未压缩消息条数
    """

    model: str = "gpt-4o"
    encoding_name: Optional[str] = None
    token_budget: int = Field(default=16000, ge=512)
    compression_token_ratio: float = Field(default=0.8, ge=0.1, le=2.0)
    min_uncompressed_messages: int = Field(default=6, ge=2)
