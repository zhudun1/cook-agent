# app/config/memory_config.py
"""
P2 长期记忆配置。

YAML 配置段: memory
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    """跨会话长期记忆配置。"""

    enabled: bool = True
    # 单用户记忆条目上限（超出按 importance 裁剪最旧/最低）
    max_memories_per_user: int = Field(default=200, ge=1)
    # 提取用 LLM 层级
    extraction_llm_type: str = "fast"
    # 回忆检索返回条数
    recall_top_k: int = Field(default=5, ge=1, le=50)
    # 回忆的最低重要性阈值（0.0 表示不过滤）
    recall_importance_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    # 对话结束后是否自动提取记忆
    auto_extract: bool = True
    # 自动提取的最小消息对数量（少于则不提取）
    min_messages_for_extraction: int = Field(default=4, ge=2)
    # 提取触发关键词（启发式回退用）
    extraction_keywords: List[str] = Field(
        default_factory=lambda: [
            "喜欢", "讨厌", "爱吃", "不吃", "忌口", "过敏",
            "目标是", "希望", "想要", "需要", "每周", "每天",
            "减肥", "减脂", "增肌", "控糖", "高蛋白", "低卡",
        ]
    )
