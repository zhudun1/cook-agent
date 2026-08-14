from __future__ import annotations


# app/llm/window.py
"""
滑动窗口上下文管理（基于 token 预算）

在 tiktoken 计数之上提供高层窗口策略：
1. `ContextWindow` - 按 token 预算组装消息：system + 摘要 + 近期消息（滑动截断）
2. 截断决策记录（dropped 消息数、预算占用明细），供轨迹与可观测性使用

与压缩器协作：
- 压缩器负责把**最早**的消息折叠成摘要（保留语义）
- 窗口负责在**单次请求**内把放不下的近期消息从旧到新丢弃（滑动）
- 两者组合解决长对话记忆退化：摘要兜住长期记忆，窗口兜住单次预算
"""


import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.llm.tokenizer import get_token_counter, TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class WindowStats:
    """一次窗口组装的可观测性数据"""

    total_budget: int
    system_tokens: int
    summary_tokens: int
    history_tokens: int
    current_tokens: int
    total_tokens: int
    dropped_messages: int = 0
    source: str = "tiktoken"  # tiktoken / heuristic

    def to_dict(self) -> dict:
        return {
            "total_budget": self.total_budget,
            "system_tokens": self.system_tokens,
            "summary_tokens": self.summary_tokens,
            "history_tokens": self.history_tokens,
            "current_tokens": self.current_tokens,
            "total_tokens": self.total_tokens,
            "dropped_messages": self.dropped_messages,
            "source": self.source,
        }


class ContextWindow:
    """
    基于 token 预算的滑动窗口组装器。

    用法::

        window = ContextWindow(token_budget=16000, keep_system_messages=2)
        messages, stats = window.build(
            system_prompt="...",
            history_summary="...",
            history=[{...}, ...],      # 已压缩之后的近期消息（旧 -> 新）
            current_message="...",
        )
    """

    def __init__(
        self,
        token_budget: int = 16000,
        keep_system_messages: int = 2,
        counter: Optional[TokenCounter] = None,
    ):
        self.token_budget = token_budget
        # system / 摘要 / 用户画像消息位于最前，截断时永不丢弃
        self.keep_system_messages = keep_system_messages
        self._counter = counter or get_token_counter()

    # ------------------------------------------------------------------
    def build(
        self,
        system_prompt: str,
        history_summary: Optional[str],
        history: List[Dict[str, Any]],
        current_message: str,
        extra_system: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], WindowStats]:
        """
        组装窗口内消息。

        Args:
            system_prompt: 基础 system prompt
            history_summary: 已压缩的历史摘要（可为 None）
            history: 近期未压缩消息（旧 -> 新，OpenAI 格式）
            current_message: 当前用户消息
            extra_system: 追加的 system 上下文（如 RAG 检索内容）

        Returns:
            (messages, stats)
        """
        messages: List[Dict[str, Any]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_summary:
            messages.append(
                {"role": "system", "content": f"## 历史对话摘要\n{history_summary}"}
            )
        if extra_system:
            messages.append({"role": "system", "content": extra_system})

        # 用户画像等已在 system_prompt 内拼好；system 消息数即固定前缀
        prefix_count = len(messages)

        # 当前消息
        messages.append({"role": "user", "content": current_message})

        # 滑动窗口：把 history 插到 system 与当前消息之间，从旧到新截断
        prefix = messages[:prefix_count]
        suffix = messages[prefix_count:]  # 当前消息

        budget_for_history = max(0, self.token_budget - self._count(prefix) - self._count(suffix))

        kept_history: List[Dict[str, Any]] = []
        total = 0
        dropped = 0
        # 从旧到新累计，超过预算即丢弃更旧的（滑动窗口语义）
        for msg in history:
            tokens = self._counter._count_single_message(msg) + 3
            if total + tokens > budget_for_history and kept_history:
                dropped += 1
                continue
            kept_history.append(msg)
            total += tokens

        messages = prefix + kept_history + suffix

        stats = WindowStats(
            total_budget=self.token_budget,
            system_tokens=self._count(prefix),
            summary_tokens=self._counter.count_tokens(history_summary or ""),
            history_tokens=self._count(kept_history),
            current_tokens=self._count(suffix),
            total_tokens=self._count(messages),
            dropped_messages=dropped,
            source=self._counter.estimate_tokens("x")["source"],
        )

        if dropped:
            logger.info(
                "Sliding window dropped %d history messages "
                "(budget=%d, used=%d)",
                dropped,
                self.token_budget,
                stats.total_tokens,
            )
        return messages, stats

    def _count(self, messages: List[Dict[str, Any]]) -> int:
        return self._counter.count_messages(messages)
