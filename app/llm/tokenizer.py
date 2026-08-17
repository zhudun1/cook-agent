from __future__ import annotations


# app/llm/tokenizer.py
"""
tiktoken 实时 Token 计数

统一封装 tiktoken 计数逻辑，提供：
1. 线程安全的编码器缓存（同一模型只加载一次编码器）
2. 无网络环境下的字符级启发式回退（中英混合按比例估算）
3. 结构化消息（OpenAI 格式）的 token 估算

设计要点：
- TokenCounter 为进程级单例，编码器懒加载
- 估算结果带 `source` 字段（"tiktoken" / "heuristic"），便于可观测性
- 所有计数函数对任意文本输入安全（不抛异常，失败回退启发式）
"""


import logging
import re
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 常见模型 -> tiktoken 编码器名称
MODEL_ENCODING_MAP = {
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
}

# 无编码器可用时的启发式估算参数
# 中文字符平均 ~1.5 token/字，英文 ~0.25 token/字母（由 cl100k 统计近似）
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_ASCII_RE = re.compile(r"[A-Za-z0-9]")
CJK_TOKENS_PER_CHAR = 1.4
ASCII_TOKENS_PER_CHAR = 0.28
OTHER_TOKENS_PER_CHAR = 0.5


class TokenCounter:
    """
    线程安全的 tiktoken Token 计数器。

    用法::

        counter = TokenCounter()
        n = counter.count_tokens("你好 world")
        n2 = counter.count_messages([{"role": "user", "content": "hi"}])
    """

    def __init__(self, model: str | None = None, encoding_name: str | None = None):
        self._model = model or "gpt-4o"
        self._encoding_name = encoding_name
        self._lock = threading.Lock()
        self._encoding = None
        self._load_failed = False
        self._load_attempted = False

    # ------------------------------------------------------------------
    # 编码器加载
    # ------------------------------------------------------------------
    def _load_encoding(self):
        """懒加载 tiktoken 编码器（仅一次，失败则回退启发式）。"""
        if self._load_attempted:
            return
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            try:
                import tiktoken

                name = self._encoding_name or MODEL_ENCODING_MAP.get(
                    self._model, "o200k_base"
                )
                self._encoding = tiktoken.get_encoding(name)
                logger.debug("TokenCounter loaded encoding=%s", name)
            except Exception as e:  # pragma: no cover - 环境无 tiktoken 时
                self._load_failed = True
                logger.warning(
                    "tiktoken unavailable (%s), falling back to heuristic counting",
                    e,
                )

    # ------------------------------------------------------------------
    # 计数 API
    # ------------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        """统计一段文本的 token 数。"""
        if not text:
            return 0
        self._load_encoding()
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text))
            except Exception as e:
                logger.debug("tiktoken encode failed (%s), using heuristic", e)
        return self._heuristic_count(text)

    def estimate_tokens(self, text: str) -> dict:
        """估算 token 数并返回来源信息（用于可观测性）。"""
        self._load_encoding()
        if self._encoding is not None:
            try:
                return {"tokens": len(self._encoding.encode(text)), "source": "tiktoken"}
            except Exception:
                pass
        return {"tokens": self._heuristic_count(text), "source": "heuristic"}

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """
        统计 OpenAI 格式消息列表的总 token 数。

        兼容 content 为 str / list（多模态 parts）以及 tool_calls。
        """
        total = 0
        for msg in messages:
            total += self._count_single_message(msg)
        # 每条消息的协议开销
        total += 3 * len(messages)
        return total

    def _count_single_message(self, msg: Dict[str, Any]) -> int:
        n = 0
        role = msg.get("role", "")
        if role:
            n += self.count_tokens(role)

        content = msg.get("content")
        if isinstance(content, str):
            n += self.count_tokens(content)
        elif isinstance(content, list):
            # 多模态 content parts
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and part.get("text"):
                        n += self.count_tokens(part["text"])
                    elif part.get("type") == "image_url":
                        # 图片按固定 token 估算（OpenAI 惯例 85 基础 + 缩放，简化处理）
                        n += 85
                elif isinstance(part, str):
                    n += self.count_tokens(part)

        # tool_calls
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    n += self.count_tokens(fn.get("name", ""))
                    n += self.count_tokens(fn.get("arguments", ""))
                else:
                    n += self.count_tokens(str(tc))

        return n

    def fit_in_budget(
        self,
        messages: List[Dict[str, Any]],
        budget: int,
        *,
        keep_first: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        滑动窗口截断：从后往前保留消息，使总 token 数不超过 budget。

        从 **末尾**（最新消息）开始保留，超出的最早消息被丢弃。
        可选 `keep_first`：强制保留前 N 条消息（如 system / 摘要消息），
        这些消息不参与截断判断。

        Args:
            messages: OpenAI 格式消息列表
            budget: token 预算
            keep_first: 必须保留的前缀消息条数（不参与截断）

        Returns:
            截断后的消息列表（保序）
        """
        if budget <= 0:
            return messages[:keep_first]

        prefix = messages[:keep_first]
        tail = messages[keep_first:]

        # 从末尾累计 token
        total = self.count_messages(prefix)
        kept: List[Dict[str, Any]] = []
        for msg in reversed(tail):
            msg_tokens = self._count_single_message(msg) + 3
            if total + msg_tokens > budget and kept:
                # 再往前加会超预算：停止（至少保留最后一条）
                break
            kept.append(msg)
            total += msg_tokens

        kept.reverse()
        result = prefix + kept
        if len(result) < len(messages):
            logger.debug(
                "Sliding window truncated %d messages (budget=%d tokens)",
                len(messages) - len(result),
                budget,
            )
        return result

    # ------------------------------------------------------------------
    # 启发式回退
    # ------------------------------------------------------------------
    @staticmethod
    def _heuristic_count(text: str) -> int:
        cjk = len(_CJK_RE.findall(text))
        ascii_chars = len(_ASCII_RE.findall(text))
        other = max(0, len(text) - cjk - ascii_chars)
        return int(
            cjk * CJK_TOKENS_PER_CHAR
            + ascii_chars * ASCII_TOKENS_PER_CHAR
            + other * OTHER_TOKENS_PER_CHAR
        ) or len(text) // 4


# 进程级单例（默认按 gpt-4o / o200k_base 估算，与主流 OpenAI 兼容模型一致）
_token_counter: Optional[TokenCounter] = None
_counter_lock = threading.Lock()


def get_token_counter(model: str | None = None) -> TokenCounter:
    """获取全局 TokenCounter 单例（可指定模型）。"""
    global _token_counter
    if _token_counter is None:
        with _counter_lock:
            if _token_counter is None:
                _token_counter = TokenCounter(model=model)
    return _token_counter


def count_tokens(text: str) -> int:
    """便捷函数：统计文本 token 数。"""
    return get_token_counter().count_tokens(text)


def count_messages(messages: List[Dict[str, Any]]) -> int:
    """便捷函数：统计消息列表 token 数。"""
    return get_token_counter().count_messages(messages)


def fit_in_budget(
    messages: List[Dict[str, Any]],
    budget: int,
    *,
    keep_first: int = 0,
) -> List[Dict[str, Any]]:
    """便捷函数：滑动窗口截断消息列表。"""
    return get_token_counter().fit_in_budget(messages, budget, keep_first=keep_first)
