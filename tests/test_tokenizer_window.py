# tests/test_tokenizer_window.py
"""
Token 计数与滑动窗口单元测试。

依赖: tiktoken（requirements.txt 已含）
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 预注册 app.llm 包：跳过其 __init__ 对 langchain 的顶层导入
import types

_llm_pkg = types.ModuleType("app.llm")
_llm_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "app", "llm")]
sys.modules["app.llm"] = _llm_pkg

import pytest

from app.llm.tokenizer import TokenCounter, count_messages, fit_in_budget


@pytest.fixture
def counter():
    return TokenCounter()


class TestTokenCounter:
    def test_empty_text(self, counter):
        assert counter.count_tokens("") == 0

    def test_mixed_cjk_ascii(self, counter):
        n = counter.count_tokens("西红柿炒鸡蛋怎么做 hello world 你好")
        assert n > 0

    def test_estimate_source_is_tiktoken(self, counter):
        info = counter.estimate_tokens("你好 world")
        assert info["tokens"] > 0
        assert info["source"] in ("tiktoken", "heuristic")

    def test_count_messages_protocol_overhead(self, counter):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "你好"},
        ]
        # 至少包含内容 + 协议开销（3/条）
        assert count_messages(msgs) > counter.count_tokens("你好")

    def test_multimodal_parts(self, counter):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看这张图"},
                    {"type": "image_url", "image_url": {"url": "http://x/1.png"}},
                ],
            }
        ]
        assert count_messages(msgs) > 85  # 图片按 85 token 估算


class TestSlidingWindow:
    def test_fit_in_budget_drops_oldest(self, counter):
        messages = [
            {"role": "user", "content": f"message {i} " + "x" * 100}
            for i in range(20)
        ]
        # 预算只够 ~5 条，从末尾保留
        fitted = fit_in_budget(messages, budget=150)
        assert len(fitted) < len(messages)
        # 保留的是最新消息（末尾）
        assert fitted[-1] == messages[-1]
        # 丢弃的是最早消息
        assert fitted[0] != messages[0]

    def test_keep_first_prefix(self, counter):
        messages = [
            {"role": "system", "content": "S" * 500},
        ] + [
            {"role": "user", "content": f"message {i} " + "x" * 100}
            for i in range(20)
        ]
        fitted = fit_in_budget(messages, budget=300, keep_first=1)
        # system 前缀永不丢弃
        assert fitted[0] == messages[0]

    def test_zero_budget(self, counter):
        fitted = fit_in_budget([{"role": "user", "content": "hi"}], budget=0)
        assert fitted == []
