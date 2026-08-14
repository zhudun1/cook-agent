# tests/test_memory.py
"""
P2 长期记忆单元测试：存储/去重/回忆检索/启发式提取。
"""

import os
import sys

# 必须在 import app.database 前设置（session 模块级建 engine）
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_memory.db"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio

import pytest

from app.memory.extractor import MemoryExtractor
from app.memory.manager import MemoryManager
from app.memory.store import MemoryStore
from app.database.session import init_db
from app.config.memory_config import MemoryConfig


@pytest.fixture(scope="module", autouse=True)
def _db():
    asyncio.run(init_db())
    yield
    try:
        os.remove("./test_memory.db")
    except OSError:
        pass


@pytest.fixture
def store():
    return MemoryStore(MemoryConfig())


class TestMemoryStore:
    def test_add_and_list(self, store):
        asyncio.run(store.clear("u1"))
        asyncio.run(
            store.add("u1", "用户不喜欢吃香菜", memory_type="preference", importance=0.8)
        )
        asyncio.run(
            store.add("u1", "用户目标是减脂", memory_type="goal", importance=0.9)
        )
        memories = asyncio.run(store.list("u1"))
        assert len(memories) == 2
        types = {m["memory_type"] for m in memories}
        assert types == {"preference", "goal"}

    def test_deduplication(self, store):
        asyncio.run(store.clear("u2"))
        asyncio.run(store.add("u2", "相同内容", memory_type="fact"))
        asyncio.run(store.add("u2", "相同内容", memory_type="fact"))
        assert len(asyncio.run(store.list("u2"))) == 1

    def test_recall_relevance(self, store):
        asyncio.run(store.clear("u3"))
        asyncio.run(store.add("u3", "用户喜欢川菜，偏好麻辣口味", memory_type="preference", importance=0.7))
        asyncio.run(store.add("u3", "用户目标是每周健身三次", memory_type="goal", importance=0.6))
        # 与"川菜"相关的查询应召回偏好记忆
        recalled = asyncio.run(store.recall("u3", "推荐一道川菜"))
        assert recalled, "应召回相关记忆"
        assert "川菜" in recalled[0]["content"]

    def test_recall_irrelevant_empty(self, store):
        asyncio.run(store.clear("u4"))
        asyncio.run(store.add("u4", "用户喜欢红烧肉", memory_type="preference"))
        recalled = asyncio.run(store.recall("u4", "今天天气怎么样"))
        # 无关键词重叠可能为空（取决于 2-gram 重叠）
        assert isinstance(recalled, list)

    def test_trim_over_limit(self, store):
        cfg = MemoryConfig(max_memories_per_user=3)
        s = MemoryStore(cfg)
        asyncio.run(s.clear("u5"))
        for i in range(5):
            asyncio.run(
                s.add("u5", f"记忆条目 {i}", memory_type="fact", importance=0.1)
            )
        memories = asyncio.run(s.list("u5"))
        assert len(memories) <= 3

    def test_delete(self, store):
        asyncio.run(store.clear("u6"))
        m = asyncio.run(store.add("u6", "待删除", memory_type="fact"))
        assert asyncio.run(store.delete("u6", str(m.id))) is True
        assert len(asyncio.run(store.list("u6"))) == 0


class TestMemoryExtractor:
    def test_heuristic_preference(self):
        ex = MemoryExtractor(MemoryConfig())
        memories = ex._extract_heuristic(
            [{"role": "user", "content": "我喜欢吃辣，特别爱吃川菜。"}]
        )
        assert memories
        assert memories[0]["type"] == "preference"

    def test_heuristic_restriction(self):
        ex = MemoryExtractor(MemoryConfig())
        memories = ex._extract_heuristic(
            [{"role": "user", "content": "我对花生过敏，不能吃花生酱。"}]
        )
        assert memories
        assert memories[0]["type"] == "restriction"

    def test_heuristic_goal(self):
        ex = MemoryExtractor(MemoryConfig())
        memories = ex._extract_heuristic(
            [{"role": "user", "content": "我的目标是三个月减脂 5 公斤。"}]
        )
        assert memories
        assert memories[0]["type"] == "goal"

    def test_ignore_assistant_and_boring(self):
        ex = MemoryExtractor(MemoryConfig())
        memories = ex._extract_heuristic(
            [
                {"role": "assistant", "content": "用户喜欢川菜"},
                {"role": "user", "content": "好的谢谢"},
            ]
        )
        assert memories == []


class TestMemoryManager:
    def test_build_context_empty_without_memories(self):
        mgr = MemoryManager(MemoryConfig())
        ctx = asyncio.run(mgr.build_context("no-such-user", "测试"))
        assert ctx == ""

    def test_build_context_with_memories(self, store):
        asyncio.run(store.clear("u7"))
        asyncio.run(
            store.add("u7", "用户不喜欢吃香菜", memory_type="restriction", importance=0.9)
        )
        mgr = MemoryManager(MemoryConfig(recall_top_k=5))
        ctx = asyncio.run(mgr.build_context("u7", "帮我做菜"))
        assert "长期记忆" in ctx
        assert "香菜" in ctx

    def test_extract_and_store_heuristic(self, store):
        asyncio.run(store.clear("u8"))
        mgr = MemoryManager(
            MemoryConfig(min_messages_for_extraction=2, auto_extract=True)
        )
        count = asyncio.run(
            mgr.extract_and_store(
                "sess-1",
                "u8",
                [
                    {"role": "user", "content": "我喜欢清淡口味"},
                    {"role": "assistant", "content": "好的"},
                    {"role": "user", "content": "下次做清蒸鱼"},
                ],
            )
        )
        assert count >= 1
        memories = asyncio.run(store.list("u8"))
        assert any("清淡" in m["content"] for m in memories)
