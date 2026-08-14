# tests/test_event_stream.py
"""
P3 会话断点恢复：事件流存储增量拉取测试。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 预注册 app.agent 包：跳过其 __init__ 对 langchain 等重依赖的顶层导入
import types

_agent_pkg = types.ModuleType("app.agent")
_agent_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "app", "agent")]
sys.modules["app.agent"] = _agent_pkg

import pytest

from app.agent.event_stream import EventStreamStore


class TestEventStreamStore:
    def test_append_and_get_all(self):
        store = EventStreamStore()
        store.append("s1", 'data: {"type": "session"}\n\n')
        store.append("s1", 'data: {"type": "text", "content": "hi"}\n\n')
        events = store.get_events("s1", after_seq=-1)
        assert len(events) == 2
        assert events[0]["seq"] == 0
        assert events[1]["seq"] == 1
        assert "text" in events[1]["event"]

    def test_incremental_pull(self):
        store = EventStreamStore()
        for i in range(5):
            store.append("s1", f"event-{i}")
        # 断点恢复：已收到 seq 0-2，拉取 2 之后
        events = store.get_events("s1", after_seq=2)
        assert [e["seq"] for e in events] == [3, 4]
        assert store.next_seq("s1") == 5

    def test_unknown_session(self):
        store = EventStreamStore()
        assert store.get_events("nope") == []
        assert store.has_stream("nope") is False

    def test_clear(self):
        store = EventStreamStore()
        store.append("s1", "x")
        assert store.has_stream("s1")
        store.clear("s1")
        assert not store.has_stream("s1")
        assert store.get_events("s1") == []

    def test_maxlen_cap(self):
        store = EventStreamStore(max_events_per_session=3)
        for i in range(10):
            store.append("s1", f"e{i}")
        events = store.get_events("s1")
        # 只保留最近 3 条
        assert len(events) == 3
        assert [e["seq"] for e in events] == [7, 8, 9]
