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

import asyncio


from app.agent.event_stream import EventStreamStore


class TestEventStreamStore:
    def test_append_and_get_all(self):
        async def main():
            store = EventStreamStore()
            await store.append("evt-a", 'data: {"type": "session"}\n\n')
            await store.append("evt-a", 'data: {"type": "text", "content": "hi"}\n\n')
            events = await store.get_events("evt-a", after_seq=-1)
            assert len(events) == 2
            assert events[0]["seq"] == 0
            assert events[1]["seq"] == 1
            assert "text" in events[1]["event"]

        asyncio.run(main())

    def test_incremental_pull(self):
        async def main():
            store = EventStreamStore()
            for i in range(5):
                await store.append("evt-b", f"event-{i}")
            # 断点恢复：已收到 seq 0-2，拉取 2 之后
            events = await store.get_events("evt-b", after_seq=2)
            assert [e["seq"] for e in events] == [3, 4]
            assert await store.next_seq("evt-b") == 5

        asyncio.run(main())

    def test_unknown_session(self):
        async def main():
            store = EventStreamStore()
            assert await store.get_events("evt-nope") == []
            assert await store.has_stream("evt-nope") is False

        asyncio.run(main())

    def test_clear(self):
        async def main():
            store = EventStreamStore()
            await store.append("evt-c", "x")
            assert await store.has_stream("evt-c")
            await store.clear("evt-c")
            assert not await store.has_stream("evt-c")
            assert await store.get_events("evt-c") == []

        asyncio.run(main())

    def test_maxlen_cap(self):
        async def main():
            store = EventStreamStore(max_events_per_session=3)
            for i in range(10):
                await store.append("evt-d", f"e{i}")
            events = await store.get_events("evt-d")
            # 只保留最近 3 条
            assert len(events) == 3
            assert [e["seq"] for e in events] == [7, 8, 9]

        asyncio.run(main())
