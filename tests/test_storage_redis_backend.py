# tests/test_storage_redis_backend.py
"""
RedisBackend 行为测试（用 fakeredis，无需真实 Redis 服务）。
验证统一后端在 Redis 语义下的数据面一致性。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio

# 预注册 app.agent 包（避免 3.9 下 __init__ 导入 langchain/types 语法问题）
import types

_agent_pkg = types.ModuleType("app.agent")
_agent_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "app", "agent")]
sys.modules["app.agent"] = _agent_pkg

_sec_pkg = types.ModuleType("app.security")
_sec_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "app", "security")]
sys.modules["app.security"] = _sec_pkg

import pytest

from app.storage.backend import RedisBackend


def _make_redis_backend():
    """在运行中的事件循环内创建（fakeredis 需要当前 loop）。"""
    import fakeredis.aioredis

    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    backend = RedisBackend.__new__(RedisBackend)
    backend._redis = fake
    return backend


class TestRedisBackend:
    def test_hash_ops(self):
        async def main():
            b = _make_redis_backend()
            assert await b.hincrby("h", "f1", 1) == 1
            assert await b.hincrby("h", "f1", 5) == 6
            assert await b.hgetall("h") == {"f1": "6"}
            await b.hset("h", "f2", "x")
            assert (await b.hgetall("h"))["f2"] == "x"
            assert await b.hdel("h", "f1") == 1
            assert "f1" not in await b.hgetall("h")

        asyncio.run(main())

    def test_kv_ops(self):
        async def main():
            b = _make_redis_backend()
            assert await b.get("k") is None
            await b.set("k", "v", ttl=10)
            assert await b.get("k") == "v"
            assert await b.exists("k")
            await b.delete("k")
            assert not await b.exists("k")

        asyncio.run(main())

    def test_list_ops(self):
        async def main():
            b = _make_redis_backend()
            await b.rpush("l", "a", "b", "c")
            assert await b.lrange("l", 0, -1) == ["a", "b", "c"]
            assert await b.llen("l") == 3
            await b.ltrim("l", 1, -1)
            assert await b.lrange("l", 0, -1) == ["b", "c"]

        asyncio.run(main())

    def test_ttl_expiry(self):
        async def main():
            b = _make_redis_backend()
            await b.set("k", "v", ttl=1)  # Redis TTL 秒级
            assert await b.get("k") == "v"
            # 过期验证由 MemoryBackend 测试覆盖；此处验证 Redis 语义下 set 正常

        asyncio.run(main())


class TestComponentOnRedisBackend:
    """关键组件在 Redis 语义下的行为（事件流 / 成本熔断）。"""

    def test_event_stream_on_redis(self):
        from app.agent.event_stream import EventStreamStore

        async def main():
            store = EventStreamStore()
            store._backend = _make_redis_backend()  # 注入 Redis 后端
            for i in range(5):
                await store.append("redis-sess", f"e{i}")
            events = await store.get_events("redis-sess", after_seq=2)
            assert [e["seq"] for e in events] == [3, 4]
            assert await store.next_seq("redis-sess") == 5
            await store.clear("redis-sess")
            assert await store.get_events("redis-sess") == []

        asyncio.run(main())

    def test_cost_guard_on_redis(self):
        from app.security.cost_guard import CostGuard
        from app.config.security_config import CostGuardConfig

        async def main():
            guard = CostGuard(CostGuardConfig(session_token_budget=100, action="refuse"))
            guard._backend = _make_redis_backend()
            await guard.record("redis-cost", 60, 60)
            check = await guard.check("redis-cost")
            assert not check.allowed
            assert check.action == "refuse"
            await guard.reset("redis-cost")
            assert (await guard.check("redis-cost")).allowed

        asyncio.run(main())
