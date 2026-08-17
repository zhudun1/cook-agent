# tests/test_storage_backend.py
"""
统一存储后端测试：MemoryBackend 原语 + 组件数据面一致性。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio

import pytest

from app.storage.backend import MemoryBackend, RedisBackend, set_storage_backend, get_storage_backend, reset_storage_backend


class TestMemoryBackend:
    @pytest.fixture(autouse=True)
    def _clean(self):
        yield

    def test_hash_ops(self):
        async def main():
            b = MemoryBackend()
            assert await b.hincrby("h", "f1", 1) == 1
            assert await b.hincrby("h", "f1", 5) == 6
            assert await b.hincrby("h", "f2", 2) == 2
            all_fields = await b.hgetall("h")
            assert all_fields == {"f1": "6", "f2": "2"}
            await b.hset("h", "f3", "x")
            assert (await b.hgetall("h"))["f3"] == "x"
            assert await b.hdel("h", "f1") == 1
            assert "f1" not in await b.hgetall("h")

        asyncio.run(main())

    def test_kv_ops(self):
        async def main():
            b = MemoryBackend()
            assert await b.get("k") is None
            await b.set("k", "v")
            assert await b.get("k") == "v"
            assert await b.exists("k")
            await b.delete("k")
            assert not await b.exists("k")

        asyncio.run(main())

    def test_ttl_expiry(self):
        async def main():
            b = MemoryBackend()
            await b.set("k", "v", ttl=0.05)
            assert await b.get("k") == "v"
            await asyncio.sleep(0.08)
            assert await b.get("k") is None

        asyncio.run(main())

    def test_list_ops(self):
        async def main():
            b = MemoryBackend()
            assert await b.rpush("l", "a", "b", "c") == 3
            assert await b.lrange("l", 0, -1) == ["a", "b", "c"]
            assert await b.llen("l") == 3
            await b.ltrim("l", 1, -1)
            assert await b.lrange("l", 0, -1) == ["b", "c"]

        asyncio.run(main())

    def test_list_range_bounds(self):
        async def main():
            b = MemoryBackend()
            await b.rpush("l", "a", "b", "c", "d")
            assert await b.lrange("l", 0, 1) == ["a", "b"]
            assert await b.lrange("l", 2, -1) == ["c", "d"]
            assert await b.lrange("l", 10, 20) == []

        asyncio.run(main())


class TestRedisBackendInterface:
    """RedisBackend 接口可用性（fakeredis 未装时跳过具体行为）。"""

    def test_importable(self):
        # RedisBackend 类应可导入（连接惰性）

        assert RedisBackend.name == "redis"


class TestBackendFactory:
    def test_default_memory(self):
        reset_storage_backend()
        assert get_storage_backend().name == "memory"

    def test_injection(self):
        custom = MemoryBackend()
        set_storage_backend(custom)
        assert get_storage_backend() is custom
        reset_storage_backend()
