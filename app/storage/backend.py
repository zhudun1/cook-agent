# app/storage/backend.py
"""
统一存储后端抽象（Storage Backend）

生产级 Agent 的数据面抽象：把进程内组件（成本熔断计数 / 审批请求 /
事件流 / 工具 SLO）的存储统一到一个可插拔后端，支撑多实例部署。

实现:
- MemoryBackend: 进程内（单机 / 默认，测试友好）
- RedisBackend:  Redis（生产，多实例共享状态）

原语:
- hash:  hincrby / hgetall / hset / hdel   （计数器 / 字段聚合）
- kv:    get / set(ttl) / delete / exists   （对象存储）
- list:  rpush / lrange / llen / ltrim      （流 / 事件序列）
- ttl:   expire

设计原则:
- 值一律字符串（调用方负责 JSON 序列化），接口最小
- 组件通过 `get_storage_backend()` 获取单例，无感知切换
"""

from __future__ import annotations

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """存储后端抽象。"""

    name: str = "abstract"

    # ------------------------------------------------------------------
    # hash
    # ------------------------------------------------------------------
    @abstractmethod
    async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        """hash 字段自增，返回新值。"""

    @abstractmethod
    async def hgetall(self, key: str) -> Dict[str, str]:
        """读取 hash 全部字段。"""

    @abstractmethod
    async def hset(self, key: str, field: str, value: str) -> None:
        """写入 hash 字段。"""

    @abstractmethod
    async def hdel(self, key: str, *fields: str) -> int:
        """删除 hash 字段。"""

    # ------------------------------------------------------------------
    # kv
    # ------------------------------------------------------------------
    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """读取值。"""

    @abstractmethod
    async def set(self, key: str, value: str, ttl: Optional[float] = None) -> None:
        """写入值（可选 TTL 秒）。"""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除键。"""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """键是否存在。"""

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------
    @abstractmethod
    async def rpush(self, key: str, *values: str) -> int:
        """尾部追加，返回长度。"""

    @abstractmethod
    async def lrange(self, key: str, start: int, stop: int) -> List[str]:
        """区间读取（stop=-1 表示末尾）。"""

    @abstractmethod
    async def llen(self, key: str) -> int:
        """列表长度。"""

    @abstractmethod
    async def ltrim(self, key: str, start: int, stop: int) -> None:
        """裁剪列表。"""

    # ------------------------------------------------------------------
    # ttl / misc
    # ------------------------------------------------------------------
    @abstractmethod
    async def expire(self, key: str, ttl: float) -> None:
        """设置过期时间（秒）。"""

    async def close(self) -> None:
        """释放资源（应用退出时调用）。"""


class MemoryBackend(StorageBackend):
    """进程内实现（单机默认，测试友好）。"""

    name = "memory"

    def __init__(self):
        self._data: Dict[str, str] = {}
        self._hash: Dict[str, Dict[str, str]] = {}
        self._lists: Dict[str, List[str]] = {}
        self._expires: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _alive(self, key: str) -> bool:
        exp = self._expires.get(key)
        return exp is None or time.time() < exp

    def _touch(self, key: str) -> None:
        if not self._alive(key):
            self._data.pop(key, None)
            self._hash.pop(key, None)
            self._lists.pop(key, None)
            self._expires.pop(key, None)

    # hash
    async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        with self._lock:
            self._touch(key)
            h = self._hash.setdefault(key, {})
            new = int(h.get(field, 0)) + amount
            h[field] = str(new)
            return new

    async def hgetall(self, key: str) -> Dict[str, str]:
        with self._lock:
            self._touch(key)
            return dict(self._hash.get(key, {}))

    async def hset(self, key: str, field: str, value: str) -> None:
        with self._lock:
            self._touch(key)
            self._hash.setdefault(key, {})[field] = value

    async def hdel(self, key: str, *fields: str) -> int:
        with self._lock:
            self._touch(key)
            h = self._hash.get(key)
            if not h:
                return 0
            n = 0
            for f in fields:
                if h.pop(f, None) is not None:
                    n += 1
            return n

    # kv
    async def get(self, key: str) -> Optional[str]:
        with self._lock:
            self._touch(key)
            return self._data.get(key)

    async def set(self, key: str, value: str, ttl: Optional[float] = None) -> None:
        with self._lock:
            self._touch(key)
            self._data[key] = value
            if ttl is not None:
                self._expires[key] = time.time() + ttl
            else:
                self._expires.pop(key, None)

    async def delete(self, key: str) -> bool:
        with self._lock:
            existed = key in self._data or key in self._hash or key in self._lists
            self._data.pop(key, None)
            self._hash.pop(key, None)
            self._lists.pop(key, None)
            self._expires.pop(key, None)
            return existed

    async def exists(self, key: str) -> bool:
        with self._lock:
            self._touch(key)
            return key in self._data or key in self._hash or key in self._lists

    # list
    async def rpush(self, key: str, *values: str) -> int:
        with self._lock:
            self._touch(key)
            lst = self._lists.setdefault(key, [])
            lst.extend(values)
            return len(lst)

    async def lrange(self, key: str, start: int, stop: int) -> List[str]:
        with self._lock:
            self._touch(key)
            lst = self._lists.get(key, [])
            if stop < 0:
                stop = len(lst) - 1
            if start < 0:
                start = max(0, len(lst) + start)
            if start > stop or start >= len(lst):
                return []
            return lst[start : stop + 1]

    async def llen(self, key: str) -> int:
        with self._lock:
            self._touch(key)
            return len(self._lists.get(key, []))

    async def ltrim(self, key: str, start: int, stop: int) -> None:
        with self._lock:
            self._touch(key)
            lst = self._lists.get(key)
            if lst is None:
                return
            if stop < 0:
                stop = len(lst) - 1
            self._lists[key] = lst[start : stop + 1]

    # ttl
    async def expire(self, key: str, ttl: float) -> None:
        with self._lock:
            self._expires[key] = time.time() + ttl


class RedisBackend(StorageBackend):
    """Redis 实现（生产，多实例共享状态）。"""

    name = "redis"

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0,
                 password: Optional[str] = None, **kwargs):
        from redis.asyncio import Redis

        self._redis = Redis(
            host=host, port=port, db=db, password=password,
            decode_responses=True, **kwargs,
        )

    # hash
    async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        return await self._redis.hincrby(key, field, amount)

    async def hgetall(self, key: str) -> Dict[str, str]:
        return await self._redis.hgetall(key)

    async def hset(self, key: str, field: str, value: str) -> None:
        await self._redis.hset(key, field, value)

    async def hdel(self, key: str, *fields: str) -> int:
        return await self._redis.hdel(key, *fields)

    # kv
    async def get(self, key: str) -> Optional[str]:
        return await self._redis.get(key)

    async def set(self, key: str, value: str, ttl: Optional[float] = None) -> None:
        if ttl is not None:
            # Redis TTL 秒级：向上取整（最小 1s）
            await self._redis.set(key, value, ex=max(1, math.ceil(ttl)))
        else:
            await self._redis.set(key, value)

    async def delete(self, key: str) -> bool:
        return bool(await self._redis.delete(key))

    async def exists(self, key: str) -> bool:
        return bool(await self._redis.exists(key))

    # list
    async def rpush(self, key: str, *values: str) -> int:
        return await self._redis.rpush(key, *values)

    async def lrange(self, key: str, start: int, stop: int) -> List[str]:
        return await self._redis.lrange(key, start, stop)

    async def llen(self, key: str) -> int:
        return await self._redis.llen(key)

    async def ltrim(self, key: str, start: int, stop: int) -> None:
        await self._redis.ltrim(key, start, stop)

    # ttl
    async def expire(self, key: str, ttl: float) -> None:
        await self._redis.expire(key, int(ttl))

    async def close(self) -> None:
        await self._redis.aclose()


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

_backend: Optional[StorageBackend] = None
_backend_lock = threading.Lock()


def get_storage_backend() -> StorageBackend:
    """获取全局存储后端单例（按 settings.storage.backend 选择）。"""
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                _backend = _create_backend()
                logger.info("Storage backend: %s", _backend.name)
    return _backend


def _create_backend() -> StorageBackend:
    try:
        from app.config import settings

        cfg = settings.storage
        if cfg.backend == "redis":
            db_cfg = settings.database.redis
            return RedisBackend(
                host=db_cfg.host,
                port=db_cfg.port,
                db=db_cfg.db,
                password=db_cfg.password,
            )
    except Exception as e:
        logger.warning("Failed to init configured backend, fallback to memory: %s", e)
    return MemoryBackend()


def set_storage_backend(backend: StorageBackend) -> StorageBackend:
    """测试/注入用：覆盖全局后端。"""
    global _backend
    _backend = backend
    return backend


def reset_storage_backend() -> None:
    """测试用：重置单例。"""
    global _backend
    _backend = None
