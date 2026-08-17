# app/agent/event_stream.py
"""
会话事件流存储（Session Event Stream Store）

P3 断点恢复：把 Agent 执行的 SSE 事件按会话顺序缓存，
前端断线重连后通过 `after_seq` 拉取缺失事件回放。

- 统一存储后端（memory / redis），支撑多实例部署
- 事件带全局递增序号（seq），支持增量拉取
- 滑动窗口（每会话保留最近 N 条）+ TTL 自动过期
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class EventStreamStore:
    """会话事件流缓存（统一存储后端：memory / redis）。"""

    def __init__(
        self,
        max_events_per_session: int = 500,
        ttl_seconds: float = 3600.0,
    ):
        self.max_events_per_session = max_events_per_session
        self.ttl_seconds = ttl_seconds
        self._backend = None

    def _b(self):
        if self._backend is None:
            from app.storage.backend import get_storage_backend

            self._backend = get_storage_backend()
        return self._backend

    @staticmethod
    def _stream_key(session_id: str) -> str:
        return f"event:stream:{session_id}"

    @staticmethod
    def _meta_key(session_id: str) -> str:
        return f"event:meta:{session_id}"

    # ------------------------------------------------------------------
    async def append(self, session_id: str, event: str) -> int:
        """追加一条事件，返回其序号（list + seq 元数据）。"""
        import json as _json

        b = self._b()
        stream_key = self._stream_key(session_id)
        meta_key = self._meta_key(session_id)
        seq = await b.hincrby(meta_key, "seq", 1) - 1
        await b.rpush(stream_key, _json.dumps([seq, event], ensure_ascii=False))
        # 滑动窗口裁剪
        length = await b.llen(stream_key)
        if length > self.max_events_per_session:
            await b.ltrim(stream_key, length - self.max_events_per_session, -1)
        if self.ttl_seconds > 0:
            await b.expire(stream_key, self.ttl_seconds)
            await b.expire(meta_key, self.ttl_seconds)
        return seq

    async def get_events(
        self,
        session_id: str,
        after_seq: int = -1,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        获取增量事件。

        Args:
            session_id: 会话 ID
            after_seq: 只返回序号大于该值的事件（-1 返回全部）
            limit: 最大返回条数

        Returns:
            [{"seq": seq, "event": "..."}]
        """
        import json as _json

        try:
            raw = await self._b().lrange(self._stream_key(session_id), 0, -1)
        except Exception as e:
            logger.debug("Event stream read failed: %s", e)
            return []
        events = []
        for item in raw:
            try:
                seq, ev = _json.loads(item)
            except Exception:
                continue
            if seq > after_seq:
                events.append({"seq": seq, "event": ev})
        return events[-limit:]

    async def next_seq(self, session_id: str) -> int:
        """会话当前最大序号（用于断点标记）。"""
        try:
            raw = await self._b().hgetall(self._meta_key(session_id))
            return int(raw.get("seq", 0))
        except Exception:
            return 0

    async def has_stream(self, session_id: str) -> bool:
        try:
            return await self._b().exists(self._stream_key(session_id))
        except Exception:
            return False

    async def clear(self, session_id: str) -> None:
        """清理会话流。"""
        try:
            b = self._b()
            await b.delete(self._stream_key(session_id))
            await b.delete(self._meta_key(session_id))
        except Exception as e:
            logger.debug("Event stream clear failed: %s", e)


# 全局单例
event_stream_store = EventStreamStore()
