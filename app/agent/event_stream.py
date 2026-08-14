# app/agent/event_stream.py
"""
会话事件流存储（Session Event Stream Store）

P3 断点恢复：把 Agent 执行的 SSE 事件按会话顺序缓存，
前端断线重连后通过 `after_seq` 拉取缺失事件回放。

- 进程内滑动窗口（每会话保留最近 N 条，防止内存膨胀）
- 事件带全局递增序号（seq），支持增量拉取
- 会话结束/超时清理

生产多实例部署时应替换为 Redis 流（接口已隔离）。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EventStreamStore:
    """会话事件流缓存。"""

    def __init__(self, max_events_per_session: int = 500):
        self.max_events_per_session = max_events_per_session
        self._streams: Dict[str, Deque[Tuple[int, str]]] = {}
        self._next_seq: Dict[str, int] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def append(self, session_id: str, event: str) -> int:
        """追加一条事件，返回其序号。"""
        with self._lock:
            seq = self._next_seq.get(session_id, 0)
            stream = self._streams.setdefault(session_id, deque(maxlen=self.max_events_per_session))
            stream.append((seq, event))
            self._next_seq[session_id] = seq + 1
            return seq

    def get_events(
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
            [{"seq": n, "event": "..."}]
        """
        with self._lock:
            stream = self._streams.get(session_id)
            if not stream:
                return []
            events = [
                {"seq": seq, "event": ev}
                for seq, ev in stream
                if seq > after_seq
            ]
        return events[-limit:]

    def next_seq(self, session_id: str) -> int:
        """会话当前最大序号（用于断点标记）。"""
        with self._lock:
            return self._next_seq.get(session_id, 0)

    def has_stream(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._streams

    def clear(self, session_id: str) -> None:
        """清理会话流。"""
        with self._lock:
            self._streams.pop(session_id, None)
            self._next_seq.pop(session_id, None)

    def clear_expired(self, ttl_seconds: float = 3600) -> int:
        """清理超过 TTL 的会话流（惰性治理）。"""
        now = time.time()
        # 简化：依赖 deque 长度上限；TTL 治理留给分布式实现
        return 0


# 全局单例
event_stream_store = EventStreamStore()
