# app/evaluation/tool_metrics.py
"""
工具级 SLO 采集（Tool Metrics）

生产级 Agent 的可观测性第一层：每个工具调用的
- 成功率（success rate）
- 延迟（平均 / P50 / P95）
- 错误码分布（error_code）

接入点：ToolExecutor.execute（每次调用前后计时与记录）。
聚合为进程内内存窗口（默认保留最近 1000 条），提供查询接口。
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolMetricsCollector:
    """
    工具调用指标采集器（滑动窗口）。

    用法::

        collector = ToolMetricsCollector()
        collector.record("web_search", success=True, duration_ms=120)
        collector.record("web_search", success=False, error_code="TIMEOUT", duration_ms=30000)
        stats = collector.get_stats()  # 按工具聚合
    """

    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self._records: Deque[Dict[str, Any]] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def record(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        error_code: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """记录一次工具调用。"""
        with self._lock:
            self._records.append(
                {
                    "tool": tool_name,
                    "success": bool(success),
                    "duration_ms": round(float(duration_ms), 2),
                    "error_code": error_code,
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": time.time(),
                }
            )

    # ------------------------------------------------------------------
    def get_stats(
        self,
        tool_name: Optional[str] = None,
        since: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        聚合统计（按工具）。

        Args:
            tool_name: 只统计指定工具
            since: 只统计该时间戳之后的记录

        Returns:
            {tools: {tool_name: {...}}, totals: {...}}
        """
        with self._lock:
            records = list(self._records)
        if since is not None:
            records = [r for r in records if r["timestamp"] >= since]
        if tool_name:
            records = [r for r in records if r["tool"] == tool_name]

        by_tool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in records:
            by_tool[r["tool"]].append(r)

        stats = {"tools": {}, "totals": {}}
        for tool, items in by_tool.items():
            stats["tools"][tool] = self._aggregate(items)
        stats["totals"] = self._aggregate(records)
        return stats

    def _aggregate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            return {
                "calls": 0,
                "success_rate": None,
                "avg_duration_ms": None,
                "p50_ms": None,
                "p95_ms": None,
                "error_codes": {},
            }
        durations = [r["duration_ms"] for r in records]
        durations_sorted = sorted(durations)
        success = sum(1 for r in records if r["success"])
        error_codes: Dict[str, int] = defaultdict(int)
        for r in records:
            if not r["success"]:
                error_codes[r["error_code"] or "UNKNOWN"] += 1

        def percentile(p: float) -> Optional[float]:
            if not durations_sorted:
                return None
            idx = min(len(durations_sorted) - 1, int(len(durations_sorted) * p))
            return round(durations_sorted[idx], 2)

        return {
            "calls": len(records),
            "success_rate": round(success / len(records), 4),
            "avg_duration_ms": round(statistics.mean(durations), 2),
            "p50_ms": percentile(0.5),
            "p95_ms": percentile(0.95),
            "error_codes": dict(error_codes),
        }

    def reset(self) -> None:
        with self._lock:
            self._records.clear()


# 全局单例
tool_metrics = ToolMetricsCollector()
