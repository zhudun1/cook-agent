# app/evaluation/tool_metrics.py
"""
工具级 SLO 采集（Tool Metrics）

生产级 Agent 的可观测性第一层：每个工具调用的
- 成功率（success rate）
- 延迟（平均 / P50 / P95）
- 错误码分布（error_code）

接入点：ToolExecutor.execute（每次调用前后计时与记录）。
数据面走统一存储后端（memory / redis），支持多实例聚合。
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_RECORDS_KEY = "toolmetrics:records"


class ToolMetricsCollector:
    """
    工具调用指标采集器（滑动窗口 + 统一存储后端）。

    用法::

        collector = ToolMetricsCollector()
        await collector.record("web_search", success=True, duration_ms=120)
        await collector.record("web_search", success=False, error_code="TIMEOUT", duration_ms=30000)
        stats = await collector.get_stats()  # 按工具聚合
    """

    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self._backend = None

    def _b(self):
        if self._backend is None:
            from app.storage.backend import get_storage_backend

            self._backend = get_storage_backend()
        return self._backend

    # ------------------------------------------------------------------
    async def record(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        error_code: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """记录一次工具调用（追加到滑动窗口）。"""
        item = json.dumps(
            {
                "tool": tool_name,
                "success": bool(success),
                "duration_ms": round(float(duration_ms), 2),
                "error_code": error_code,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": time.time(),
            },
            ensure_ascii=False,
            default=str,
        )
        try:
            b = self._b()
            await b.rpush(_RECORDS_KEY, item)
            length = await b.llen(_RECORDS_KEY)
            if length > self.max_records:
                await b.ltrim(_RECORDS_KEY, length - self.max_records, -1)
        except Exception as e:
            logger.debug("Tool metrics record failed: %s", e)

    # ------------------------------------------------------------------
    async def get_stats(
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
        try:
            raw = await self._b().lrange(_RECORDS_KEY, 0, -1)
        except Exception as e:
            logger.debug("Tool metrics read failed: %s", e)
            raw = []

        records: List[Dict[str, Any]] = []
        for item in raw:
            try:
                r = json.loads(item)
            except Exception:
                continue
            if since is not None and r.get("timestamp", 0) < since:
                continue
            if tool_name and r.get("tool") != tool_name:
                continue
            records.append(r)

        by_tool: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            by_tool.setdefault(r["tool"], []).append(r)

        stats: Dict[str, Any] = {"tools": {}, "totals": {}}
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
        error_codes: Dict[str, int] = {}
        for r in records:
            if not r["success"]:
                code = r.get("error_code") or "UNKNOWN"
                error_codes[code] = error_codes.get(code, 0) + 1

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
            "error_codes": error_codes,
        }

    async def reset(self) -> None:
        try:
            await self._b().delete(_RECORDS_KEY)
        except Exception as e:
            logger.debug("Tool metrics reset failed: %s", e)


# 全局单例
tool_metrics = ToolMetricsCollector()
