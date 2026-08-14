# app/harness/todos.py
"""
结构化任务清单（TodoWrite）

借鉴 DeepSeek Harness 的 todo_write 工具：
- **整体替换语义**：每次写入提交完整清单（无部分更新，无逐项编辑）
- 状态机：pending -> in_progress -> completed
- 允许 in_progress 多项（并行任务）；工作未完成时至少一项 in_progress
- 按 trace_id / goal_id 作用域持久化，支持回放查看

存储：JSON 文件（data/harness/todos/<scope>.json）
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

VALID_STATUSES = {"pending", "in_progress", "completed"}


def _default_dir() -> str:
    try:
        from app.config import settings

        return os.path.join(settings.telemetry.trajectory.storage_dir, "..", "harness", "todos")
    except Exception:
        return "data/harness/todos"


@dataclass
class TodoItem:
    """清单条目。"""

    content: str
    status: str = "pending"


class TodoStore:
    """
    结构化任务清单存储（按 scope 隔离）。

    用法::

        store = TodoStore()
        store.replace("trace-123", [
            {"content": "检索", "status": "in_progress"},
            {"content": "生成", "status": "pending"},
        ])
        store.mark("trace-123", "检索", "completed")
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or _default_dir()
        self._lock = threading.Lock()

    def _path_for(self, scope: str) -> Path:
        return Path(self.base_dir) / f"{scope}.json"

    def _load(self, scope: str) -> List[dict]:
        p = self._path_for(scope)
        try:
            if not p.exists():
                return []
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("Failed to load todos for %s: %s", scope, e)
            return []

    def _save(self, scope: str, items: List[dict]) -> None:
        p = self._path_for(scope)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    def replace(self, scope: str, items: List[dict]) -> List[dict]:
        """
        整体替换清单（write 语义）。

        Args:
            scope: 作用域（trace_id / goal_id）
            items: [{"content": ..., "status": ...}, ...]

        Returns:
            规范化后的清单
        """
        normalized = []
        for it in items:
            content = it.get("content", "") if isinstance(it, dict) else str(it)
            status = it.get("status", "pending") if isinstance(it, dict) else "pending"
            if status not in VALID_STATUSES:
                status = "pending"
            normalized.append({"content": content, "status": status})

        with self._lock:
            self._save(scope, normalized)
        return normalized

    def mark(self, scope: str, content: str, status: str) -> bool:
        """将指定条目标记为新状态（找不到则返回 False）。"""
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        with self._lock:
            items = self._load(scope)
            for it in items:
                if it.get("content") == content:
                    it["status"] = status
                    self._save(scope, items)
                    return True
            return False

    def get(self, scope: str) -> List[dict]:
        """读取当前清单。"""
        return self._load(scope)

    def summary(self, scope: str) -> dict:
        """返回清单统计。"""
        items = self._load(scope)
        return {
            "total": len(items),
            "pending": sum(1 for i in items if i["status"] == "pending"),
            "in_progress": sum(1 for i in items if i["status"] == "in_progress"),
            "completed": sum(1 for i in items if i["status"] == "completed"),
        }


# 全局单例
todo_store = TodoStore()
