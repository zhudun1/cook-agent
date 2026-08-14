# app/harness/goals.py
"""
持久化目标（Goal）注册表

借鉴 DeepSeek Harness 的 goal 工具：
- create_goal / get_goal / update_goal，带 **版本号（revision）乐观并发控制**
- 状态机：active -> paused / completed / blocked；paused -> active
- 自动续跑计数（rounds_started / max_rounds）
- 武装语义（armed / disarmed）：会话恢复后目标自动解除武装，
  需显式 resume 重新武装（防止误续跑）

存储：JSON 文件（data/harness/goals.json），进程级锁保护，幂等。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class GoalConflictError(RuntimeError):
    """版本号冲突：目标已被其他调用方更新。"""


class GoalStateError(RuntimeError):
    """状态机非法转换。"""


def _default_store_path() -> str:
    try:
        from app.config import settings

        return os.path.join(settings.telemetry.trajectory.storage_dir, "..", "harness", "goals.json")
    except Exception:
        return "data/harness/goals.json"


@dataclass
class Goal:
    """持久化目标。"""

    goal_id: str
    objective: str
    phase: str = "active"  # active | paused | completed | blocked
    revision: int = 1
    max_goal_rounds: Optional[int] = None
    rounds_started: int = 0
    blocked_reason: Optional[str] = None
    activation: str = "armed"  # armed | disarmed
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


class GoalStore:
    """
    文件持久化的 Goal 注册表。

    用法::

        store = GoalStore("data/harness/goals.json")
        goal = store.create_goal("build-rag", "完成 RAG 评测系统")
        goal = store.get_goal(goal.goal_id)
        goal = store.update_goal(goal.goal_id, goal.revision, "edit",
                                 objective="新目标", max_goal_rounds=5)
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or _default_store_path()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _load(self) -> Dict[str, dict]:
        try:
            p = Path(self.path)
            if not p.exists():
                return {}
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("Failed to load goals: %s", e)
            return {}

    def _save(self, data: Dict[str, dict]) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _touch(self, goal: Goal) -> Goal:
        goal.updated_at = datetime.now(timezone.utc).isoformat()
        return goal

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def create_goal(
        self,
        objective: str,
        max_goal_rounds: Optional[int] = None,
        goal_id: Optional[str] = None,
    ) -> Goal:
        """
        创建目标。

        Args:
            objective: 完成目标描述
            max_goal_rounds: 自动续跑轮次上限（None 不限）
            goal_id: 显式 ID（默认自动生成）

        Returns:
            新建的 Goal
        """
        with self._lock:
            data = self._load()
            gid = goal_id or uuid.uuid4().hex
            goal = Goal(
                goal_id=gid,
                objective=objective,
                max_goal_rounds=max_goal_rounds,
            )
            data[gid] = goal.to_dict()
            self._save(data)
            logger.info("Goal created: %s (%s)", gid, objective[:60])
            return goal

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """读取目标（含精确 revision，供 update 乐观并发控制）。"""
        with self._lock:
            data = self._load()
            raw = data.get(goal_id)
            if not raw:
                return None
            return Goal(**raw)

    def list_goals(self, phase: Optional[str] = None) -> List[Goal]:
        """列出目标（可按状态过滤）。"""
        with self._lock:
            data = self._load()
            goals = [Goal(**raw) for raw in data.values()]
        if phase:
            goals = [g for g in goals if g.phase == phase]
        return sorted(goals, key=lambda g: g.created_at, reverse=True)

    def update_goal(
        self,
        goal_id: str,
        revision: int,
        action: str,
        *,
        objective: Optional[str] = None,
        max_goal_rounds: Optional[int] = None,
        blocked_reason: Optional[str] = None,
    ) -> Goal:
        """
        更新目标（乐观并发控制：revision 必须匹配当前版本）。

        Args:
            goal_id: 目标 ID
            revision: 期望的当前版本号（必须与 get_goal 返回一致）
            action: edit | pause | resume | complete | blocked
            objective: 新目标描述（仅 edit）
            max_goal_rounds: 新轮次上限（仅 edit）
            blocked_reason: 阻塞原因（仅 blocked）

        Returns:
            更新后的 Goal

        Raises:
            GoalConflictError: revision 不匹配（目标已被其他调用方更新）
            GoalStateError: 状态机非法转换
        """
        with self._lock:
            data = self._load()
            raw = data.get(goal_id)
            if not raw:
                raise KeyError(f"Goal not found: {goal_id}")
            goal = Goal(**raw)

            # 乐观并发控制
            if goal.revision != revision:
                raise GoalConflictError(
                    f"Goal {goal_id} revision mismatch: expected {revision}, "
                    f"actual {goal.revision} (concurrent update detected)"
                )

            # 状态机转换
            if action == "edit":
                if objective is not None:
                    goal.objective = objective
                if max_goal_rounds is not None:
                    goal.max_goal_rounds = max_goal_rounds
            elif action == "pause":
                if goal.phase != "active":
                    raise GoalStateError(f"Cannot pause goal in phase '{goal.phase}'")
                goal.phase = "paused"
                goal.activation = "disarmed"
            elif action == "resume":
                if goal.phase not in ("active", "paused"):
                    raise GoalStateError(f"Cannot resume goal in phase '{goal.phase}'")
                goal.phase = "active"
                goal.activation = "armed"
            elif action == "complete":
                goal.phase = "completed"
                goal.activation = "disarmed"
            elif action == "blocked":
                if goal.phase != "active":
                    raise GoalStateError(f"Cannot block goal in phase '{goal.phase}'")
                goal.phase = "blocked"
                goal.blocked_reason = blocked_reason
                goal.activation = "disarmed"
            else:
                raise GoalStateError(f"Unknown action: {action}")

            goal.revision += 1
            goal = self._touch(goal)
            data[goal_id] = goal.to_dict()
            self._save(data)
            logger.info(
                "Goal %s updated: action=%s revision=%d phase=%s",
                goal_id, action, goal.revision, goal.phase,
            )
            return goal

    def record_round(self, goal_id: str) -> Optional[Goal]:
        """记录一次自动续跑轮次（自增 rounds_started）。"""
        with self._lock:
            data = self._load()
            raw = data.get(goal_id)
            if not raw:
                return None
            goal = Goal(**raw)
            goal.rounds_started += 1
            if (
                goal.max_goal_rounds is not None
                and goal.rounds_started >= goal.max_goal_rounds
            ):
                goal.phase = "completed"
                goal.activation = "disarmed"
            goal.revision += 1
            goal = self._touch(goal)
            data[goal_id] = goal.to_dict()
            self._save(data)
            return goal


# 全局单例
goal_store = GoalStore()
