from __future__ import annotations


# app/telemetry/trajectory.py
"""
Agent turn 轨迹 JSON 持久化与回放调试

将一次 Agent 执行的完整调用链（推理 -> 工具调用 -> 工具结果 -> 最终回答）
记录为结构化 JSON 轨迹文件，支持：
- 回放调试：按 trace_id 加载完整轨迹，还原每一步决策
- 排查记忆退化：结合 window_stats 查看滑动窗口截断了哪些消息
- 审计：记录模型、token、耗时、重试/降级事件

轨迹文件格式（data/trajectories/<trace_id>.json）::

    {
      "trace_id": "...",
      "session_id": "...",
      "user_id": "...",
      "agent": "default",
      "model": "...",
      "started_at": "...",
      "finished_at": "...",
      "turns": [
        {"turn": 0, "phase": "reasoning", "data": {...}, "timestamp": "..."},
        {"turn": 1, "phase": "tool_call", "data": {...}, "timestamp": "..."},
        {"turn": 1, "phase": "tool_result", "data": {...}, "timestamp": "..."},
        {"turn": 2, "phase": "answer", "data": {...}, "timestamp": "..."}
      ],
      "final_answer": "...",
      "window_stats": {...},
      "total_duration_ms": 1234
    }
"""


import glob
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _default_storage_dir() -> str:
    try:
        from app.config import settings

        return settings.telemetry.trajectory.storage_dir
    except Exception:
        return "data/trajectories"


class TrajectoryRecorder:
    """
    轨迹记录器：收集一次 Agent turn 的调用链，落盘为 JSON。

    用法::

        recorder = TrajectoryRecorder(
            trace_id="...", session_id="...", user_id="...", agent="default"
        )
        recorder.record("reasoning", {...})
        recorder.record("tool_call", {...})
        recorder.save()
    """

    def __init__(
        self,
        trace_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent: str = "default",
        storage_dir: Optional[str] = None,
        enabled: Optional[bool] = None,
        record_full_payload: Optional[bool] = None,
        max_payload_chars: Optional[int] = None,
    ):
        try:
            from app.config import settings

            tcfg = settings.telemetry.trajectory
            self.enabled = tcfg.enabled if enabled is None else enabled
            self.storage_dir = tcfg.storage_dir if storage_dir is None else storage_dir
            self.record_full_payload = (
                tcfg.record_full_payload
                if record_full_payload is None
                else record_full_payload
            )
            self.max_payload_chars = (
                tcfg.max_payload_chars if max_payload_chars is None else max_payload_chars
            )
        except Exception:
            self.enabled = True if enabled is None else enabled
            self.storage_dir = storage_dir or _default_storage_dir()
            self.record_full_payload = True if record_full_payload is None else record_full_payload
            self.max_payload_chars = 20000 if max_payload_chars is None else max_payload_chars

        self.trace_id = trace_id
        self.session_id = session_id
        self.user_id = user_id
        self.agent = agent
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.finished_at: Optional[str] = None
        self.turns: List[Dict[str, Any]] = []
        self.final_answer: Optional[str] = None
        self.window_stats: Optional[dict] = None
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def record(
        self,
        phase: str,
        data: Dict[str, Any],
        turn: Optional[int] = None,
    ) -> None:
        """记录一个轨迹步骤。

        Args:
            phase: reasoning | tool_call | tool_result | answer | error | llm_call
            data: 步骤数据
            turn: ReAct 迭代序号（自动推断）
        """
        if not self.enabled:
            return
        if turn is None:
            turn = len([t for t in self.turns if t["phase"] in ("tool_call", "tool_result", "reasoning")])
        self.turns.append(
            {
                "turn": turn,
                "phase": phase,
                "data": self._sanitize(data),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """按配置截断/脱敏 payload。"""
        if self.record_full_payload:
            return data
        result = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > self.max_payload_chars:
                result[k] = v[: self.max_payload_chars] + "...[truncated]"
            elif isinstance(v, dict):
                result[k] = self._sanitize(v)
            elif isinstance(v, list):
                result[k] = [
                    self._sanitize(i) if isinstance(i, dict) else i for i in v
                ]
            else:
                result[k] = v
        return result

    def set_final_answer(self, answer: str) -> None:
        self.final_answer = answer

    def set_window_stats(self, stats: Optional[dict]) -> None:
        self.window_stats = stats

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent": self.agent,
            "started_at": self.started_at,
            "finished_at": self.finished_at or datetime.now(timezone.utc).isoformat(),
            "turns": self.turns,
            "final_answer": self.final_answer,
            "window_stats": self.window_stats,
            "metadata": self.metadata,
            "total_turns": len(self.turns),
        }

    def save(self) -> Optional[str]:
        """将轨迹写入 JSON 文件。

        Returns:
            文件路径（禁用或失败时返回 None）
        """
        if not self.enabled:
            return None
        try:
            storage = Path(self.storage_dir)
            storage.mkdir(parents=True, exist_ok=True)
            path = storage / f"{self.trace_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)
            logger.info("Trajectory saved: %s (%d turns)", path, len(self.turns))
            return str(path)
        except Exception as e:
            logger.error("Failed to save trajectory %s: %s", self.trace_id, e)
            return None


# ---------------------------------------------------------------------------
# 回放调试
# ---------------------------------------------------------------------------

def load_trajectory(
    trace_id: str, storage_dir: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """按 trace_id 加载轨迹（回放调试入口）。"""
    storage = Path(storage_dir or _default_storage_dir())
    path = storage / f"{trace_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load trajectory %s: %s", trace_id, e)
        return None


def list_trajectories(
    storage_dir: Optional[str] = None, limit: int = 50
) -> List[Dict[str, Any]]:
    """列出最近的轨迹（按修改时间倒序）。"""
    storage = Path(storage_dir or _default_storage_dir())
    if not storage.exists():
        return []
    files = sorted(
        glob.glob(str(storage / "*.json")),
        key=os.path.getmtime,
        reverse=True,
    )[:limit]
    result = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                traj = json.load(f)
            result.append(
                {
                    "trace_id": traj.get("trace_id"),
                    "session_id": traj.get("session_id"),
                    "user_id": traj.get("user_id"),
                    "agent": traj.get("agent"),
                    "started_at": traj.get("started_at"),
                    "total_turns": traj.get("total_turns"),
                    "final_answer_preview": (traj.get("final_answer") or "")[:120],
                }
            )
        except Exception:
            continue
    return result


def render_replay(trajectory: Dict[str, Any]) -> str:
    """将轨迹渲染为人类可读的回放文本（调试用）。"""
    lines = [
        f"# Trajectory {trajectory.get('trace_id')}",
        f"- session: {trajectory.get('session_id')}",
        f"- agent: {trajectory.get('agent')}",
        f"- started: {trajectory.get('started_at')}",
        f"- total turns: {trajectory.get('total_turns')}",
        "",
    ]
    ws = trajectory.get("window_stats")
    if ws:
        lines.append(
            "## 滑动窗口统计\n"
            f"- budget: {ws.get('total_budget')} tokens, used: {ws.get('total_tokens')}, "
            f"dropped messages: {ws.get('dropped_messages')}\n"
        )
    for t in trajectory.get("turns", []):
        phase = t.get("phase")
        data = t.get("data", {})
        if phase == "reasoning":
            lines.append(f"[{t.get('turn')}] 🧠 reasoning: {str(data)[:200]}")
        elif phase == "tool_call":
            lines.append(
                f"[{t.get('turn')}] 🔧 tool_call: {data.get('name')}({data.get('arguments')})"
            )
        elif phase == "tool_result":
            ok = data.get("success")
            mark = "✅" if ok else "❌"
            lines.append(
                f"[{t.get('turn')}] {mark} tool_result: "
                f"{str(data.get('result'))[:200] or data.get('error')}"
            )
        elif phase == "answer":
            lines.append(f"[{t.get('turn')}] 💬 answer: {str(data)[:200]}")
        elif phase == "error":
            lines.append(f"[{t.get('turn')}] ⛔ error: {data}")
        else:
            lines.append(f"[{t.get('turn')}] {phase}: {str(data)[:200]}")
    fa = trajectory.get("final_answer")
    if fa:
        lines.append(f"\n## Final Answer\n{fa[:500]}")
    return "\n".join(lines)
