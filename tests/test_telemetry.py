# tests/test_telemetry.py
"""
telemetry 单元测试：traceId 上下文、结构化日志脱敏、轨迹持久化与回放。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))



from app.telemetry.trace import (
    clear_trace_id,
    get_span_id,
    get_trace_id,
    set_attribute,
    span,
    trace,
)
from app.telemetry.logger import redact
from app.telemetry.trajectory import (
    TrajectoryRecorder,
    list_trajectories,
    load_trajectory,
    render_replay,
)


class TestTrace:
    def test_trace_id_propagation(self):
        assert get_trace_id() is None
        with trace(trace_id="req-1", user_id="u1") as tid:
            assert tid == "req-1"
            assert get_trace_id() == "req-1"
        assert get_trace_id() is None

    def test_span_stack(self):
        with trace(trace_id="t1"):
            with span("outer") as outer:
                assert get_span_id() == outer.span_id
                with span("inner"):
                    assert get_current_span_name() == "inner"
                    assert get_current_span_parent() == outer.span_id

    def test_attributes(self):
        with trace(trace_id="t1"):
            set_attribute("user_id", "u1")
            from app.telemetry.trace import get_attributes

            assert get_attributes()["user_id"] == "u1"

    def test_clear(self):
        set_trace_id_via_context()
        clear_trace_id()
        assert get_trace_id() is None


def get_current_span_name():
    from app.telemetry.trace import get_current_span

    return get_current_span().name


def get_current_span_parent():
    from app.telemetry.trace import get_current_span

    return get_current_span().parent_id


def set_trace_id_via_context():
    from app.telemetry.trace import set_trace_id

    set_trace_id("x")


class TestRedact:
    def test_sensitive_fields(self):
        out = redact(
            {"api_key": "sk-x", "nested": {"password": "p", "ok": 1}},
            ["api_key", "password"],
        )
        assert out["api_key"] == "***"
        assert out["nested"]["password"] == "***"
        assert out["nested"]["ok"] == 1

    def test_long_string_truncated(self):
        out = redact({"content": "x" * 200000}, [])
        assert out["content"].endswith("...[truncated]")


class TestTrajectory:
    def test_save_load_replay(self, tmp_path):
        rec = TrajectoryRecorder(
            "traj-1",
            session_id="s1",
            user_id="u1",
            agent="default",
            storage_dir=str(tmp_path),
        )
        rec.record("tool_call", {"name": "web_search", "arguments": {"q": "红烧肉"}})
        rec.record("tool_result", {"name": "web_search", "success": True, "result": "..."})
        rec.set_final_answer("红烧肉的做法是...")
        rec.set_window_stats({"total_budget": 16000, "dropped_messages": 2})
        path = rec.save()
        assert path and os.path.basename(path) == "traj-1.json"

        loaded = load_trajectory("traj-1", storage_dir=str(tmp_path))
        assert loaded is not None
        assert loaded["final_answer"].startswith("红烧肉")
        assert loaded["total_turns"] == 2
        assert loaded["window_stats"]["dropped_messages"] == 2

        replay = render_replay(loaded)
        assert "tool_call" in replay and "红烧肉" in replay

    def test_list_trajectories(self, tmp_path):
        for tid in ("t1", "t2"):
            rec = TrajectoryRecorder(tid, storage_dir=str(tmp_path))
            rec.save()
        lst = list_trajectories(storage_dir=str(tmp_path))
        assert {x["trace_id"] for x in lst} == {"t1", "t2"}

    def test_missing_returns_none(self, tmp_path):
        assert load_trajectory("nope", storage_dir=str(tmp_path)) is None
