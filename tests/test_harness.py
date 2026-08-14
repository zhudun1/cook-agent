# tests/test_harness.py
"""
Harness 模式组件单元测试：goals / todos / schema / jobs / workflow。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio

import pytest

from app.harness.goals import GoalConflictError, GoalStateError, GoalStore
from app.harness.todos import TodoStore
from app.harness.schema import validate_result
from app.harness.jobs import JobManager
from app.harness.workflow import Phase, parallel, pipeline, run_phases


class TestGoals:
    def test_crud_and_revision(self, tmp_path):
        store = GoalStore(str(tmp_path / "goals.json"))
        g = store.create_goal("完成目标", max_goal_rounds=3)
        assert g.phase == "active" and g.revision == 1

        # 乐观并发：错误 revision 冲突
        with pytest.raises(GoalConflictError):
            store.update_goal(g.goal_id, 99, "pause")

        g = store.get_goal(g.goal_id)
        g = store.update_goal(g.goal_id, g.revision, "pause")
        assert g.phase == "paused" and g.activation == "disarmed"
        g = store.update_goal(g.goal_id, g.revision, "resume")
        assert g.phase == "active" and g.activation == "armed"

        # 状态机：completed 后不能 resume
        g = store.update_goal(g.goal_id, g.revision, "complete")
        with pytest.raises(GoalStateError):
            store.update_goal(g.goal_id, g.revision, "resume")

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "goals.json")
        store1 = GoalStore(path)
        g = store1.create_goal("持久化目标")
        store2 = GoalStore(path)
        g2 = store2.get_goal(g.goal_id)
        assert g2 is not None and g2.objective == "持久化目标"


class TestTodos:
    def test_replace_and_mark(self, tmp_path):
        store = TodoStore(str(tmp_path))
        store.replace(
            "scope-1",
            [
                {"content": "a", "status": "in_progress"},
                {"content": "b"},
            ],
        )
        assert store.mark("scope-1", "a", "completed")
        assert not store.mark("scope-1", "不存在", "completed")
        summary = store.summary("scope-1")
        assert summary["completed"] == 1 and summary["pending"] == 1


class TestSchema:
    def test_valid(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "steps": {"type": "integer"},
            },
            "required": ["name"],
        }
        assert validate_result({"name": "红烧肉", "steps": 3}, schema).valid

    def test_wrong_type(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        r = validate_result({"name": 42}, schema)
        assert not r.valid and "期望类型 string" in r.errors[0]

    def test_missing_required(self):
        schema = {"type": "object", "required": ["a", "b"]}
        r = validate_result({"a": 1}, schema)
        assert not r.valid and "缺少必填字段" in r.errors[0]

    def test_enum_and_oneof(self):
        assert validate_result("fast", {"enum": ["fast", "normal"]}).valid
        assert validate_result("slow", {"enum": ["fast", "normal"]}).errors
        oneof = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
        assert validate_result(3, oneof).valid


class TestJobs:
    def test_completed_output(self):
        async def main():
            mgr = JobManager()

            async def work(n):
                await asyncio.sleep(0.01)
                return n * 2

            jid = mgr.start("work", work, 21)
            st = await mgr.output(jid, wait=True, timeout_ms=5000)
            assert st["status"] == "completed"
            assert st["output"] == 42

        asyncio.run(main())

    def test_kill_before_start(self):
        # Python 3.9 下取消未启动任务不会触发协程异常处理，
        # kill() 必须同步置为 KILLED
        async def main():
            mgr = JobManager()

            async def slow():
                await asyncio.sleep(10)
                return "x"

            jid = mgr.start("slow", slow)
            assert mgr.kill(jid, "test")
            st = await mgr.output(jid, wait=True, timeout_ms=5000)
            assert st["status"] == "killed"

        asyncio.run(main())

    def test_failed_job(self):
        async def main():
            mgr = JobManager()

            async def boom():
                raise ValueError("boom")

            jid = mgr.start("boom", boom)
            st = await mgr.output(jid, wait=True, timeout_ms=5000)
            assert st["status"] == "failed"
            assert "boom" in st["error"]

        asyncio.run(main())


class TestWorkflow:
    def test_parallel(self):
        async def val(v, d):
            await asyncio.sleep(d)
            return v

        async def main():
            outs = await parallel([lambda: val("a", 0.01), lambda: val("b", 0.02)])
            assert outs == ["a", "b"]

        asyncio.run(main())

    def test_pipeline(self):
        async def double(prev, item, i):
            return prev * 2

        async def plus1(prev, item, i):
            return prev + 1

        async def main():
            res = await pipeline([1, 2, 3], double, plus1)
            assert res == [3, 5, 7]

        asyncio.run(main())

    def test_phases_with_schema(self):
        async def val(v, d):
            await asyncio.sleep(d)
            return v

        schema = {
            "type": "object",
            "properties": {"hits": {"type": "integer"}},
            "required": ["hits"],
        }

        async def main():
            phases = await run_phases(
                [
                    Phase("检索", jobs=[lambda: val({"hits": 5}, 0.01)], result_schema=schema),
                ]
            )
            assert phases["检索"][0]["hits"] == 5

        asyncio.run(main())
