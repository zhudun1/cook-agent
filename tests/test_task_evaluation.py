# tests/test_task_evaluation.py
"""
P1 任务级评测单元测试：工具 SLO 采集 / 任务集加载 / 启发式判定。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import pytest
import asyncio

from app.evaluation.tool_metrics import ToolMetricsCollector
from app.evaluation.task_runner import (
    AgentTask,
    AgentTaskDataset,
    TaskDatasetError,
    TaskJudge,
)


class TestToolMetrics:
    def test_record_and_aggregate(self):
        async def main():
            c = ToolMetricsCollector(max_records=100)
            await c.record("calculator", success=True, duration_ms=50)
            await c.record("calculator", success=True, duration_ms=150)
            await c.record("calculator", success=False, duration_ms=30000, error_code="TIMEOUT")

            stats = await c.get_stats("calculator")
            agg = stats["tools"]["calculator"]
            assert agg["calls"] == 3
            assert abs(agg["success_rate"] - 2 / 3) < 0.001  # round(..., 4) 后比较
            assert agg["error_codes"] == {"TIMEOUT": 1}
            assert agg["avg_duration_ms"] == pytest.approx((50 + 150 + 30000) / 3)
            assert agg["p95_ms"] is not None
            await c.reset()

        asyncio.run(main())

    def test_per_tool_and_totals(self):
        async def main():
            c = ToolMetricsCollector(max_records=100)
            await c.record("calculator", success=True, duration_ms=10)
            await c.record("datetime", success=True, duration_ms=20)
            stats = await c.get_stats()
            assert set(stats["tools"].keys()) == {"calculator", "datetime"}
            assert stats["totals"]["calls"] == 2
            await c.reset()

        asyncio.run(main())

    def test_window_since(self):
        async def main():
            c = ToolMetricsCollector(max_records=100)
            await c.record("calculator", success=True, duration_ms=10)
            stats = await c.get_stats(since=0)  # 全部
            assert stats["totals"]["calls"] == 1
            stats_now = await c.get_stats(since=2 ** 31)  # 未来时间戳 -> 无记录
            assert stats_now["totals"]["calls"] == 0
            await c.reset()

        asyncio.run(main())


class TestAgentTaskDataset:
    def test_load(self, tmp_path):
        p = tmp_path / "tasks.jsonl"
        p.write_text(
            json.dumps(
                {
                    "task": "计算 15*4+2",
                    "expected_outcome": "包含 62",
                    "tools_required": ["calculator"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        ds = AgentTaskDataset.load(p)
        assert len(ds) == 1
        assert ds.tasks[0].tools_required == ["calculator"]

    def test_missing_required_fields(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"task": "没有预期结果"}\n', encoding="utf-8")
        with pytest.raises(TaskDatasetError):
            AgentTaskDataset.load(p)

    def test_missing_file(self):
        with pytest.raises(TaskDatasetError):
            AgentTaskDataset.load("no_such_tasks.jsonl")

    def test_real_sample(self):
        ds = AgentTaskDataset.load("testsets/agent_tasks.jsonl")
        assert len(ds) == 4
        assert all(t.tools_required for t in ds.tasks)


class TestTaskJudgeHeuristic:
    def test_achieved_with_keywords_and_tools(self):
        judge = TaskJudge()
        task = AgentTask(
            task="计算 15*4+2",
            expected_outcome="回答应包含数字 62",
            tools_required=["calculator"],
        )
        trace = [
            {"action": "tool_call", "name": "calculator"},
            {"action": "tool_result", "name": "calculator", "content": '{"result": 62}'},
        ]
        result = judge._judge_heuristic(task, "结果是 62。", trace)
        assert result["achieved"] is True
        assert result["judge"] == "heuristic"

    def test_failed_missing_tool(self):
        judge = TaskJudge()
        task = AgentTask(
            task="搜索红烧肉做法",
            expected_outcome="包含做法",
            tools_required=["web_search"],
        )
        trace = [
            {"action": "tool_call", "name": "calculator"},
            {"action": "tool_result", "name": "calculator", "content": "42"},
        ]
        result = judge._judge_heuristic(task, "红烧肉的做法是焯水。", trace)
        assert result["achieved"] is False
        assert "web_search" in result["reason"]

    def test_failed_keyword_miss(self):
        judge = TaskJudge()
        task = AgentTask(
            task="计算 15*4+2",
            expected_outcome="回答应包含数字 62",
            tools_required=["calculator"],
        )
        trace = [
            {"action": "tool_result", "name": "calculator", "content": '{"result": 62}'},
        ]
        result = judge._judge_heuristic(task, "我不知道。", trace)
        assert result["achieved"] is False
        assert "关键词" in result["reason"]

    def test_no_tools_required_keyword_only(self):
        judge = TaskJudge()
        # 预期结果使用可字面命中的关键词（启发式判定依赖字面命中）
        task = AgentTask(task="你好", expected_outcome="回答中应包含你好")
        result = judge._judge_heuristic(task, "你好！有什么可以帮你？", [])
        assert result["achieved"] is True


class TestRegressionCheck:
    def test_regression_detected(self, tmp_path):
        from app.evaluation.task_runner import TaskEvaluationRunner

        baseline_path = str(tmp_path / "baseline.json")
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump({"tasks": {"completion_rate": 0.9}}, f)

        results = {"tasks": {"completion_rate": 0.6}}
        reg = TaskEvaluationRunner._check_regression(results, baseline_path, 0.05)
        assert reg["has_baseline"]
        assert len(reg["regressions"]) == 1
        assert reg["regressions"][0]["status"] == "REGRESSION"
        assert reg["ok"] is False

    def test_no_regression(self, tmp_path):
        from app.evaluation.task_runner import TaskEvaluationRunner

        baseline_path = str(tmp_path / "baseline.json")
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump({"tasks": {"completion_rate": 0.6}}, f)

        results = {"tasks": {"completion_rate": 0.7}}
        reg = TaskEvaluationRunner._check_regression(results, baseline_path, 0.05)
        assert reg["ok"] is True
        assert reg["regressions"] == []
