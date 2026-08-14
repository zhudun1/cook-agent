# app/evaluation/task_runner.py
"""
任务级端到端评测（Task-Level Evaluation）

生产级 Agent 评测的第二层：用黄金任务集跑完整 Agent 流程，
判定"任务是否达成"（而不是回答好不好）。

数据集（tasksets/*.jsonl）行格式::

    {
      "task": "帮我计算 15*4+2 等于多少",
      "expected_outcome": "最终回答应包含数字 62",
      "tools_required": ["calculator"],
      "metadata": {"difficulty": "easy"}
    }

判定器：
1. LLM 判定（推荐）：把 task + expected_outcome + Agent 完整轨迹 + 最终回答
   交给评测 LLM，输出 {achieved: bool, reason}
2. 启发式回退（无 LLM key 时）：关键词命中 + 必需工具成功调用

回归拦截：--baseline 对比完成率，下降超阈值标记 REGRESSION。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.config.evaluation_config import EvaluationConfig

logger = logging.getLogger(__name__)


class TaskDatasetError(ValueError):
    """任务集校验失败。"""


@dataclass
class AgentTask:
    """一条端到端评测任务。"""

    task: str
    expected_outcome: str
    tools_required: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "expected_outcome": self.expected_outcome,
            "tools_required": self.tools_required,
            "metadata": self.metadata,
        }


class AgentTaskDataset:
    """黄金任务集（JSONL 加载 + 校验）。"""

    def __init__(self, name: str, tasks: List[AgentTask]):
        self.name = name
        self.tasks = tasks

    @classmethod
    def load(cls, path: str | Path) -> "AgentTaskDataset":
        p = Path(path)
        if not p.exists():
            raise TaskDatasetError(f"Task dataset not found: {p}")
        tasks: List[AgentTask] = []
        with open(p, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise TaskDatasetError(f"{p}:{lineno} invalid JSON: {e}") from e
                if not obj.get("task") or not obj.get("expected_outcome"):
                    raise TaskDatasetError(
                        f"{p}:{lineno} requires 'task' and 'expected_outcome'"
                    )
                tasks.append(
                    AgentTask(
                        task=obj["task"],
                        expected_outcome=obj["expected_outcome"],
                        tools_required=obj.get("tools_required", []) or [],
                        metadata=obj.get("metadata", {}) or {},
                    )
                )
        if not tasks:
            raise TaskDatasetError(f"Task dataset is empty: {p}")
        return cls(name=p.stem, tasks=tasks)

    @classmethod
    def load_dir(cls, directory: str | Path) -> List["AgentTaskDataset"]:
        d = Path(directory)
        if not d.exists():
            logger.warning("Task datasets dir not found: %s", d)
            return []
        datasets = []
        for p in sorted(d.glob("*.jsonl")):
            try:
                datasets.append(cls.load(p))
            except TaskDatasetError as e:
                logger.error("Skipping task dataset %s: %s", p, e)
        return datasets

    def __len__(self) -> int:
        return len(self.tasks)


# ---------------------------------------------------------------------------
# 判定器
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """你是一个 Agent 任务评测器。根据【任务】【预期结果】与【Agent 执行轨迹】，
判定任务是否成功达成。

判定规则：
- achieved=true 当且仅当 Agent 的最终回答确实满足了预期结果（数值/事实/操作完成）
- 中间过程出错但最终给出了正确回答 -> achieved=true
- 最终回答缺失、明显错误或未完成任务 -> achieved=false
- 只输出 JSON：{"achieved": true/false, "reason": "一句话理由"}
"""


class TaskJudge:
    """
    任务达成判定器：LLM 判定 + 启发式回退。
    """

    def __init__(self, llm_type: str = "fast"):
        self.llm_type = llm_type

    async def judge(
        self,
        task: AgentTask,
        final_answer: str,
        trace: List[dict],
    ) -> Dict[str, Any]:
        """判定任务是否达成。优先 LLM，失败回退启发式。"""
        llm_result = await self._judge_with_llm(task, final_answer, trace)
        if llm_result is not None:
            return llm_result
        return self._judge_heuristic(task, final_answer, trace)

    async def _judge_with_llm(
        self,
        task: AgentTask,
        final_answer: str,
        trace: List[dict],
    ) -> Optional[Dict[str, Any]]:
        """LLM 判定（失败返回 None）。"""
        try:
            from app.llm.provider import LLMProvider

            provider = LLMProvider(settings.llm)
            invoker = provider.create_invoker(llm_type=self.llm_type, temperature=0.0)

            trace_text = json.dumps(trace, ensure_ascii=False, default=str)[:6000]
            user_prompt = (
                f"【任务】\n{task.task}\n\n"
                f"【预期结果】\n{task.expected_outcome}\n\n"
                f"【Agent 执行轨迹】\n{trace_text or '(无)'}\n\n"
                f"【最终回答】\n{final_answer or '(无)'}\n\n"
                "请判定任务是否达成，只输出 JSON。"
            )
            response = await invoker.ainvoke(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            content = getattr(response, "content", "") or ""
            from app.utils.structured_json import parse_json_auto

            result = parse_json_auto(content)
            return {
                "achieved": bool(result.get("achieved")),
                "reason": str(result.get("reason", ""))[:200],
                "judge": "llm",
            }
        except Exception as e:
            logger.debug("LLM judge failed, falling back to heuristic: %s", e)
            return None

    def _judge_heuristic(
        self,
        task: AgentTask,
        final_answer: str,
        trace: List[dict],
    ) -> Dict[str, Any]:
        """启发式判定：必需工具成功调用 + 预期结果关键词命中。"""
        reasons: List[str] = []
        achieved = True

        # 1. 必需工具必须成功调用
        called_tools = {
            t.get("name")
            for t in trace
            if t.get("action") == "tool_result" and t.get("content") is not None
        }
        for tool in task.tools_required:
            if tool not in called_tools:
                achieved = False
                reasons.append(f"必需工具 '{tool}' 未被成功调用")
            else:
                reasons.append(f"工具 '{tool}' 已成功调用")

        # 2. 预期结果关键词命中
        # 提取策略：引号/书名号内容 > "包含X" 模式 > 数字 > 标点分词
        answer = final_answer or ""
        keywords: List[str] = []
        # 引号/书名号包裹的明确目标
        keywords += re.findall(r"[「\"'“”『』]([^」\"'“”『』]{2,30})[」\"'“”『』]", task.expected_outcome)
        # "包含X / 应包含X / 包括X" 模式
        keywords += re.findall(
            r"(?:包含|应包含|包括|含有)\s*([^，。；,;、\s]{2,20})",
            task.expected_outcome,
        )
        # 数字目标（如 "62"）
        keywords += re.findall(r"\d{1,10}", task.expected_outcome)
        # 兜底：按标点分词
        if not keywords:
            keywords = [
                kw.strip()
                for kw in re.split(r"[，。；、,; ]+", task.expected_outcome)
                if len(kw.strip()) >= 2
            ]
        # 去重保序
        seen = set()
        keywords = [k for k in keywords if not (k in seen or seen.add(k))]

        hit = [kw for kw in keywords if kw in answer]
        if not hit:
            achieved = False
            reasons.append(f"最终回答未命中预期结果关键词: {keywords[:3]}")
        else:
            reasons.append(f"命中关键词: {hit[:3]}")

        return {
            "achieved": achieved,
            "reason": "; ".join(reasons),
            "judge": "heuristic",
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TaskEvaluationRunner:
    """
    任务级端到端评测执行器。

    用法::

        runner = TaskEvaluationRunner()
        result = await runner.run_dataset(dataset)
        result = await runner.run_all()
    """

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        agent_service: Any = None,
    ):
        self.config = config or settings.evaluation
        if agent_service is None:
            from app.agent.service import agent_service as _svc

            agent_service = _svc
        self.agent_service = agent_service
        self.judge = TaskJudge(llm_type=self.config.llm_type)

    # ------------------------------------------------------------------
    async def _run_single(
        self,
        task: AgentTask,
        dataset_name: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """执行一条任务并判定。"""
        started = time.time()
        final_answer = ""
        trace: List[dict] = []
        session_id: Optional[str] = None
        errors: List[str] = []

        try:
            async for event in self.agent_service.chat(
                session_id=None,
                user_id=user_id,
                message=task.task,
                agent_name="default",
                streaming=False,
            ):
                if event.startswith("data: "):
                    try:
                        data = json.loads(event[6:].strip())
                    except json.JSONDecodeError:
                        continue
                    etype = data.get("type")
                    if etype == "text":
                        final_answer += data.get("content", "")
                    elif etype == "session":
                        session_id = data.get("session_id")
                    elif etype == "tool_call":
                        trace.append(
                            {
                                "action": "tool_call",
                                "name": data.get("name"),
                                "arguments": data.get("arguments"),
                            }
                        )
                    elif etype == "tool_result":
                        trace.append(
                            {
                                "action": "tool_result",
                                "name": data.get("name"),
                                "success": data.get("success"),
                                "content": data.get("result"),
                                "error": data.get("error"),
                            }
                        )
                    elif etype == "error":
                        errors.append(str(data.get("error", "")))
        except Exception as e:
            errors.append(str(e))

        verdict = await self.judge.judge(task, final_answer, trace)

        return {
            "task": task.task,
            "expected_outcome": task.expected_outcome,
            "tools_required": task.tools_required,
            "achieved": verdict["achieved"],
            "reason": verdict.get("reason"),
            "judge": verdict.get("judge"),
            "final_answer": final_answer[:500],
            "trace": trace,
            "errors": errors,
            "session_id": session_id,
            "duration_ms": int((time.time() - started) * 1000),
            "metadata": task.metadata,
        }

    # ------------------------------------------------------------------
    async def run_dataset(
        self,
        dataset: AgentTaskDataset,
        user_id: str = "task-eval-user",
    ) -> Dict[str, Any]:
        """评测一个任务集。"""
        logger.info("Running task evaluation on %s ...", dataset.name)
        rows = []
        for task in dataset.tasks:
            row = await self._run_single(task, dataset.name, user_id)
            rows.append(row)
            status = "✅" if row["achieved"] else "❌"
            logger.info(
                "%s [%s] %s -> %s (%s)",
                status, row["judge"], task.task[:50], row["achieved"], row["reason"][:60],
            )

        achieved_count = sum(1 for r in rows if r["achieved"])
        return {
            "dataset": dataset.name,
            "total_tasks": len(rows),
            "achieved": achieved_count,
            "completion_rate": round(achieved_count / len(rows), 4) if rows else 0,
            "tasks": rows,
            "failures": [r for r in rows if not r["achieved"]],
        }

    async def run_all(
        self,
        tasksets_dir: Optional[str] = None,
        user_id: str = "task-eval-user",
        baseline_path: Optional[str] = None,
        regression_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        评测 tasksets 目录下全部任务集，并做回归拦截。

        Args:
            tasksets_dir: 任务集目录（默认 config.tasksets_dir 复用）
            user_id: 评测用户
            baseline_path: 基准分数文件路径（存在则对比回归）
            regression_threshold: 完成率下降超过该比例标记 REGRESSION

        Returns:
            {datasets: {...}, regression: {...}}
        """
        dir_path = tasksets_dir or self.config.testsets_dir
        datasets = AgentTaskDataset.load_dir(dir_path)
        if not datasets:
            logger.warning("No task datasets found in %s", dir_path)
            return {"datasets": {}, "regression": None}

        results: Dict[str, Any] = {}
        for ds in datasets:
            results[ds.name] = await self.run_dataset(ds, user_id=user_id)

        regression = None
        if baseline_path:
            regression = self._check_regression(
                results, baseline_path, regression_threshold
            )
        return {"datasets": results, "regression": regression}

    # ------------------------------------------------------------------
    @staticmethod
    def _check_regression(
        results: Dict[str, Any],
        baseline_path: str,
        threshold: float,
    ) -> Dict[str, Any]:
        """与基准对比完成率，下降超阈值标记 REGRESSION。"""
        baseline: Dict[str, Any] = {}
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline = json.load(f)
        except Exception as e:
            logger.warning("Failed to load baseline %s: %s", baseline_path, e)
            return {"has_baseline": False, "error": str(e)}

        issues = []
        for name, res in results.items():
            prev = baseline.get(name)
            if prev is None:
                continue
            prev_rate = prev.get("completion_rate", 0)
            new_rate = res.get("completion_rate", 0)
            delta = new_rate - prev_rate
            if delta < -threshold:
                issues.append(
                    {
                        "dataset": name,
                        "baseline_rate": prev_rate,
                        "current_rate": new_rate,
                        "delta": round(delta, 4),
                        "status": "REGRESSION",
                    }
                )
        return {
            "has_baseline": True,
            "baseline_path": baseline_path,
            "threshold": threshold,
            "regressions": issues,
            "ok": not issues,
        }


# 惰性单例（避免模块导入时实例化 -> 拉入 app.agent/langchain 依赖链）
_task_runner_instance: Optional["TaskEvaluationRunner"] = None


def get_task_evaluation_runner() -> "TaskEvaluationRunner":
    """获取 TaskEvaluationRunner 单例（惰性创建）。"""
    global _task_runner_instance
    if _task_runner_instance is None:
        _task_runner_instance = TaskEvaluationRunner()
    return _task_runner_instance
