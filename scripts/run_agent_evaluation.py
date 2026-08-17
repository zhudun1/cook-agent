#!/usr/bin/env python3
"""
任务级端到端评测 CLI（P1：任务完成率 + 回归拦截）

用法:
    python -m scripts.run_agent_evaluation                        # 评测全部任务集
    python -m scripts.run_agent_evaluation --taskset testsets/agent_tasks.jsonl
    python -m scripts.run_agent_evaluation --save-baseline data/evaluation_reports/task_baseline.json
    python -m scripts.run_agent_evaluation --baseline data/evaluation_reports/task_baseline.json --threshold 0.05
    python -m scripts.run_agent_evaluation --dry-run              # 仅校验任务集格式

选项:
    --taskset PATH      指定单个任务集文件
    --baseline PATH     基准完成率文件（对比回归）
    --threshold FLOAT   回归判定阈值（完成率下降超过该值标记 REGRESSION，默认 0.05）
    --save-baseline PATH 将本次完成率保存为新的基准
    --user-id STR       评测用户 ID
    --dry-run           仅校验任务集
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def main(args: argparse.Namespace) -> int:
    from app.config import settings
    from app.evaluation.task_runner import (
        AgentTaskDataset,
        TaskEvaluationRunner,
        TaskDatasetError,
    )

    config = settings.evaluation

    if args.dry_run:
        if args.taskset:
            datasets = [AgentTaskDataset.load(args.taskset)]
        else:
            datasets = AgentTaskDataset.load_dir(config.task_testsets_dir)
        for ds in datasets:
            print(f"[OK] {ds.name}: {len(ds)} tasks")
        return 0

    runner = TaskEvaluationRunner()
    summary = {"datasets": {}}

    if args.taskset:
        try:
            dataset = AgentTaskDataset.load(args.taskset)
        except TaskDatasetError as e:
            print(f"ERROR: {e}")
            return 1
        result = await runner.run_dataset(dataset, user_id=args.user_id)
        summary["datasets"][dataset.name] = result
        print(f"\n=== {dataset.name} ===")
        print(f"完成率: {result['completion_rate']:.1%} ({result['achieved']}/{result['total_tasks']})")
    else:
        all_results = await runner.run_all(
            tasksets_dir=getattr(config, "task_testsets_dir", "testsets/tasks"),
            user_id=args.user_id,
            baseline_path=args.baseline,
            regression_threshold=args.threshold,
        )
        summary = all_results
        for name, res in all_results["datasets"].items():
            print(f"\n=== {name} ===")
            print(f"完成率: {res['completion_rate']:.1%} ({res['achieved']}/{res['total_tasks']})")
            for f in res.get("failures", []):
                print(f"  ❌ {f['task'][:60]} -> {f['reason'][:80]}")

    reg = summary.get("regression")
    if reg and reg.get("has_baseline"):
        if reg["ok"]:
            print("\n✅ 无回归")
        else:
            print("\n⚠️ 检测到回归!")
            for r in reg["regressions"]:
                print(f"  {r['dataset']}: {r['baseline_rate']:.1%} -> {r['current_rate']:.1%} ({r['delta']:+.1%})")
            if args.fail_on_regression:
                return 2

    if args.save_baseline:
        baseline = {}
        for name, res in summary["datasets"].items():
            baseline[name] = {
                "completion_rate": res["completion_rate"],
                "total_tasks": res["total_tasks"],
            }
        import os

        os.makedirs(os.path.dirname(args.save_baseline) or ".", exist_ok=True)
        with open(args.save_baseline, "w", encoding="utf-8") as f:
            json.dump(baseline, f, ensure_ascii=False, indent=2)
        print(f"\n基准已保存: {args.save_baseline}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent task-level end-to-end evaluation")
    parser.add_argument("--taskset", type=str, default=None, help="single task dataset file")
    parser.add_argument("--baseline", type=str, default=None, help="baseline completion rates file")
    parser.add_argument("--threshold", type=float, default=0.05, help="regression threshold")
    parser.add_argument("--save-baseline", type=str, default=None, help="save baseline to file")
    parser.add_argument("--user-id", type=str, default="task-eval-user")
    parser.add_argument("--dry-run", action="store_true", help="validate task datasets only")
    parser.add_argument("--fail-on-regression", action="store_true", help="exit code 2 on regression")
    args = parser.parse_args()

    sys.exit(asyncio.run(main(args)))
