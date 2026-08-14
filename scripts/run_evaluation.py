#!/usr/bin/env python3
from __future__ import annotations


"""
RAG Grounding-Truth 离线评测 CLI

用法:
    python -m scripts.run_evaluation                      # 评测 testsets 目录下全部测试集
    python -m scripts.run_evaluation --testset testsets/cooking_basics.jsonl
    python -m scripts.run_evaluation --no-persist --no-report
    python -m scripts.run_evaluation --dry-run            # 只加载并校验测试集，不评测

选项:
    --testset PATH   指定单个测试集文件（默认评测配置目录下全部）
    --no-persist     不将结果写入数据库
    --no-report      不生成报告文件
    --dry-run        仅校验测试集格式
"""


import argparse
import asyncio
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main(args: argparse.Namespace) -> int:
    from app.config import settings
    from app.evaluation.dataset import GroundTruthDataset
    from app.evaluation.runner import OfflineEvaluationRunner

    config = settings.evaluation

    if args.dry_run:
        if args.testset:
            datasets = [GroundTruthDataset.load(args.testset)]
        else:
            datasets = GroundTruthDataset.load_dir(config.testsets_dir)
        for ds in datasets:
            print(f"[OK] {ds.summary()}")
        return 0

    runner = OfflineEvaluationRunner()

    if args.testset:
        dataset = GroundTruthDataset.load(args.testset)
        result = await runner.run_dataset(dataset, persist=args.persist)

        if args.report:
            from app.evaluation.report import generate_report

            path = await asyncio.to_thread(
                generate_report,
                {dataset.name: result},
                config.report_dir,
            )
            print(f"\n报告: {path}")

        print(json.dumps(result.get("summary", {}), ensure_ascii=False, indent=2))
    else:
        results = await runner.run_all(persist=args.persist, report=args.report)
        for name, res in results.items():
            if name.startswith("_"):
                continue
            print(f"\n=== {name} ===")
            print(json.dumps(res.get("summary", {}), ensure_ascii=False, indent=2))
        if results.get("_report_path"):
            print(f"\n报告: {results['_report_path']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG grounding-truth offline evaluation")
    parser.add_argument("--testset", type=str, default=None, help="single testset file")
    parser.add_argument("--no-persist", action="store_false", dest="persist", default=True)
    parser.add_argument("--no-report", action="store_false", dest="report", default=True)
    parser.add_argument("--dry-run", action="store_true", help="validate testsets only")
    args = parser.parse_args()

    sys.exit(asyncio.run(main(args)))
