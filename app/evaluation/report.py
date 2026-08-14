from __future__ import annotations


# app/evaluation/report.py
"""
评测报告生成：JSON + Markdown。

JSON 报告：data/evaluation_reports/<ts>_<dataset>.json
Markdown 报告：data/evaluation_reports/<ts>_<dataset>.md

包含：逐条得分、指标汇总、失败用例清单、检索上下文与标准答案对照。
"""


import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_report(
    all_results: Dict[str, Any],
    report_dir: str,
) -> str:
    """
    生成 JSON + Markdown 报告。

    Args:
        all_results: run_all() 的返回（dataset_name -> 结果）
        report_dir: 报告输出目录

    Returns:
        主报告文件路径（JSON）
    """
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{_ts()}_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    md_path = out_dir / f"{_ts()}_evaluation_report.md"
    md_path.write_text(
        _render_markdown(all_results),
        encoding="utf-8",
    )

    logger.info("Evaluation report generated: %s, %s", json_path, md_path)
    return str(json_path)


def _render_markdown(all_results: Dict[str, Any]) -> str:
    """渲染 Markdown 报告。"""
    lines = [
        "# RAG Grounding-Truth 评测报告",
        "",
        f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    for dataset_name, result in all_results.items():
        if dataset_name.startswith("_"):
            continue
        lines.append(f"## 测试集: {dataset_name}")
        lines.append("")
        summary = result.get("summary", result)
        metrics = summary.get("metrics", {})
        if metrics:
            lines.append("### 指标汇总")
            lines.append("")
            lines.append("| 指标 | Mean | Min | Max | 有效样本 |")
            lines.append("| --- | --- | --- | --- | --- |")
            for name, stat in metrics.items():
                lines.append(
                    f"| {name} | {stat.get('mean')} | {stat.get('min')} | "
                    f"{stat.get('max')} | {stat.get('count')} |"
                )
            lines.append("")
        lines.append(
            f"- 总用例: {summary.get('total_cases')} / "
            f"成功评测: {summary.get('evaluated_cases')} / "
            f"失败: {summary.get('failed_cases')}"
        )
        lines.append("")

        failures = result.get("failures", [])
        if failures:
            lines.append(f"### 失败用例（{len(failures)}）")
            lines.append("")
            for i, f in enumerate(failures, 1):
                lines.append(f"{i}. **{f['question']}**")
                lines.append(f"   - 原因: {', '.join(f['reasons'])}")
            lines.append("")

        cases = result.get("cases", [])
        if cases:
            lines.append(f"### 逐条明细（{len(cases)}）")
            lines.append("")
            for i, c in enumerate(cases, 1):
                metrics_str = (
                    ", ".join(f"{k}={v:.2f}" for k, v in (c.get("metrics") or {}).items() if v is not None)
                    or "N/A"
                )
                lines.append(
                    f"{i}. **{c['question']}** — {metrics_str}"
                )
                if c.get("error"):
                    lines.append(f"   - ⛔ {c['error']}")
                if c.get("generated_answer"):
                    lines.append(
                        f"   - 生成答案: {c['generated_answer'][:120]}"
                    )
                lines.append(
                    f"   - 标准答案: {c.get('reference_answer', '')[:120]}"
                )
                lines.append(
                    f"   - 检索上下文数: {len(c.get('retrieved_contexts', []))} / "
                    f"标准上下文数: {len(c.get('reference_contexts', []))}"
                )
            lines.append("")

    return "\n".join(lines)
