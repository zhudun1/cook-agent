# app/harness/workflow.py
"""
多阶段编排原语（Workflow）

借鉴 DeepSeek Harness 的 workflow 思想：
- phase：阶段（含标题/描述），阶段内并发 fan-out，阶段间有 barrier
- pipeline：把 items 依序经过多个 stage，stage 之间无 barrier（流水线）
- parallel：并发执行一批 thunk，全部完成（barrier）
- 结构化结果：子任务返回对象时可带 schema 校验，失败即 null（不炸整条流水线）

用法::

    async def stage1(item, index): ...
    results = await pipeline(items, stage1, stage2)

    outs = await parallel([lambda: f(), lambda: g()])

    await run_phases([
        Phase("检索", "并行检索多路", jobs=[...]),
        Phase("生成", "汇总生成", jobs=[...]),
    ])
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.harness.schema import validate_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

async def pipeline(
    items: List[Any],
    *stages: Callable[[Any, Any, int], Awaitable[Any]],
) -> List[Any]:
    """
    将每个 item 依次经过所有 stage（无 barrier 的流水线）。

    - 每个 stage 收到 (prev_result, item, index)
    - stage 抛异常 -> 该 item 的结果为 None，跳过剩余 stage
    - 全部 stage 成功 -> item 的最终结果

    Args:
        items: 输入列表
        *stages: 阶段函数（async）

    Returns:
        与 items 等长的结果列表
    """
    results: List[Any] = []
    for index, item in enumerate(items):
        prev: Any = item
        failed = False
        for stage in stages:
            try:
                prev = await stage(prev, item, index)
            except Exception as e:
                logger.warning("pipeline stage %s failed for item %d: %s",
                               getattr(stage, "__name__", stage), index, e)
                prev = None
                failed = True
                break
        results.append(None if failed else prev)
    return results


# ---------------------------------------------------------------------------
# parallel
# ---------------------------------------------------------------------------

async def parallel(thunks: List[Callable[[], Awaitable[Any]]]) -> List[Any]:
    """
    并发执行一批 thunk 并等待全部完成（barrier）。

    抛异常的 thunk 结果为 None（不影响其他 thunk）。

    Args:
        thunks: 零参异步函数列表

    Returns:
        与 thunks 等长的结果列表
    """
    async def _safe(t: Callable[[], Awaitable[Any]]) -> Any:
        try:
            return await t()
        except Exception as e:
            logger.warning("parallel thunk failed: %s", e)
            return None

    return await asyncio.gather(*[_safe(t) for t in thunks])


# ---------------------------------------------------------------------------
# phase-based orchestration
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    """工作流阶段。"""

    title: str
    detail: Optional[str] = None
    jobs: List[Callable[[], Awaitable[Any]]] = field(default_factory=list)
    # 阶段内任务的结构化结果 schema（校验失败 -> null）
    result_schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {"title": self.title, "detail": self.detail, "jobs": len(self.jobs)}


async def run_phases(
    phases: List[Phase],
    *,
    on_phase_start: Optional[Callable[[Phase], None]] = None,
    on_phase_end: Optional[Callable[[Phase, List[Any]], None]] = None,
) -> Dict[str, Any]:
    """
    顺序执行阶段，阶段内并行（barrier 语义）。

    Args:
        phases: 阶段列表
        on_phase_start / on_phase_end: 阶段钩子（日志/进度上报）

    Returns:
        {phase_title: results}
    """
    outputs: Dict[str, Any] = {}
    for phase in phases:
        if on_phase_start:
            on_phase_start(phase)
        logger.info("Workflow phase start: %s", phase.title)

        raw = await parallel(phase.jobs) if phase.jobs else []

        # 结构化结果校验（失败 -> None）
        if phase.result_schema:
            checked = []
            for r in raw:
                if r is None:
                    checked.append(None)
                    continue
                vr = validate_result(r, phase.result_schema)
                if vr.valid:
                    checked.append(r)
                else:
                    logger.warning(
                        "Phase %s produced invalid result: %s",
                        phase.title, "; ".join(vr.errors[:3]),
                    )
                    checked.append(None)
            raw = checked

        outputs[phase.title] = raw
        if on_phase_end:
            on_phase_end(phase, raw)
        logger.info("Workflow phase end: %s (%d results)", phase.title, len(raw))
    return outputs


# ---------------------------------------------------------------------------
# structured sub-task helper
# ---------------------------------------------------------------------------

async def run_structured(
    fn: Callable[[], Awaitable[Any]],
    schema: Dict[str, Any],
) -> Optional[Any]:
    """
    执行一个返回结构化对象的子任务并校验。

    Args:
        fn: 异步函数
        schema: 结果 schema

    Returns:
        校验通过的对象；执行失败或校验失败返回 None
    """
    try:
        result = await fn()
        vr = validate_result(result, schema)
        if not vr.valid:
            logger.warning("Structured result invalid: %s", "; ".join(vr.errors[:3]))
            return None
        return result
    except Exception as e:
        logger.warning("Structured task failed: %s", e)
        return None
