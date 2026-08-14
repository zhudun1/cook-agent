# app/api/v1/endpoints/evaluation.py
"""
API endpoints for RAG evaluation metrics.
Provides access to evaluation statistics, trends, and quality alerts.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from app.config import settings
from app.database.evaluation_repository import evaluation_repository

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.get("/statistics")
async def get_evaluation_statistics(
    request: Request,
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
):
    """
    Get aggregated evaluation statistics.

    Returns statistics including:
    - Total evaluations count
    - Average metrics (faithfulness, answer_relevancy)
    - Min/max values for key metrics
    - Pending and failed evaluation counts

    Query Parameters:
        start_date: Filter by start date (ISO format, e.g., "2024-01-01")
        end_date: Filter by end date (ISO format, e.g., "2024-01-31")
    """
    # Get user_id from auth if available
    user_id = getattr(request.state, "user_id", None)

    # Parse dates
    start_dt = None
    end_dt = None
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format")
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format")

    stats = await evaluation_repository.get_statistics(
        user_id=user_id,
        start_date=start_dt,
        end_date=end_dt,
    )

    return stats


@router.get("/conversation/{conversation_id}")
async def get_conversation_evaluations(
    request: Request,
    conversation_id: str,
    limit: int = Query(100, ge=1, le=500),
):
    """
    Get evaluation details for a specific conversation.

    Returns a list of evaluation records for the conversation,
    including metrics and status for each evaluated message.
    """
    evaluations = await evaluation_repository.get_by_conversation(
        conversation_id=conversation_id,
        limit=limit,
    )

    return {
        "conversation_id": conversation_id,
        "count": len(evaluations),
        "evaluations": evaluations,
    }


@router.get("/trends")
async def get_evaluation_trends(
    request: Request,
    days: int = Query(7, ge=1, le=90, description="Number of days to look back"),
    granularity: str = Query("day", regex="^(day|hour)$", description="Grouping granularity"),
):
    """
    Get evaluation metric trends over time.

    Returns time-series data for evaluation metrics,
    grouped by day or hour.

    Query Parameters:
        days: Number of days to look back (1-90)
        granularity: Grouping granularity ("day" or "hour")
    """
    user_id = getattr(request.state, "user_id", None)

    trends = await evaluation_repository.get_trends(
        days=days,
        granularity=granularity,
        user_id=user_id,
    )

    return {
        "period_days": days,
        "granularity": granularity,
        "data_points": len(trends),
        "trends": trends,
    }


@router.get("/alerts")
async def get_quality_alerts(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
):
    """
    Get evaluations that fall below quality thresholds.

    Returns a list of evaluation records where one or more metrics
    are below the configured alert thresholds.

    The thresholds are defined in the application configuration.
    """
    user_id = getattr(request.state, "user_id", None)

    # Get thresholds from config
    thresholds = {
        "faithfulness": settings.evaluation.alert_thresholds.faithfulness,
        "answer_relevancy": settings.evaluation.alert_thresholds.answer_relevancy,
    }

    alerts = await evaluation_repository.get_alerts(
        thresholds=thresholds,
        limit=limit,
        user_id=user_id,
    )

    return {
        "thresholds": thresholds,
        "count": len(alerts),
        "alerts": alerts,
    }


@router.get("/health")
async def evaluation_health():
    """
    Check the health status of the evaluation system.

    Returns:
    - enabled: Whether evaluation is enabled
    - async_mode: Whether async evaluation is enabled
    - sample_rate: Current sampling rate
    - configured_metrics: List of configured metrics
    """
    return {
        "enabled": settings.evaluation.enabled,
        "async_mode": settings.evaluation.async_mode,
        "sample_rate": settings.evaluation.sample_rate,
        "configured_metrics": settings.evaluation.metrics,
        "timeout_seconds": settings.evaluation.timeout_seconds,
        "alert_thresholds": {
            "faithfulness": settings.evaluation.alert_thresholds.faithfulness,
            "answer_relevancy": settings.evaluation.alert_thresholds.answer_relevancy,
        },
    }


# =============================================================================
# P1 任务级评测 API（端到端任务完成率）
# =============================================================================

@router.post("/tasks/run")
async def run_task_evaluation(
    request: Request,
    taskset: Optional[str] = Query(None, description="任务集文件路径（默认全部）"),
    background: bool = Query(True, description="后台运行（推荐）"),
):
    """
    触发任务级端到端评测（黄金任务集 -> Agent 完整执行 -> 判定完成率）。

    **Parameters:**
    - `taskset`: 可选，指定单个任务集文件；默认评测 testsets 下全部
    - `background`: 后台运行（默认 true，立即返回 job_id）；false 同步等待

    Returns:
        background=true: {"job_id": ...}（轮询 GET /evaluation/tasks/latest）
        background=false: 完整评测结果
    """
    from app.evaluation.task_runner import (
        AgentTaskDataset,
        TaskEvaluationRunner,
        get_task_evaluation_runner,
    )
    from app.harness.jobs import job_manager

    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="需要登录")

    runner = get_task_evaluation_runner()

    async def _run():
        if taskset:
            dataset = AgentTaskDataset.load(taskset)
            result = await runner.run_dataset(dataset, user_id=user_id)
            return {"datasets": {dataset.name: result}, "regression": None}
        return await runner.run_all(user_id=user_id)

    if background:
        job_id = job_manager.start("task-evaluation", _run)
        return {"job_id": job_id, "status": "running", "message": "任务评测已提交后台运行"}
    result = await _run()
    return result


@router.get("/tasks/latest")
async def get_latest_task_evaluation(
    request: Request,
    job_id: Optional[str] = Query(None, description="后台运行返回的 job_id"),
):
    """
    查询最近一次任务评测结果（或指定后台任务）。

    **Parameters:**
    - `job_id`: 可选，指定后台任务 ID
    """
    from app.harness.jobs import job_manager

    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="需要登录")

    if job_id:
        job = await job_manager.output(job_id)
        if not job or job.get("status") == "unknown":
            raise HTTPException(status_code=404, detail="任务不存在")
        return job
    # 最近一次（取最后一个 task-evaluation 任务，含 output）
    jobs = [
        j for j in job_manager.list() if j["name"] == "task-evaluation"
    ]
    if not jobs:
        return {"status": "no_task_run", "message": "尚未运行过任务评测"}
    last = await job_manager.output(jobs[-1]["job_id"])
    return last


# =============================================================================
# P1 可观测性: 工具级 SLO（成功率 / 延迟 / P95）
# =============================================================================

@router.get("/tool-metrics")
async def get_tool_metrics(
    request: Request,
    tool: Optional[str] = Query(None, description="按工具名过滤"),
    minutes: int = Query(60, ge=1, le=10080, description="统计最近 N 分钟"),
):
    """
    工具调用 SLO 统计（成功率 / 平均延迟 / P50 / P95 / 错误码分布）。

    **Parameters:**
    - `tool`: 可选，只统计指定工具
    - `minutes`: 统计窗口（分钟，默认 60）
    """
    from app.evaluation.tool_metrics import tool_metrics

    since = None
    if minutes:
        import time as _time

        since = _time.time() - minutes * 60

    stats = tool_metrics.get_stats(tool_name=tool, since=since)
    return {
        "window_minutes": minutes,
        "tool": tool,
        "stats": stats,
    }


@router.get("/{evaluation_id}")
async def get_evaluation_detail(
    request: Request,
    evaluation_id: str,
):
    """
    Get detailed information for a single evaluation.

    Returns the full evaluation record including:
    - Original query and response
    - Context used for generation
    - All metric scores
    - Evaluation status and timing
    """
    evaluation = await evaluation_repository.get_by_id(evaluation_id)

    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return evaluation
