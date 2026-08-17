# app/api/v1/endpoints/llm_stats.py
"""
API endpoints for LLM usage statistics.
Provides access to token usage, model distribution, and module-level metrics.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from app.database.llm_usage_repository import llm_usage_repository

router = APIRouter(prefix="/llm-stats", tags=["LLM Statistics"])


@router.get("/summary")
async def get_llm_stats_summary(
    request: Request,
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    conversation_id: Optional[str] = Query(None, description="Filter by conversation ID"),
):
    """
    Get aggregated LLM usage summary.

    Returns statistics including:
    - Total API calls count
    - Total tokens used (input, output, total)
    - Average tokens per call
    - Average response duration

    Query Parameters:
        start_date: Filter by start date (ISO format, e.g., "2024-01-01")
        end_date: Filter by end date (ISO format, e.g., "2024-01-31")
        conversation_id: Filter by specific conversation
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

    summary = await llm_usage_repository.get_summary(
        user_id=user_id,
        start_date=start_dt,
        end_date=end_dt,
        conversation_id=conversation_id,
    )

    return summary


@router.get("/time-series")
async def get_llm_stats_time_series(
    request: Request,
    days: int = Query(7, ge=1, le=90, description="Number of days to look back"),
    granularity: str = Query("day", regex="^(day|hour)$", description="Grouping granularity"),
    module_name: Optional[str] = Query(None, description="Filter by module name"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
):
    """
    Get LLM usage trends over time.

    Returns time-series data for LLM usage metrics,
    grouped by day or hour.

    Query Parameters:
        days: Number of days to look back (1-90)
        granularity: Grouping granularity ("day" or "hour")
        module_name: Filter by specific module
        model_name: Filter by specific model
    """
    user_id = getattr(request.state, "user_id", None)

    time_series = await llm_usage_repository.get_time_series(
        days=days,
        granularity=granularity,
        user_id=user_id,
        module_name=module_name,
        model_name=model_name,
    )

    return {
        "period_days": days,
        "granularity": granularity,
        "data_points": len(time_series),
        "time_series": time_series,
    }


@router.get("/distribution/by-module")
async def get_distribution_by_module(
    request: Request,
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
):
    """
    Get LLM usage distribution by module.

    Returns breakdown of API calls and token usage per module.

    Query Parameters:
        start_date: Filter by start date (ISO format)
        end_date: Filter by end date (ISO format)
    """
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

    distribution = await llm_usage_repository.get_distribution_by_module(
        user_id=user_id,
        start_date=start_dt,
        end_date=end_dt,
    )

    return {
        "distribution": distribution,
        "count": len(distribution),
    }


@router.get("/distribution/by-model")
async def get_distribution_by_model(
    request: Request,
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
):
    """
    Get LLM usage distribution by model.

    Returns breakdown of API calls and token usage per model.

    Query Parameters:
        start_date: Filter by start date (ISO format)
        end_date: Filter by end date (ISO format)
    """
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

    distribution = await llm_usage_repository.get_distribution_by_model(
        user_id=user_id,
        start_date=start_dt,
        end_date=end_dt,
    )

    return {
        "distribution": distribution,
        "count": len(distribution),
    }


@router.get("/conversation/{conversation_id}")
async def get_conversation_llm_stats(
    request: Request,
    conversation_id: str,
    limit: int = Query(100, ge=1, le=500),
):
    """
    Get LLM usage details for a specific conversation.

    Returns a list of LLM usage records for the conversation,
    including token counts, models used, and timing information.
    """
    logs = await llm_usage_repository.get_by_conversation(
        conversation_id=conversation_id,
        limit=limit,
    )

    # Calculate totals for the conversation
    total_input_tokens = sum(log.get("input_tokens") or 0 for log in logs)
    total_output_tokens = sum(log.get("output_tokens") or 0 for log in logs)
    total_tokens = sum(log.get("total_tokens") or 0 for log in logs)

    return {
        "conversation_id": conversation_id,
        "count": len(logs),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "logs": logs,
    }


@router.get("/modules")
async def get_available_modules():
    """
    Get list of available module names that have logged LLM usage.

    Returns a list of unique module names from the usage logs.
    """
    modules = await llm_usage_repository.get_distinct_modules()

    return {
        "modules": modules,
        "count": len(modules),
    }


@router.get("/models")
async def get_available_models():
    """
    Get list of available model names that have been used.

    Returns a list of unique model names from the usage logs.
    """
    models = await llm_usage_repository.get_distinct_models()

    return {
        "models": models,
        "count": len(models),
    }


@router.get("/tools")
async def get_available_tools():
    """
    Get list of available tool names that have been used.

    Returns a list of unique tool names from the usage logs.
    """
    tools = await llm_usage_repository.get_distinct_tools()

    return {
        "tools": tools,
        "count": len(tools),
    }


@router.get("/distribution/by-tool")
async def get_distribution_by_tool(
    request: Request,
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    module_name: Optional[str] = Query(None, description="Filter by module name"),
):
    """
    Get LLM usage distribution by tool.

    Returns breakdown of API calls and token usage per tool.

    Query Parameters:
        start_date: Filter by start date
        end_date: Filter by end date
        model_name: Filter by specific model
        module_name: Filter by specific module
    """
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

    distribution = await llm_usage_repository.get_distribution_by_tool(
        user_id=user_id,
        start_date=start_dt,
        end_date=end_dt,
        model_name=model_name,
        module_name=module_name,
    )

    return {
        "distribution": distribution,
        "count": len(distribution),
    }


@router.get("/time-series/tools")
async def get_tool_time_series(
    request: Request,
    days: int = Query(7, ge=1, le=90, description="Number of days to look back"),
    granularity: str = Query("day", regex="^(day|hour)$", description="Grouping granularity"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    module_name: Optional[str] = Query(None, description="Filter by module name"),
):
    """
    Get tool usage trends over time.

    Returns time-series data for tool usage metrics,
    grouped by day or hour.

    Query Parameters:
        days: Number of days to look back (1-90)
        granularity: Grouping granularity ("day" or "hour")
        model_name: Filter by specific model
        module_name: Filter by specific module
    """
    user_id = getattr(request.state, "user_id", None)

    time_series = await llm_usage_repository.get_tool_time_series(
        days=days,
        granularity=granularity,
        user_id=user_id,
        model_name=model_name,
        module_name=module_name,
    )

    return {
        "period_days": days,
        "granularity": granularity,
        "data_points": len(time_series),
        "time_series": time_series,
    }
