# app/database/evaluation_repository.py
"""
Repository for RAG evaluation data access.
Handles CRUD operations for RAGEvaluationModel.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import RAGEvaluationModel
from app.database.session import get_session_context

logger = logging.getLogger(__name__)


class EvaluationRepository:
    """
    Data access layer for RAG evaluations.
    Provides methods for creating, updating, and querying evaluation records.
    """

    async def create(
        self,
        message_id: str,
        conversation_id: str,
        query: str,
        context: str,
        response: str,
        rewritten_query: Optional[str] = None,
        user_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        reference_answer: Optional[str] = None,
        reference_contexts: Optional[List[str]] = None,
    ) -> RAGEvaluationModel:
        """
        Create a new evaluation record with pending status.

        Args:
            message_id: ID of the message being evaluated
            conversation_id: ID of the conversation
            query: Original user query
            context: Retrieved context used for generation
            response: Generated response
            rewritten_query: Rewritten query (if any)
            user_id: User ID (if available)
            created_at: Optional creation time override (defaults to now)
            reference_answer: Ground truth answer（离线评测）
            reference_contexts: Ground truth contexts（离线评测）

        Returns:
            Created RAGEvaluationModel instance
        """
        async with get_session_context() as session:
            evaluation = RAGEvaluationModel(
                id=uuid.uuid4(),
                message_id=uuid.UUID(message_id),
                conversation_id=uuid.UUID(conversation_id),
                user_id=user_id,
                query=query,
                rewritten_query=rewritten_query,
                context=context,
                response=response,
                reference_answer=reference_answer,
                reference_contexts=reference_contexts,
                evaluation_status="pending",
                created_at=created_at or datetime.utcnow(),
            )
            session.add(evaluation)
            await session.commit()
            await session.refresh(evaluation)

            logger.info(
                "Created evaluation record: id=%s message_id=%s",
                evaluation.id,
                message_id,
            )
            return evaluation

    async def update_results(
        self,
        evaluation_id: str,
        results: Dict[str, float],
        duration_ms: int,
        status: str = "completed",
        error_message: Optional[str] = None,
        evaluated_at: Optional[datetime] = None,
    ) -> bool:
        """
        Update evaluation record with results.

        Args:
            evaluation_id: ID of the evaluation to update
            results: Dictionary of metric scores
            duration_ms: Evaluation duration in milliseconds
            status: New status (completed/failed)
            error_message: Error message if failed
            evaluated_at: Optional evaluated_at override (defaults to now)

        Returns:
            True if update successful, False otherwise
        """
        async with get_session_context() as session:
            stmt = select(RAGEvaluationModel).where(
                RAGEvaluationModel.id == uuid.UUID(evaluation_id)
            )
            result = await session.execute(stmt)
            evaluation = result.scalar_one_or_none()

            if not evaluation:
                logger.warning("Evaluation not found: %s", evaluation_id)
                return False

            # Update metrics（含 grounding truth 指标）
            evaluation.faithfulness = results.get("faithfulness")
            evaluation.answer_relevancy = results.get("answer_relevancy")
            evaluation.context_precision = results.get("context_precision")
            evaluation.context_recall = results.get("context_recall")
            evaluation.answer_correctness = results.get("answer_correctness")

            # Update metadata
            evaluation.evaluation_status = status
            evaluation.error_message = error_message
            evaluation.evaluation_duration_ms = duration_ms
            evaluation.evaluated_at = evaluated_at or datetime.utcnow()

            await session.commit()

            logger.info(
                "Updated evaluation: id=%s status=%s duration_ms=%d",
                evaluation_id,
                status,
                duration_ms,
            )
            return True

    async def get_by_id(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """Get a single evaluation by ID."""
        async with get_session_context() as session:
            stmt = select(RAGEvaluationModel).where(
                RAGEvaluationModel.id == uuid.UUID(evaluation_id)
            )
            result = await session.execute(stmt)
            evaluation = result.scalar_one_or_none()

            if evaluation:
                return evaluation.to_dict()
            return None

    async def get_by_conversation(
        self, conversation_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all evaluations for a conversation."""
        async with get_session_context() as session:
            stmt = (
                select(RAGEvaluationModel)
                .where(RAGEvaluationModel.conversation_id == uuid.UUID(conversation_id))
                .order_by(RAGEvaluationModel.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            evaluations = result.scalars().all()

            return [e.to_dict() for e in evaluations]

    async def get_statistics(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get aggregated statistics for evaluations.

        Args:
            user_id: Filter by user ID
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            Dictionary with aggregated metrics
        """
        async with get_session_context() as session:
            # Build filter conditions
            conditions = [RAGEvaluationModel.evaluation_status == "completed"]

            if user_id:
                conditions.append(RAGEvaluationModel.user_id == user_id)
            if start_date:
                conditions.append(RAGEvaluationModel.created_at >= start_date)
            if end_date:
                conditions.append(RAGEvaluationModel.created_at <= end_date)

            # Query for aggregated metrics
            stmt = select(
                func.count(RAGEvaluationModel.id).label("total"),
                func.avg(RAGEvaluationModel.faithfulness).label("avg_faithfulness"),
                func.avg(RAGEvaluationModel.answer_relevancy).label("avg_answer_relevancy"),
                func.min(RAGEvaluationModel.faithfulness).label("min_faithfulness"),
                func.max(RAGEvaluationModel.faithfulness).label("max_faithfulness"),
                func.min(RAGEvaluationModel.answer_relevancy).label("min_answer_relevancy"),
                func.max(RAGEvaluationModel.answer_relevancy).label("max_answer_relevancy"),
                func.avg(RAGEvaluationModel.evaluation_duration_ms).label("avg_duration_ms"),
            ).where(and_(*conditions))

            result = await session.execute(stmt)
            row = result.one()

            # Count pending and failed
            pending_stmt = select(func.count(RAGEvaluationModel.id)).where(
                and_(
                    RAGEvaluationModel.evaluation_status == "pending",
                    *conditions[1:],  # Exclude status filter
                )
            )
            failed_stmt = select(func.count(RAGEvaluationModel.id)).where(
                and_(
                    RAGEvaluationModel.evaluation_status == "failed",
                    *conditions[1:],
                )
            )

            pending_result = await session.execute(pending_stmt)
            failed_result = await session.execute(failed_stmt)

            return {
                "total_evaluations": row.total or 0,
                "pending_count": pending_result.scalar() or 0,
                "failed_count": failed_result.scalar() or 0,
                "period": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None,
                },
                "metrics": {
                    "faithfulness": {
                        "mean": float(row.avg_faithfulness) if row.avg_faithfulness else None,
                        "min": float(row.min_faithfulness) if row.min_faithfulness else None,
                        "max": float(row.max_faithfulness) if row.max_faithfulness else None,
                    },
                    "answer_relevancy": {
                        "mean": float(row.avg_answer_relevancy) if row.avg_answer_relevancy else None,
                        "min": float(row.min_answer_relevancy) if row.min_answer_relevancy else None,
                        "max": float(row.max_answer_relevancy) if row.max_answer_relevancy else None,
                    },
                },
                "avg_evaluation_duration_ms": float(row.avg_duration_ms) if row.avg_duration_ms else None,
            }

    async def get_trends(
        self,
        days: int = 7,
        granularity: str = "day",
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get evaluation trends over time.

        Args:
            days: Number of days to look back
            granularity: "day" or "hour"
            user_id: Filter by user ID

        Returns:
            List of trend data points
        """
        async with get_session_context() as session:
            start_date = datetime.utcnow() - timedelta(days=days)

            conditions = [
                RAGEvaluationModel.evaluation_status == "completed",
                RAGEvaluationModel.created_at >= start_date,
            ]
            if user_id:
                conditions.append(RAGEvaluationModel.user_id == user_id)

            # Group by date
            if granularity == "hour":
                date_trunc = func.date_trunc("hour", RAGEvaluationModel.created_at)
            else:
                date_trunc = func.date_trunc("day", RAGEvaluationModel.created_at)

            stmt = (
                select(
                    date_trunc.label("period"),
                    func.count(RAGEvaluationModel.id).label("count"),
                    func.avg(RAGEvaluationModel.faithfulness).label("avg_faithfulness"),
                    func.avg(RAGEvaluationModel.answer_relevancy).label("avg_answer_relevancy"),
                )
                .where(and_(*conditions))
                .group_by(date_trunc)
                .order_by(date_trunc)
            )

            result = await session.execute(stmt)
            rows = result.all()

            return [
                {
                    "period": row.period.isoformat() if row.period else None,
                    "count": row.count,
                    "metrics": {
                        "faithfulness": float(row.avg_faithfulness) if row.avg_faithfulness else None,
                        "answer_relevancy": float(row.avg_answer_relevancy) if row.avg_answer_relevancy else None,
                    },
                }
                for row in rows
            ]

    async def get_alerts(
        self,
        thresholds: Dict[str, float],
        limit: int = 50,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get evaluations that fall below quality thresholds.

        Args:
            thresholds: Dictionary of metric -> threshold value
            limit: Maximum number of alerts to return
            user_id: Filter by user ID

        Returns:
            List of evaluation records that triggered alerts
        """
        async with get_session_context() as session:
            conditions = [RAGEvaluationModel.evaluation_status == "completed"]

            if user_id:
                conditions.append(RAGEvaluationModel.user_id == user_id)

            # Build OR conditions for threshold violations
            threshold_conditions = []
            if "faithfulness" in thresholds:
                threshold_conditions.append(
                    RAGEvaluationModel.faithfulness < thresholds["faithfulness"]
                )
            if "answer_relevancy" in thresholds:
                threshold_conditions.append(
                    RAGEvaluationModel.answer_relevancy < thresholds["answer_relevancy"]
                )

            if not threshold_conditions:
                return []

            from sqlalchemy import or_
            conditions.append(or_(*threshold_conditions))

            stmt = (
                select(RAGEvaluationModel)
                .where(and_(*conditions))
                .order_by(RAGEvaluationModel.created_at.desc())
                .limit(limit)
            )

            result = await session.execute(stmt)
            evaluations = result.scalars().all()

            # Add alert info to each evaluation
            alerts = []
            for e in evaluations:
                alert_data = e.to_dict()
                alert_data["violated_thresholds"] = []

                if e.faithfulness is not None and "faithfulness" in thresholds:
                    if e.faithfulness < thresholds["faithfulness"]:
                        alert_data["violated_thresholds"].append("faithfulness")

                if e.answer_relevancy is not None and "answer_relevancy" in thresholds:
                    if e.answer_relevancy < thresholds["answer_relevancy"]:
                        alert_data["violated_thresholds"].append("answer_relevancy")

                alerts.append(alert_data)

            return alerts


# Singleton instance
evaluation_repository = EvaluationRepository()
