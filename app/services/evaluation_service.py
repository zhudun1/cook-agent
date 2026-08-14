# app/services/evaluation_service.py
"""
RAG Evaluation Service using RAGAS framework.
Provides asynchronous evaluation of RAG responses for quality monitoring.
"""

import asyncio
import logging
import math
import time
from typing import Any, Dict, List, Optional

from app.config import settings
from app.config.evaluation_config import EvaluationConfig
from app.database.evaluation_repository import evaluation_repository
from app.llm import LLMProvider, get_usage_callbacks, llm_context

logger = logging.getLogger(__name__)


class FilteredChatOpenAI:
    """
    A wrapper around ChatOpenAI that filters out unsupported parameters.
    Some API providers (e.g., ModelScope) don't support the 'n' parameter
    that RAGAS uses internally.
    """

    # Parameters that some API providers don't support
    UNSUPPORTED_PARAMS = {"n"}

    def __init__(self, base_llm, callbacks=None):
        self._base_llm = base_llm
        self._callbacks = callbacks or []
        # Copy common attributes from base LLM for compatibility
        for attr in ["model_name", "temperature", "max_tokens", "model"]:
            if hasattr(base_llm, attr):
                setattr(self, attr, getattr(base_llm, attr))

    def _filter_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove unsupported parameters from kwargs."""
        filtered = {k: v for k, v in kwargs.items() if k not in self.UNSUPPORTED_PARAMS}
        # Inject callbacks if not already present
        if self._callbacks and "callbacks" not in filtered:
            filtered["callbacks"] = self._callbacks
        return filtered

    def invoke(self, *args, **kwargs):
        return self._base_llm.invoke(*args, **self._filter_kwargs(kwargs))

    async def ainvoke(self, *args, **kwargs):
        return await self._base_llm.ainvoke(*args, **self._filter_kwargs(kwargs))

    def generate(self, *args, **kwargs):
        return self._base_llm.generate(*args, **self._filter_kwargs(kwargs))

    async def agenerate(self, *args, **kwargs):
        return await self._base_llm.agenerate(*args, **self._filter_kwargs(kwargs))

    def generate_prompt(self, *args, **kwargs):
        return self._base_llm.generate_prompt(*args, **self._filter_kwargs(kwargs))

    async def agenerate_prompt(self, *args, **kwargs):
        return await self._base_llm.agenerate_prompt(
            *args, **self._filter_kwargs(kwargs)
        )

    def bind(self, **kwargs):
        """Return a new FilteredChatOpenAI with bound parameters."""
        bound_llm = self._base_llm.bind(**self._filter_kwargs(kwargs))
        return FilteredChatOpenAI(bound_llm, callbacks=self._callbacks)

    def __getattr__(self, name):
        """Forward attribute access to the base LLM."""
        return getattr(self._base_llm, name)


class EvaluationService:
    """
    RAG evaluation service based on RAGAS framework.
    Supports asynchronous evaluation to avoid blocking the main conversation flow.
    """

    MODULE_NAME = "quality_evaluation"

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize the evaluation service.

        Args:
            config: Evaluation configuration. Uses global config if not provided.
        """
        self.config = config or settings.evaluation
        self._ragas_initialized = False
        self._metrics = None
        self._llm = None
        self._embeddings = None
        self._callbacks = get_usage_callbacks()

    def _init_ragas_sync(self):
        """
        Synchronous RAGAS initialization (runs in thread pool).
        This method may block due to HuggingFace model downloads.
        """
        try:
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            )
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            # Get LLM configuration
            provider = LLMProvider(settings.llm)
            base_llm = provider.create_llm(self.config.llm_type, temperature=0.0)

            # Wrap with FilteredChatOpenAI to filter out unsupported params (e.g., 'n')
            # Pass callbacks for usage tracking
            filtered_llm = FilteredChatOpenAI(base_llm, callbacks=self._callbacks)

            # Wrap for RAGAS
            self._llm = LangchainLLMWrapper(filtered_llm)

            # Get embeddings configuration from RAG config
            from app.rag.embeddings.embedding_factory import get_embedding_model

            # Use the embedding factory to create appropriate embedding model
            # This handles both local models (HuggingFace) and API-based models
            base_embeddings = get_embedding_model(settings.rag)

            self._embeddings = LangchainEmbeddingsWrapper(base_embeddings)

            # Build metrics map
            # Reference-free: faithfulness, answer_relevancy
            # Reference-based (grounding truth): context_precision, context_recall,
            # answer_correctness
            self._metrics_map = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "answer_correctness": answer_correctness,
            }

            # 全部可配置指标（reference 指标仅在提供 reference 数据时启用）
            self._all_configured = list(
                dict.fromkeys(self.config.metrics + self.config.ground_truth_metrics)
            )

            # Configure metrics with LLM and embeddings
            for metric in self._metrics_map.values():
                if hasattr(metric, "llm"):
                    metric.llm = self._llm
                if hasattr(metric, "embeddings"):
                    metric.embeddings = self._embeddings

            self._ragas_initialized = True
            logger.info(
                "RAGAS initialized with metrics: %s",
                self._all_configured,
            )

        except ImportError as e:
            logger.error("Failed to import RAGAS: %s", e)
            raise
        except Exception as e:
            logger.error("Failed to initialize RAGAS: %s", e)
            raise

    async def _init_ragas(self):
        """
        Lazily initialize RAGAS components asynchronously.
        Runs blocking initialization in thread pool to avoid blocking the event loop.
        """
        if self._ragas_initialized:
            return

        await asyncio.to_thread(self._init_ragas_sync)

    async def evaluate(
        self,
        query: str,
        context: str,
        response: str,
        reference_answer: Optional[str] = None,
        reference_contexts: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG response using RAGAS metrics.

        Args:
            query: Original user query
            context: Retrieved context used for generation
            response: Generated response
            reference_answer: Ground truth answer（提供时启用 answer_correctness）
            reference_contexts: Ground truth contexts（提供时启用
                context_precision / context_recall）

        Returns:
            Dictionary of metric scores
        """
        await self._init_ragas()

        try:
            from datasets import Dataset
            from ragas import evaluate

            # Prepare dataset for RAGAS
            # RAGAS expects contexts as a list of strings
            contexts = [context] if context else [""]

            data = {
                "question": [query],
                "answer": [response],
                "contexts": [contexts],
            }
            has_reference = bool(reference_answer) or bool(reference_contexts)
            if reference_answer:
                data["reference"] = [[reference_answer]]
            if reference_contexts:
                data["reference_contexts"] = [list(reference_contexts)]

            dataset = Dataset.from_dict(data)

            # Reference 指标仅在提供 grounding truth 时启用
            if has_reference:
                self._metrics = [
                    self._metrics_map[m]
                    for m in self._all_configured
                    if m in self._metrics_map
                ]
            else:
                self._metrics = [
                    self._metrics_map[m]
                    for m in self.config.metrics
                    if m in self._metrics_map
                ]

            # Run evaluation
            result = await asyncio.to_thread(
                evaluate,
                dataset,
                metrics=self._metrics,
            )

            # Extract scores from RAGAS result
            # In RAGAS 0.4.x, result is an EvaluationResult object
            # We need to access the scores differently
            scores = {}

            # Try to get scores from the result object
            if hasattr(result, "scores"):
                # RAGAS 0.4.x returns scores as a list of dicts
                score_list = result.scores  # type: ignore
                if score_list and len(score_list) > 0:
                    first_score = score_list[0]
                    for metric in self._metrics:
                        metric_name = metric.name
                        if metric_name in first_score:
                            value = first_score[metric_name]
                            scores[metric_name] = (
                                float(value)
                                if value is not None and not math.isnan(value)
                                else None
                            )
            elif hasattr(result, "to_pandas"):
                # Alternative: use pandas DataFrame
                df = result.to_pandas()  # type: ignore
                for metric in self._metrics:
                    metric_name = metric.name
                    if metric_name in df.columns:
                        value = df[metric_name].iloc[0]
                        scores[metric_name] = (
                            float(value)
                            if value is not None and not math.isnan(value)
                            else None
                        )
            else:
                # Fallback: try direct dict-like access
                for metric in self._metrics:
                    metric_name = metric.name
                    try:
                        # Try to access as attribute
                        if hasattr(result, metric_name):
                            value = getattr(result, metric_name)
                            if hasattr(value, "__iter__") and not isinstance(
                                value, str
                            ):
                                scores[metric_name] = (
                                    float(list(value)[0]) if value else None
                                )
                            else:
                                scores[metric_name] = (
                                    float(value)
                                    if value is not None and not math.isnan(value)  # type: ignore
                                    else None
                                )  # type: ignore
                    except Exception:
                        pass

            logger.info("Evaluation completed: %s", scores)
            return scores

        except Exception as e:
            logger.error("Evaluation failed: %s", e, exc_info=True)
            raise

    async def schedule_evaluation(
        self,
        message_id: str,
        conversation_id: str,
        query: str,
        context: str,
        response: str,
        rewritten_query: Optional[str] = None,
        user_id: Optional[str] = None,
        reference_answer: Optional[str] = None,
        reference_contexts: Optional[List[str]] = None,
    ):
        """
        Schedule an asynchronous evaluation for a RAG response.

        This method creates an evaluation record and runs the evaluation
        in the background without blocking the main conversation flow.

        Args:
            message_id: ID of the message being evaluated
            conversation_id: ID of the conversation
            query: Original user query
            context: Retrieved context used for generation
            response: Generated response
            rewritten_query: Rewritten query (if any)
            user_id: User ID (if available)
        """
        if not self.config.enabled:
            logger.debug("Evaluation disabled, skipping")
            return

        # Check sampling rate
        if not self.config.should_evaluate():
            logger.debug("Evaluation skipped due to sampling")
            return

        # Skip if no context (non-RAG response)
        if not context:
            logger.debug("No context provided, skipping evaluation")
            return

        # Skip if no response (empty response)
        if not response or not response.strip():
            logger.debug("No response provided, skipping evaluation")
            return

        try:
            # Create evaluation record
            evaluation = await evaluation_repository.create(
                message_id=message_id,
                conversation_id=conversation_id,
                query=query,
                context=context,
                response=response,
                rewritten_query=rewritten_query,
                user_id=user_id,
            )

            # Run evaluation asynchronously
            if self.config.async_mode:
                asyncio.create_task(
                    self._run_evaluation(
                        str(evaluation.id),
                        query,
                        context,
                        response,
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                )
            else:
                await self._run_evaluation(
                    str(evaluation.id),
                    query,
                    context,
                    response,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )

        except Exception as e:
            logger.error("Failed to schedule evaluation: %s", e, exc_info=True)

    async def _run_evaluation(
        self,
        evaluation_id: str,
        query: str,
        context: str,
        response: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ):
        """
        Run the actual evaluation and update the database.

        Args:
            evaluation_id: ID of the evaluation record
            query: Original user query
            context: Retrieved context
            response: Generated response
            user_id: User ID for tracking (optional)
            conversation_id: Conversation ID for tracking (optional)
        """
        start_time = time.time()

        try:
            # Run evaluation with timeout
            # Use llm_context for usage tracking
            with llm_context(self.MODULE_NAME, user_id, conversation_id):
                results = await asyncio.wait_for(
                    self.evaluate(query, context, response),
                    timeout=self.config.timeout_seconds,
                )

            duration_ms = int((time.time() - start_time) * 1000)

            # Update database with results
            await evaluation_repository.update_results(
                evaluation_id=evaluation_id,
                results=results,
                duration_ms=duration_ms,
                status="completed",
            )

            logger.info(
                "Evaluation %s completed in %dms: %s",
                evaluation_id,
                duration_ms,
                results,
            )

        except asyncio.TimeoutError:
            duration_ms = int((time.time() - start_time) * 1000)
            await evaluation_repository.update_results(
                evaluation_id=evaluation_id,
                results={},
                duration_ms=duration_ms,
                status="failed",
                error_message=f"Evaluation timed out after {self.config.timeout_seconds}s",
            )
            logger.warning("Evaluation %s timed out", evaluation_id)

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            await evaluation_repository.update_results(
                evaluation_id=evaluation_id,
                results={},
                duration_ms=duration_ms,
                status="failed",
                error_message=str(e)[:500],
            )
            logger.error("Evaluation %s failed: %s", evaluation_id, e, exc_info=True)


# Singleton instance
evaluation_service = EvaluationService()
