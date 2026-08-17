from __future__ import annotations


# app/config/evaluation_config.py
"""
Configuration for RAG evaluation using RAGAS framework.

支持两类评测：
1. 在线评估（实时）：faithfulness / answer_relevancy（无需 ground truth）
2. 离线评测（grounding truth）：context_precision / context_recall / answer_correctness，
   基于测试集（question + reference_contexts + reference_answer）
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AlertThresholds:
    """Thresholds for quality alerts. Evaluations below these values trigger alerts."""
    faithfulness: float = 0.3
    answer_relevancy: float = 0.5
    context_precision: float = 0.3
    context_recall: float = 0.3
    answer_correctness: float = 0.3


@dataclass
class EvaluationConfig:
    """
    Configuration for RAG evaluation.

    Attributes:
        enabled: Whether evaluation is enabled
        async_mode: Whether to run evaluation asynchronously (recommended)
        sample_rate: Fraction of requests to evaluate (0.0-1.0)
        metrics: List of metrics to compute
        llm_type: LLM tier to use for evaluation (fast recommended for cost)
        timeout_seconds: Timeout for evaluation
        alert_thresholds: Thresholds for quality alerts
        testsets_dir: Directory containing grounding-truth testsets (.jsonl)
        ground_truth_metrics: Reference-based metrics for offline evaluation
        report_dir: Output directory for offline evaluation reports
        max_question_chars: Max length of a question in testsets
    """
    enabled: bool = True
    async_mode: bool = True
    sample_rate: float = 1.0

    # RAGAS metrics to compute
    # Reference-free: faithfulness, answer_relevancy
    # Reference-based (grounding truth): context_precision, context_recall, answer_correctness
    metrics: List[str] = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevancy",
    ])

    # LLM configuration for evaluation
    llm_type: str = "fast"

    # Timeout for evaluation (seconds)
    timeout_seconds: int = 600

    # Alert thresholds
    alert_thresholds: AlertThresholds = field(default_factory=AlertThresholds)

    # ------------------------------------------------------------------
    # Grounding truth (offline evaluation)
    # ------------------------------------------------------------------
    testsets_dir: str = "testsets"
    # Agent 任务评测集目录（与 RAG 测试集分离，避免 loader 互相误读）
    task_testsets_dir: str = "testsets/tasks"
    ground_truth_metrics: List[str] = field(default_factory=lambda: [
        "context_precision",
        "context_recall",
        "answer_correctness",
    ])
    report_dir: str = "data/evaluation_reports"
    max_question_chars: int = 2000

    def should_evaluate(self) -> bool:
        """Check if evaluation should be performed based on sampling."""
        if not self.enabled:
            return False
        if self.sample_rate >= 1.0:
            return True
        import random
        return random.random() < self.sample_rate


# Default configuration
DefaultEvaluationConfig = EvaluationConfig()
