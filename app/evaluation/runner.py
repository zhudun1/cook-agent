from __future__ import annotations


# app/evaluation/runner.py
"""
Grounding truth 离线批量评测执行器。

流程（对每条测试用例）:
1. 检索：rag_service.retrieve(question) 获取系统检索到的上下文
2. 生成：基于检索上下文生成答案（可配置生成 prompt）
3. 评测：RAGAS reference 指标（context_precision / context_recall /
   answer_correctness）+ 可选 reference-free 指标（faithfulness / answer_relevancy）
4. 聚合：逐条得分 + 汇总统计 + 失败用例识别

评测结果同时落库（rag_evaluations，message_id 使用占位 UUID）与生成报告。
"""


import asyncio
import logging
import math
import time
import uuid
from typing import Any, Dict, List, Optional

from app.config import settings
from app.config.evaluation_config import EvaluationConfig
from app.evaluation.dataset import GroundTruthDataset, GroundTruthCase

logger = logging.getLogger(__name__)

# 离线评测生成答案的系统提示（聚焦事实性回答，便于 answer_correctness 评分）
RAG_GENERATION_SYSTEM_PROMPT = (
    "你是一个饮食助手。请仅根据提供的参考资料回答用户问题，"
    "如果资料不足请明确说明。回答要简洁、准确、完整。"
)

# 占位 message_id（离线评测无真实对话消息）
PLACEHOLDER_MESSAGE_ID = "00000000-0000-0000-0000-000000000001"


class OfflineEvaluationRunner:
    """
    Grounding truth 离线评测执行器。

    用法::

        runner = OfflineEvaluationRunner()
        result = await runner.run_dataset(dataset)
        result = await runner.run_all()   # 评测 testsets 目录下全部测试集
    """

    MODULE_NAME = "offline_evaluation"

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        rag_service: Any = None,
    ):
        self.config = config or settings.evaluation
        if rag_service is None:
            from app.services.rag_service import rag_service_instance

            rag_service = rag_service_instance
        self.rag_service = rag_service

        self._initialized = False
        self._metrics = None
        self._metrics_map = {}
        self._llm = None
        self._embeddings = None
        from app.llm.callbacks import get_usage_callbacks

        self._callbacks = get_usage_callbacks()

    # ------------------------------------------------------------------
    # RAGAS 初始化
    # ------------------------------------------------------------------
    def _init_ragas_sync(self) -> None:
        """同步初始化 RAGAS（在线程池中执行，避免阻塞事件循环）。"""
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

            from app.llm.provider import LLMProvider

            provider = LLMProvider(settings.llm)
            base_llm = provider.create_llm(self.config.llm_type, temperature=0.0)

            # 过滤不支持的参数（如 ModelScope 不支持 'n'）
            from app.services.evaluation_service import FilteredChatOpenAI

            filtered_llm = FilteredChatOpenAI(base_llm, callbacks=self._callbacks)
            self._llm = LangchainLLMWrapper(filtered_llm)

            from app.rag.embeddings.embedding_factory import get_embedding_model

            base_embeddings = get_embedding_model(settings.rag)
            self._embeddings = LangchainEmbeddingsWrapper(base_embeddings)

            # reference 指标（需要 grounding truth）
            reference_metrics = {
                "context_precision": context_precision,
                "context_recall": context_recall,
                "answer_correctness": answer_correctness,
            }
            # reference-free 指标
            free_metrics = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
            }
            self._metrics_map = {**reference_metrics, **free_metrics}

            # 目标指标 = 配置的在线指标 + grounding truth 指标
            target = list(dict.fromkeys(self.config.metrics + self.config.ground_truth_metrics))
            self._metrics = [
                self._metrics_map[m]
                for m in target
                if m in self._metrics_map
            ]

            for metric in self._metrics:
                if hasattr(metric, "llm"):
                    metric.llm = self._llm
                if hasattr(metric, "embeddings"):
                    metric.embeddings = self._embeddings

            self._initialized = True
            logger.info(
                "Offline evaluation initialized with metrics: %s",
                [m.name for m in self._metrics],
            )
        except Exception as e:
            logger.error("Failed to init offline RAGAS: %s", e, exc_info=True)
            raise

    async def _init_ragas(self) -> None:
        if self._initialized:
            return
        await asyncio.to_thread(self._init_ragas_sync)

    # ------------------------------------------------------------------
    # 单条用例流水线
    # ------------------------------------------------------------------
    async def _retrieve_context(self, question: str) -> List[str]:
        """检索系统上下文（文档列表）。"""
        try:
            result = await self.rag_service.retrieve(
                question,
                user_id=None,
                conversation_id=None,
            )
            return [doc.page_content for doc in result.documents]
        except Exception as e:
            logger.warning("Retrieval failed for question: %s -> %s", question[:60], e)
            return []

    async def _generate_answer(self, question: str, contexts: List[str]) -> str:
        """基于检索上下文生成答案。"""
        if not contexts:
            return ""
        from app.llm.provider import LLMProvider

        provider = LLMProvider(settings.llm)
        invoker = provider.create_invoker(llm_type=self.config.llm_type, temperature=0.0)

        context_text = "\n\n".join(contexts)
        from app.llm.context import llm_context

        with llm_context(self.MODULE_NAME):
            response = await invoker.ainvoke(
                [
                    {"role": "system", "content": RAG_GENERATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"【参考资料】\n{context_text}\n\n【问题】\n{question}",
                    },
                ]
            )
        content = getattr(response, "content", "") or ""
        return content.strip()

    async def _evaluate_case(
        self,
        case: GroundTruthCase,
        contexts: List[str],
        answer: str,
    ) -> Dict[str, Any]:
        """对单条用例执行 RAGAS 评测。"""
        await self._init_ragas()

        from datasets import Dataset
        from ragas import evaluate

        dataset = Dataset.from_dict(
            {
                "question": [case.question],
                "answer": [answer],
                "contexts": [contexts],
                # grounding truth
                "reference": [[case.reference_answer]],
                "reference_contexts": [case.reference_contexts],
            }
        )

        result = await asyncio.to_thread(evaluate, dataset, metrics=self._metrics)

        scores: Dict[str, float] = {}
        if hasattr(result, "scores") and result.scores:
            first = result.scores[0]
            for metric in self._metrics:
                name = metric.name
                value = first.get(name)
                scores[name] = (
                    float(value) if value is not None and not math.isnan(value) else None
                )
        elif hasattr(result, "to_pandas"):
            df = result.to_pandas()
            for metric in self._metrics:
                name = metric.name
                if name in df.columns:
                    value = df[name].iloc[0]
                    scores[name] = (
                        float(value) if value is not None and not math.isnan(value) else None
                    )
        return scores

    # ------------------------------------------------------------------
    # 数据集级评测
    # ------------------------------------------------------------------
    async def run_dataset(
        self,
        dataset: GroundTruthDataset,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """
        评测一个测试集。

        Args:
            dataset: 测试集
            persist: 是否将结果写入 rag_evaluations 表

        Returns:
            评测结果 dict（逐条 + 汇总 + 失败用例）
        """
        logger.info("Running offline evaluation on %s ...", dataset.summary())
        started = time.time()
        rows: List[Dict[str, Any]] = []

        for case in dataset.cases:
            row = await self._run_single(case, dataset.name, persist=persist)
            rows.append(row)

        aggregate = self._aggregate(rows)
        duration_ms = int((time.time() - started) * 1000)
        aggregate["dataset"] = dataset.name
        aggregate["duration_ms"] = duration_ms
        aggregate["cases"] = rows
        aggregate["failures"] = self._find_failures(rows)

        logger.info(
            "Offline evaluation done: %s, %d cases in %dms",
            dataset.name,
            len(rows),
            duration_ms,
        )
        return aggregate

    async def run_single(
        self,
        case: GroundTruthCase,
        dataset_name: str = "single",
        persist: bool = True,
    ) -> Dict[str, Any]:
        """评测单条用例（调试用）。"""
        return await self._run_single(case, dataset_name, persist=persist)

    async def _run_single(
        self,
        case: GroundTruthCase,
        dataset_name: str,
        persist: bool,
    ) -> Dict[str, Any]:
        case_started = time.time()
        contexts = await self._retrieve_context(case.question)
        answer = await self._generate_answer(case.question, contexts)

        scores: Dict[str, Any] = {}
        error = None
        if not contexts:
            error = "No context retrieved (retrieval returned empty)"
        elif not answer:
            error = "No answer generated"
        else:
            try:
                scores = await self._evaluate_case(case, contexts, answer)
            except Exception as e:
                error = str(e)[:300]
                logger.error("Evaluation failed for %s: %s", case.question[:60], e)

        row = {
            "question": case.question,
            "retrieved_contexts": contexts,
            "reference_contexts": case.reference_contexts,
            "reference_answer": case.reference_answer,
            "generated_answer": answer,
            "metrics": scores,
            "error": error,
            "duration_ms": int((time.time() - case_started) * 1000),
            "metadata": case.metadata,
        }

        if persist and not error:
            await self._persist(case, contexts, answer, scores, dataset_name)

        return row

    async def _persist(
        self,
        case: GroundTruthCase,
        contexts: List[str],
        answer: str,
        scores: Dict[str, Any],
        dataset_name: str,
    ) -> None:
        """评测结果落库（rag_evaluations）。"""
        try:
            from app.database.evaluation_repository import evaluation_repository

            conv_id = uuid.uuid4()
            evaluation = await evaluation_repository.create(
                message_id=PLACEHOLDER_MESSAGE_ID,
                conversation_id=str(conv_id),
                query=case.question,
                context="\n\n".join(contexts),
                response=answer,
                user_id=f"offline:{dataset_name}",
                reference_answer=case.reference_answer,
                reference_contexts=case.reference_contexts,
            )
            await evaluation_repository.update_results(
                evaluation_id=str(evaluation.id),
                results=scores,
                duration_ms=0,
                status="completed",
            )
        except Exception as e:
            logger.warning("Failed to persist evaluation result: %s", e)

    # ------------------------------------------------------------------
    # 聚合与失败识别
    # ------------------------------------------------------------------
    def _aggregate(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总指标均值（忽略 None/缺失）。"""
        metric_names = {
            "context_precision",
            "context_recall",
            "answer_correctness",
            "faithfulness",
            "answer_relevancy",
        }
        summary: Dict[str, Any] = {"metrics": {}}
        for name in metric_names:
            values = [
                r["metrics"][name]
                for r in rows
                if r.get("metrics") and r["metrics"].get(name) is not None
            ]
            if values:
                summary["metrics"][name] = {
                    "mean": round(sum(values) / len(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "count": len(values),
                }
        summary["total_cases"] = len(rows)
        summary["evaluated_cases"] = sum(1 for r in rows if r.get("metrics"))
        summary["failed_cases"] = sum(1 for r in rows if r.get("error"))
        return summary

    def _find_failures(
        self,
        rows: List[Dict[str, Any]],
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """识别低于阈值的失败用例（便于回归分析）。"""
        failures = []
        for r in rows:
            reasons = []
            metrics = r.get("metrics") or {}
            for name, val in metrics.items():
                if val is not None and val < threshold:
                    reasons.append(f"{name}={val:.2f}")
            if r.get("error"):
                reasons.append(f"error={r['error']}")
            if reasons:
                failures.append(
                    {
                        "question": r["question"],
                        "reasons": reasons,
                        "metrics": metrics,
                    }
                )
        return failures

    # ------------------------------------------------------------------
    async def run_all(
        self,
        persist: bool = True,
        report: bool = True,
    ) -> Dict[str, Any]:
        """
        评测 testsets 目录下全部测试集，并生成报告。

        Args:
            persist: 是否落库
            report: 是否生成报告文件

        Returns:
            {dataset_name: 结果} 映射
        """
        from app.evaluation.dataset import GroundTruthDataset

        datasets = GroundTruthDataset.load_dir(
            self.config.testsets_dir,
            max_question_chars=self.config.max_question_chars,
        )
        if not datasets:
            logger.warning("No testsets found in %s", self.config.testsets_dir)
            return {}

        all_results: Dict[str, Any] = {}
        for ds in datasets:
            all_results[ds.name] = await self.run_dataset(ds, persist=persist)

        if report:
            from app.evaluation.report import generate_report

            report_path = await asyncio.to_thread(
                generate_report, all_results, self.config.report_dir
            )
            all_results["_report_path"] = report_path

        return all_results


# 单例
# 惰性单例：避免模块导入时实例化（拉入 rag_service -> openai client 依赖链，
# 在无 API key 的 CI/环境会直接崩溃）
_offline_runner_instance: Optional[OfflineEvaluationRunner] = None


def get_offline_evaluation_runner() -> "OfflineEvaluationRunner":
    """获取 OfflineEvaluationRunner 单例（惰性创建）。"""
    global _offline_runner_instance
    if _offline_runner_instance is None:
        _offline_runner_instance = OfflineEvaluationRunner()
    return _offline_runner_instance
