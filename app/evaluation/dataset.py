from __future__ import annotations


# app/evaluation/dataset.py
"""
Grounding Truth 测试集加载与校验。

JSONL 行格式（testsets/*.jsonl）::

    {
      "question": "如何制作西红柿炒鸡蛋？",
      "reference_contexts": ["西红柿炒鸡蛋的做法：...", "鸡蛋的营养..."],
      "reference_answer": "西红柿炒鸡蛋的做法是...（标准答案）"
    }

字段说明:
- question: 必填，评测问题
- reference_contexts: 必填，标准答案依赖的上下文片段（用于 context_precision/recall）
- reference_answer: 必填，标准答案（用于 answer_correctness）
- metadata: 可选，附加信息（如来源菜谱、难度）
"""


import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DatasetValidationError(ValueError):
    """测试集校验失败。"""


@dataclass
class GroundTruthCase:
    """一条带 grounding truth 的评测用例。"""

    question: str
    reference_contexts: List[str]
    reference_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "reference_contexts": self.reference_contexts,
            "reference_answer": self.reference_answer,
            "metadata": self.metadata,
        }


class GroundTruthDataset:
    """
    Grounding truth 测试集。

    用法::

        dataset = GroundTruthDataset.load("testsets/cooking.jsonl")
        print(dataset.cases)  # List[GroundTruthCase]
    """

    def __init__(self, name: str, cases: List[GroundTruthCase]):
        self.name = name
        self.cases = cases

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, max_question_chars: int = 2000) -> "GroundTruthDataset":
        """
        从 JSONL 文件加载测试集。

        Args:
            path: JSONL 文件路径
            max_question_chars: 单条 question 最大长度（超出报错）

        Returns:
            GroundTruthDataset

        Raises:
            DatasetValidationError: 文件缺失或行格式非法
        """
        p = Path(path)
        if not p.exists():
            raise DatasetValidationError(f"Testset file not found: {p}")

        cases: List[GroundTruthCase] = []
        with open(p, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise DatasetValidationError(
                        f"{p}:{lineno} invalid JSON: {e}"
                    ) from e

                case = cls._parse_case(obj, lineno, max_question_chars)
                cases.append(case)

        if not cases:
            raise DatasetValidationError(f"Testset is empty: {p}")

        logger.info("Loaded testset %s: %d cases", p, len(cases))
        return cls(name=p.stem, cases=cases)

    @classmethod
    def load_dir(
        cls,
        directory: str | Path,
        max_question_chars: int = 2000,
    ) -> List["GroundTruthDataset"]:
        """
        加载目录下所有 *.jsonl 测试集。

        Args:
            directory: 测试集目录
            max_question_chars: 单条 question 最大长度

        Returns:
            测试集列表（按文件名排序）
        """
        d = Path(directory)
        if not d.exists():
            logger.warning("Testsets dir not found: %s", d)
            return []
        datasets = []
        for p in sorted(d.glob("*.jsonl")):
            try:
                datasets.append(cls.load(p, max_question_chars))
            except DatasetValidationError as e:
                logger.error("Skipping testset %s: %s", p, e)
        return datasets

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_case(obj: Dict[str, Any], lineno: int, max_question_chars: int) -> GroundTruthCase:
        question = obj.get("question")
        ref_contexts = obj.get("reference_contexts")
        ref_answer = obj.get("reference_answer")

        if not question or not isinstance(question, str):
            raise DatasetValidationError(f"line {lineno}: 'question' required (string)")
        if len(question) > max_question_chars:
            raise DatasetValidationError(
                f"line {lineno}: question too long ({len(question)} > {max_question_chars})"
            )
        if not ref_contexts or not isinstance(ref_contexts, list):
            raise DatasetValidationError(
                f"line {lineno}: 'reference_contexts' required (non-empty list)"
            )
        if not ref_answer or not isinstance(ref_answer, str):
            raise DatasetValidationError(
                f"line {lineno}: 'reference_answer' required (string)"
            )

        return GroundTruthCase(
            question=question,
            reference_contexts=[c for c in ref_contexts if isinstance(c, str) and c],
            reference_answer=ref_answer,
            metadata=obj.get("metadata", {}) or {},
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def summary(self) -> str:
        return f"{self.name}: {len(self.cases)} cases"
