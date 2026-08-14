# tests/test_evaluation_dataset.py
"""
Grounding truth 测试集加载单元测试。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import pytest

from app.evaluation.dataset import (
    DatasetValidationError,
    GroundTruthDataset,
    GroundTruthCase,
)


@pytest.fixture
def sample_file(tmp_path):
    rows = [
        {
            "question": "西红柿炒鸡蛋怎么做？",
            "reference_contexts": ["鸡蛋打散炒熟", "西红柿切块炒出汁"],
            "reference_answer": "先炒蛋后炒西红柿再混合。",
            "metadata": {"source": "HowToCook"},
        },
        {
            "question": "红烧肉步骤？",
            "reference_contexts": ["焯水", "炒糖色", "慢炖"],
            "reference_answer": "焯水、炒糖色、慢炖。",
        },
    ]
    p = tmp_path / "cooking.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


class TestGroundTruthDataset:
    def test_load(self, sample_file):
        ds = GroundTruthDataset.load(sample_file)
        assert len(ds) == 2
        first: GroundTruthCase = ds.cases[0]
        assert first.question == "西红柿炒鸡蛋怎么做？"
        assert len(first.reference_contexts) == 2
        assert first.metadata["source"] == "HowToCook"

    def test_load_dir(self, tmp_path, sample_file):
        datasets = GroundTruthDataset.load_dir(str(tmp_path))
        assert len(datasets) == 1
        assert datasets[0].name == "cooking"

    def test_missing_file(self):
        with pytest.raises(DatasetValidationError):
            GroundTruthDataset.load("no_such_file.jsonl")

    def test_invalid_row(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.write('{"question": "没有上下文的题"}\n')
        with pytest.raises(DatasetValidationError):
            GroundTruthDataset.load(p)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        with pytest.raises(DatasetValidationError):
            GroundTruthDataset.load(p)

    def test_question_too_long(self, tmp_path):
        p = tmp_path / "long.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "question": "x" * 5000,
                        "reference_contexts": ["c"],
                        "reference_answer": "a",
                    }
                )
                + "\n"
            )
        with pytest.raises(DatasetValidationError):
            GroundTruthDataset.load(p, max_question_chars=100)
