# tests/test_json_repair.py
"""
JSON 自动修复单元测试。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.utils.structured_json import parse_json_auto, extract_first_valid_json


class TestParseJsonAuto:
    def test_code_fence(self):
        assert parse_json_auto('```json\n{"a": 1}\n```') == {"a": 1}

    def test_trailing_comma(self):
        assert parse_json_auto('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_single_quotes(self):
        assert parse_json_auto("{'a': 1, 'b': 'x'}") == {"a": 1, "b": "x"}

    def test_noise_around_object(self):
        assert parse_json_auto('回答如下：{"name": "红烧肉", "steps": 3} 请参考') == {
            "name": "红烧肉",
            "steps": 3,
        }

    def test_js_bare_values(self):
        assert parse_json_auto('{"a": 1, "b": undefined}') == {"a": 1, "b": None}

    def test_truncated_json(self):
        assert parse_json_auto('{"key": "value') == {"key": "value"}

    def test_unquoted_keys(self):
        assert parse_json_auto("{name: '红烧肉'}") == {"name": "红烧肉"}

    def test_plain_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_json_auto("完全不是 JSON 的内容")


class TestExtractFirstValidJson:
    def test_backward_compat(self):
        assert extract_first_valid_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_direct(self):
        assert extract_first_valid_json('{"a": 1}') == {"a": 1}
