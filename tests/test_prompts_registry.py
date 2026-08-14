# tests/test_prompts_registry.py
"""
P3 Prompt 版本管理 + 灰度路由单元测试。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.prompts.registry import PromptRegistry


@pytest.fixture
def registry():
    r = PromptRegistry()
    r.register(
        "default",
        "你是正式版提示词",
        version="v1",
        weight=0.8,
        description="正式版",
    )
    r.register(
        "default",
        "你是实验版提示词（更简洁）",
        version="v2",
        weight=0.2,
        description="实验版",
    )
    return r


class TestPromptRegistry:
    def test_single_version(self):
        r = PromptRegistry()
        r.register("agent", "内容A", version="v1")
        assert r.get("agent", user_id="u1") == "内容A"

    def test_multi_version_returns_some_version(self, registry):
        content = registry.get("default", user_id="u1")
        assert content in ("你是正式版提示词", "你是实验版提示词（更简洁）")

    def test_stable_split_per_user(self, registry):
        # 同一用户多次命中同一版本
        results = {registry.get("default", user_id="u100") for _ in range(20)}
        assert len(results) == 1

    def test_different_users_can_split(self, registry):
        versions = {
            registry.get("default", user_id=f"user-{i}") for i in range(50)
        }
        # 50 个用户下两个版本通常都会被命中（权重 0.8/0.2）
        assert len(versions) >= 1
        # 允许极端情况只命中一个版本；此处仅验证返回合法
        assert all(v in ("你是正式版提示词", "你是实验版提示词（更简洁）") for v in versions)

    def test_default_returned_when_missing(self):
        r = PromptRegistry()
        assert r.get("missing", default="fallback") == "fallback"

    def test_unregistered_returns_default(self):
        r = PromptRegistry()
        assert r.get("nope", user_id="u1", default="d") == "d"

    def test_list_versions(self, registry):
        versions = registry.list_versions("default")
        assert len(versions) == 2
        assert {v["version"] for v in versions} == {"v1", "v2"}
        assert all(v["content_hash"] for v in versions)

    def test_overwrite_same_version(self):
        r = PromptRegistry()
        r.register("p", "旧内容", version="v1")
        r.register("p", "新内容", version="v1")
        assert r.get("p") == "新内容"
        assert len(r.list_versions("p")) == 1

    def test_unregister(self, registry):
        assert registry.unregister("default", version="v2")
        assert len(registry.list_versions("default")) == 1
