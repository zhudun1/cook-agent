# app/memory/extractor.py
"""
记忆提取器（Memory Extractor）

从对话中提取用户长期记忆（偏好/目标/限制/事实）：
1. LLM 提取（推荐）：给定对话片段，输出 JSON [{content, type, importance}]
2. 启发式回退（无 LLM key / 提取失败）：命中关键词的句子直接入库
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.config.memory_config import MemoryConfig

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM_PROMPT = """你是 CookHero 的「用户长期记忆提取器」。从用户与助手的对话中提取
值得长期记住的用户信息，用于后续个性化饮食推荐。

【提取范围】
1. 饮食偏好：喜欢/讨厌的食材、菜系、口味（清淡/重口/辣）
2. 健康目标：减肥/增肌/控糖/高蛋白等目标及量化指标（体重、热量预算）
3. 饮食限制：过敏、忌口、素食、宗教饮食要求、医生建议
4. 关键事实：人数、生活方式、厨具条件、常做饭场景

【规则】
- 只提取明确表达的信息，不臆测
- 每条记忆独立、简短、客观（第三人称："用户不喜欢吃香菜"）
- 忽略寒暄和一次性请求
- 输出 JSON：{"memories": [{"content": "...", "type": "preference|goal|restriction|fact", "importance": 0.0-1.0}]}
- 没有可提取内容时输出 {"memories": []}
"""


class MemoryExtractor:
    """对话 -> 记忆条目提取。"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or _default_config()

    # ------------------------------------------------------------------
    async def extract(
        self,
        messages: List[Dict[str, str]],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """从消息列表提取记忆条目（LLM 优先，回退启发式）。"""
        llm_memories = await self._extract_with_llm(messages, user_id, session_id)
        if llm_memories:
            return llm_memories
        return self._extract_heuristic(messages)

    async def _extract_with_llm(
        self,
        messages: List[Dict[str, str]],
        user_id: Optional[str],
        session_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """LLM 提取（失败返回空列表）。"""
        try:
            from app.llm.provider import LLMProvider
            from app.llm.context import llm_context
            from app.config import settings

            provider = LLMProvider(settings.llm)
            invoker = provider.create_invoker(
                llm_type=self.config.extraction_llm_type, temperature=0.0
            )

            # 取最近的消息（截断避免超长）
            recent = messages[-20:]
            conv_text = "\n".join(
                f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')[:300]}"
                for m in recent
            )

            with llm_context("memory_extractor", user_id, session_id):
                response = await invoker.ainvoke(
                    [
                        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                        {"role": "user", "content": f"【对话内容】\n{conv_text}"},
                    ]
                )
            content = getattr(response, "content", "") or ""
            from app.utils.structured_json import parse_json_auto

            result = parse_json_auto(content)
            memories = result.get("memories", [])
            normalized = []
            for m in memories:
                if isinstance(m, dict) and m.get("content"):
                    mtype = m.get("type", "fact")
                    if mtype not in ("preference", "goal", "restriction", "fact"):
                        mtype = "fact"
                    normalized.append(
                        {
                            "content": str(m["content"]).strip()[:500],
                            "type": mtype,
                            "importance": max(0.0, min(1.0, float(m.get("importance", 0.5)))),
                        }
                    )
            return normalized
        except Exception as e:
            logger.debug("LLM memory extraction failed, using heuristic: %s", e)
            return []

    # ------------------------------------------------------------------
    def _extract_heuristic(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """启发式提取：命中关键词的用户语句直接入库。"""
        keywords = self.config.extraction_keywords or []
        memories: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            text = msg.get("content", "")
            for kw in keywords:
                if kw in text:
                    # 提取包含关键词的句子
                    for sentence in re.split(r"[。！？!?]", text):
                        sentence = sentence.strip()
                        if kw in sentence and 4 <= len(sentence) <= 120:
                            memory_type = self._classify(sentence, kw)
                            memories.append(
                                {
                                    "content": f"用户{sentence}",
                                    "type": memory_type,
                                    "importance": 0.6
                                    if memory_type == "restriction"
                                    else 0.5,
                                }
                            )
                            break  # 每句只取一次
                    break  # 每条消息只取一条（避免噪音）
        return memories

    @staticmethod
    def _classify(sentence: str, keyword: str) -> str:
        """按关键词/句式分类。"""
        if any(k in sentence for k in ("过敏", "忌口", "不吃", "不能吃", "禁止")):
            return "restriction"
        if any(k in sentence for k in ("目标是", "希望", "想要", "打算", "减肥", "减脂", "增肌", "控糖")):
            return "goal"
        if any(k in sentence for k in ("喜欢", "爱吃", "讨厌", "偏好", "口味")):
            return "preference"
        return "fact"


def _default_config() -> MemoryConfig:
    try:
        from app.config import settings

        return settings.memory
    except Exception:
        return MemoryConfig()


# 全局单例
memory_extractor = MemoryExtractor()
