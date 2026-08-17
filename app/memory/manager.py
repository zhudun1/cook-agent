# app/memory/manager.py
"""
记忆管理器（Memory Manager）

整合提取与回忆，为 Agent 提供跨会话长期记忆：
- build_context(user_id, query): 检索相关记忆 -> 组装上下文文本（注入 system prompt）
- extract_and_store(session_id, user_id, messages): 对话后后台提取并入库
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.config.memory_config import MemoryConfig
from app.memory.extractor import memory_extractor
from app.memory.store import memory_store

logger = logging.getLogger(__name__)


class MemoryManager:
    """长期记忆编排。"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or _default_config()

    # ------------------------------------------------------------------
    async def build_context(
        self,
        user_id: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        """
        构建长期记忆上下文（注入 system prompt 用）。

        Args:
            user_id: 用户 ID
            query: 当前用户消息（用于相关度检索）

        Returns:
            上下文文本（无可回忆记忆时返回空串）
        """
        if not self.config.enabled or not user_id:
            return ""
        try:
            memories = await memory_store.recall(user_id, query, top_k=top_k)
        except Exception as e:
            logger.warning("Memory recall failed: %s", e)
            return ""
        if not memories:
            return ""

        parts = []
        for m in memories:
            type_label = {
                "preference": "偏好",
                "goal": "目标",
                "restriction": "限制",
                "fact": "事实",
            }.get(m.get("memory_type"), "事实")
            parts.append(f"- [{type_label}] {m.get('content')}")
        return "以下是用户的长期记忆，请在回答中参考（尤其注意【限制】类）：\n" + "\n".join(parts)

    # ------------------------------------------------------------------
    async def extract_and_store(
        self,
        session_id: str,
        user_id: str,
        messages: List[Dict[str, str]],
    ) -> int:
        """
        对话结束后提取并存储记忆（后台任务调用）。

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            messages: 本轮对话消息列表 [{role, content}]

        Returns:
            新增记忆条数
        """
        if not self.config.enabled or not user_id:
            return 0
        if not self.config.auto_extract:
            return 0
        if len(messages) < self.config.min_messages_for_extraction:
            return 0

        try:
            memories = await memory_extractor.extract(
                messages, user_id=user_id, session_id=session_id
            )
        except Exception as e:
            logger.warning("Memory extraction failed: %s", e)
            return 0

        stored = 0
        for m in memories:
            try:
                result = await memory_store.add(
                    user_id=user_id,
                    content=m["content"],
                    memory_type=m.get("type", "fact"),
                    importance=m.get("importance", 0.5),
                    source="extracted",
                    source_session_id=session_id,
                )
                if result:
                    stored += 1
            except Exception as e:
                logger.debug("Memory store failed for %r: %s", m.get("content"), e)
        if stored:
            logger.info(
                "Stored %d long-term memories for user %s (session %s)",
                stored,
                user_id,
                session_id,
            )
        return stored


def _default_config() -> MemoryConfig:
    try:
        from app.config import settings

        return settings.memory
    except Exception:
        return MemoryConfig()


# 全局单例
memory_manager = MemoryManager()
