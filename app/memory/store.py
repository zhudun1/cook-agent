# app/memory/store.py
"""
长期记忆存储（Long-Term Memory Store）

跨会话持久化用户记忆条目（偏好/目标/限制/事实）：
- 按用户隔离
- 回忆检索：关键词重叠 + 类型加权 + 重要性排序
- 上限裁剪：超 max_memories_per_user 时删除最低重要性/最旧的条目
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.memory_config import MemoryConfig
from app.database.models import UserMemoryModel
from app.database.session import get_session_context

logger = logging.getLogger(__name__)

# 类型权重（回忆检索时偏好/限制优先于一般事实）
TYPE_WEIGHTS = {
    "restriction": 1.5,
    "goal": 1.3,
    "preference": 1.2,
    "fact": 1.0,
}

# 中文分词简化：按 2-gram 提取关键词
def _keywords(text: str, min_len: int = 2) -> set[str]:
    """提取文本关键词（2-gram 重叠检索用）。"""
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", text)
    words: set[str] = set()
    for token in tokens:
        if len(token) <= 4:
            words.add(token)
        for i in range(len(token) - 1):
            words.add(token[i : i + 2])
    return words


class MemoryStore:
    """长期记忆存取。"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or _default_config()

    # ------------------------------------------------------------------
    async def add(
        self,
        user_id: str,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        source: str = "extracted",
        source_session_id: Optional[str] = None,
    ) -> Optional[UserMemoryModel]:
        """新增记忆条目（去重：内容互相包含则跳过）。"""
        content = content.strip()
        if not content or not user_id:
            return None

        async with get_session_context() as session:
            # 去重：已有完全相同的则跳过
            existing = (
                await session.execute(
                    select(UserMemoryModel).where(
                        UserMemoryModel.user_id == user_id,
                        UserMemoryModel.content == content,
                    )
                )
            ).scalar_one_or_none()
            if existing:
                return existing

            memory = UserMemoryModel(
                id=uuid.uuid4(),
                user_id=user_id,
                content=content,
                memory_type=memory_type,
                importance=max(0.0, min(1.0, importance)),
                source=source,
                source_session_id=source_session_id,
            )
            session.add(memory)
            await session.commit()

            # 上限裁剪
            await self._trim(session, user_id)
            return memory

    async def _trim(self, session: AsyncSession, user_id: str) -> None:
        """超出上限时删除最不重要/最旧的条目。"""
        limit = self.config.max_memories_per_user
        count = (
            await session.execute(
                select(UserMemoryModel.id).where(UserMemoryModel.user_id == user_id)
            )
        ).scalars().all()
        if len(count) <= limit:
            return
        excess = len(count) - limit
        # 按 (importance asc, created_at asc) 删除最早的
        to_delete = (
            await session.execute(
                select(UserMemoryModel.id)
                .where(UserMemoryModel.user_id == user_id)
                .order_by(UserMemoryModel.importance.asc(), UserMemoryModel.created_at.asc())
                .limit(excess)
            )
        ).scalars().all()
        if to_delete:
            await session.execute(
                delete(UserMemoryModel).where(UserMemoryModel.id.in_(to_delete))
            )
            await session.commit()

    # ------------------------------------------------------------------
    async def list(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """列出用户记忆。"""
        async with get_session_context() as session:
            stmt = select(UserMemoryModel).where(UserMemoryModel.user_id == user_id)
            if memory_type:
                stmt = stmt.where(UserMemoryModel.memory_type == memory_type)
            stmt = stmt.order_by(
                UserMemoryModel.importance.desc(), UserMemoryModel.created_at.desc()
            ).limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [r.to_dict() for r in rows]

    async def recall(
        self,
        user_id: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """
        回忆检索：返回与 query 相关的记忆（关键词重叠 + 类型加权 + 重要性）。

        Args:
            user_id: 用户 ID
            query: 当前消息/查询
            top_k: 返回条数（默认用配置）

        Returns:
            按相关度排序的记忆 dict 列表
        """
        async with get_session_context() as session:
            stmt = select(UserMemoryModel).where(UserMemoryModel.user_id == user_id)
            rows = (await session.execute(stmt)).scalars().all()

        if not rows:
            return []

        query_kw = _keywords(query)
        scored = []
        for m in rows:
            mem_kw = _keywords(m.content)
            overlap = len(query_kw & mem_kw)
            if overlap == 0:
                continue
            type_weight = TYPE_WEIGHTS.get(m.memory_type, 1.0)
            score = overlap * type_weight + m.importance
            scored.append((score, m))

        # 兜底：无关键词重叠时，限制类（安全相关）记忆仍按重要性返回，
        # 避免过敏/忌口等关键信息在泛化查询下被漏掉
        if not scored:
            restrictions = [
                (m.importance, m)
                for m in rows
                if m.memory_type == "restriction"
            ]
            if restrictions:
                restrictions.sort(key=lambda x: x[0], reverse=True)
                scored = [(1.0 + m.importance, m) for _, m in restrictions[:top_k]]

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = top_k or self.config.recall_top_k
        threshold = self.config.recall_importance_threshold

        results = []
        for score, m in scored[: top_k * 3]:
            if m.importance < threshold:
                continue
            results.append(m)
            if len(results) >= top_k:
                break

        # 更新访问统计
        if results:
            async with get_session_context() as session:
                for m in results:
                    m.access_count += 1
                    m.last_accessed_at = datetime.utcnow()
                await session.commit()

        return [r.to_dict() for r in results]

    async def delete(self, user_id: str, memory_id: str) -> bool:
        """删除一条记忆。"""
        async with get_session_context() as session:
            result = await session.execute(
                delete(UserMemoryModel).where(
                    UserMemoryModel.id == uuid.UUID(memory_id),
                    UserMemoryModel.user_id == user_id,
                )
            )
            await session.commit()
            return result.rowcount > 0

    async def clear(self, user_id: str) -> int:
        """清空用户全部记忆。"""
        async with get_session_context() as session:
            result = await session.execute(
                delete(UserMemoryModel).where(UserMemoryModel.user_id == user_id)
            )
            await session.commit()
            return result.rowcount or 0


def _default_config() -> MemoryConfig:
    try:
        from app.config import settings

        return settings.memory
    except Exception:
        return MemoryConfig()


# 全局单例
memory_store = MemoryStore()
