# app/database/session.py
"""
Async database session management for CookHero.
Provides session factory and dependency injection for FastAPI.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.database.models import Base

logger = logging.getLogger(__name__)

# 支持 DATABASE_URL 环境变量覆盖（12-factor）。
# 生产默认走 config.yml 的 PostgreSQL；本地开发/测试可用
# sqlite+aiosqlite:///./cookhero.db 一键起服务。
DATABASE_URL = os.getenv("DATABASE_URL") or settings.database.postgres.async_url
_IS_SQLITE = DATABASE_URL.startswith("sqlite")


def _create_engine(url: str, *, pool_size: int = 5, max_overflow: int = 10) -> AsyncEngine:
    """按方言创建异步引擎（sqlite 不使用连接池参数）。"""
    if url.startswith("sqlite"):
        return create_async_engine(url, echo=settings.database.postgres.echo)
    return create_async_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=settings.database.postgres.pool_timeout,
        pool_recycle=settings.database.postgres.pool_recycle,
        echo=settings.database.postgres.echo,
    )


# Create async engine
_engine = _create_engine(
    DATABASE_URL,
    pool_size=settings.database.postgres.pool_size,
    max_overflow=settings.database.postgres.max_overflow,
)

# Create session factory
async_session_factory = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


# ==================== Background Thread Database Support ====================
# Separate engine and session factory for use in background threads
# (e.g., LLM usage logging callbacks that run in a different event loop)

_background_engine: Optional[AsyncEngine] = None
_background_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_background_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create a session factory for use in background threads.

    This creates a separate database engine that can be used from
    a different event loop than the main FastAPI application.
    """
    global _background_engine, _background_session_factory

    if _background_session_factory is None:
        _background_engine = _create_engine(
            DATABASE_URL,
            pool_size=2,  # Smaller pool for background operations
            max_overflow=2,
        )
        _background_session_factory = async_sessionmaker(
            bind=_background_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    return _background_session_factory


@asynccontextmanager
async def get_background_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for background thread session handling."""
    factory = get_background_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_background_db() -> None:
    """Close background database connections."""
    global _background_engine, _background_session_factory
    if _background_engine is not None:
        await _background_engine.dispose()
        _background_engine = None
        _background_session_factory = None
        logger.info("Background database connections closed.")


# ==================== Main Database Functions ====================


async def init_db() -> None:
    """Initialize database schema (create tables if not exist)."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized.")

    # 轻量迁移：为已有 PostgreSQL 库补充新增的评测字段（幂等）。
    # SQLite（本地开发）由 create_all 全量重建，无需迁移。
    if _IS_SQLITE:
        return
    try:
        from sqlalchemy import text

        async with _engine.begin() as conn:
            for ddl in (
                "ALTER TABLE rag_evaluations ADD COLUMN IF NOT EXISTS answer_correctness FLOAT",
                "ALTER TABLE rag_evaluations ADD COLUMN IF NOT EXISTS reference_answer TEXT",
                "ALTER TABLE rag_evaluations ADD COLUMN IF NOT EXISTS reference_contexts JSON",
            ):
                await conn.execute(text(ddl))
    except Exception as e:
        logger.warning("Evaluation column migration skipped: %s", e)


async def close_db() -> None:
    """Close database connections."""
    await _engine.dispose()
    logger.info("Database connections closed.")


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection for async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for manual session handling."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
