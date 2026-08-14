from __future__ import annotations


# app/config/database_config.py
"""
Database configuration for CookHero.
Contains all database connection configurations:
- PostgreSQL: persistent storage for conversations and user data
- Redis: caching layer (L1 exact match cache)
- Milvus: vector database for embeddings and semantic cache
"""

from typing import Optional

from pydantic import BaseModel


class PostgresConfig(BaseModel):
    """PostgreSQL database configuration."""

    # Connection settings
    host: str = "localhost"
    port: int = 5432
    database: str = "cookhero"
    user: str = "cookhero"
    password: Optional[str] = None

    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800  # Recycle connections after 30 minutes

    # Enable echo for debugging (logs all SQL statements)
    echo: bool = False

    @property
    def async_url(self) -> str:
        """Build async database URL for SQLAlchemy."""
        password_part = f":{self.password}" if self.password else ""
        return f"postgresql+asyncpg://{self.user}{password_part}@{self.host}:{self.port}/{self.database}"

    @property
    def sync_url(self) -> str:
        """Build sync database URL for SQLAlchemy."""
        password_part = f":{self.password}" if self.password else ""
        return f"postgresql+psycopg2://{self.user}{password_part}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseModel):
    """Redis configuration for caching."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None


class MilvusConfig(BaseModel):
    """Milvus vector database configuration."""

    host: str = "localhost"
    port: int = 19530
    user: Optional[str] = None
    password: Optional[str] = None
    secure: bool = False


class DatabaseConfig(BaseModel):
    """
    Unified database configuration container.
    Contains all database connection settings for the application.
    """

    postgres: PostgresConfig = PostgresConfig()
    redis: RedisConfig = RedisConfig()
    milvus: MilvusConfig = MilvusConfig()
