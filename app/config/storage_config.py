# app/config/storage_config.py
"""
统一存储后端配置。

YAML 配置段: storage
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    """存储后端配置。"""

    # backend: memory（单机/默认） | redis（生产多实例）
    backend: str = Field(default="memory", pattern="^(memory|redis)$")
    # Redis 连接（默认复用 database.redis 的连接配置）
    redis_host: Optional[str] = None
    redis_port: Optional[int] = None
    redis_db: Optional[int] = None
    redis_password: Optional[str] = None
    # 事件流默认 TTL（秒），超过自动过期（断点恢复窗口）
    event_stream_ttl_seconds: float = Field(default=3600.0, ge=0)
    # 工具 SLO 记录窗口上限
    tool_metrics_max_records: int = Field(default=1000, ge=10)
