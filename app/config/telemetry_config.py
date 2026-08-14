from __future__ import annotations


# app/config/telemetry_config.py
"""
可观测性配置：traceId 全链路追踪、结构化日志、轨迹持久化。
"""

from pydantic import BaseModel, Field


class TraceConfig(BaseModel):
    """traceId 追踪配置。"""

    enabled: bool = True
    # 从 HTTP 请求头读取 traceId 的字段名（便于与网关/SIEM 联动）
    header_name: str = "X-Trace-Id"
    # 无上游 traceId 时是否自动生成
    auto_generate: bool = True


class StructuredLogConfig(BaseModel):
    """结构化 JSON 日志配置。"""

    enabled: bool = True
    # JSON 行日志文件路径（None 则仅 stdlib logging，不写文件）
    file_path: str | None = None
    # 最低日志级别（DEBUG/INFO/WARNING/ERROR）
    level: str = "INFO"
    # 需要脱敏的字段名（递归）
    sensitive_fields: list = Field(
        default_factory=lambda: ["api_key", "token", "password", "authorization"]
    )


class TrajectoryConfig(BaseModel):
    """Agent turn 轨迹持久化配置。"""

    enabled: bool = True
    # 轨迹 JSON 落盘目录（相对项目根）
    storage_dir: str = "data/trajectories"
    # 是否同时写入数据库（agent_messages.trace 已存，这里指独立轨迹表）
    persist_to_db: bool = False
    # 是否记录完整输入/输出（含敏感内容）；关闭时截断 content
    record_full_payload: bool = True
    max_payload_chars: int = Field(default=20000, ge=100)


class TelemetryConfig(BaseModel):
    """可观测性总配置。"""

    trace: TraceConfig = Field(default_factory=TraceConfig)
    structured_log: StructuredLogConfig = Field(default_factory=StructuredLogConfig)
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
