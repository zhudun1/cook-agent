from __future__ import annotations


# app/telemetry/logger.py
"""
结构化 JSON 日志

在 stdlib logging 之上提供 JSON 行输出，自动携带 traceId / spanId / 模块名，
支持敏感字段递归脱敏。

用法::

    from app.telemetry.logger import log_structured_event, setup_structured_logging

    log_structured_event("llm_call", {"model": "gpt-4o", "tokens": 123})
    setup_structured_logging()  # 应用启动时调用
"""


import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.telemetry.trace import (
    get_attributes,
    get_span_id,
    get_trace_id,
)

logger = logging.getLogger(__name__)

_STRUCTURED_LOGGER_NAME = "cookhero.structured"

# 已挂载标志（进程级）
_setup_done = False
_handler: Optional["JsonLineHandler"] = None


def redact(value: Any, sensitive_fields: list[str], _key: Optional[str] = None) -> Any:
    """递归脱敏：字段名命中敏感列表时替换为 '***'。"""
    if _key and any(s in _key.lower() for s in sensitive_fields):
        return "***"
    if isinstance(value, dict):
        return {
            k: redact(v, sensitive_fields, _key=k) for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v, sensitive_fields, _key) for v in value]
    if isinstance(value, str) and len(value) > 100_000:
        return value[:100_000] + "...[truncated]"
    return value


class JsonLineHandler(logging.Handler):
    """将日志记录序列化为 JSON 行（含 traceId/spanId）。"""

    def __init__(self, file_path: Optional[str] = None, sensitive_fields: Optional[list] = None):
        super().__init__()
        self.sensitive_fields = sensitive_fields or [
            "api_key", "token", "password", "authorization",
        ]
        self._stream = None
        self._file_path = file_path
        if file_path:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = open(path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "trace_id": get_trace_id(),
                "span_id": get_span_id(),
            }
            attrs = get_attributes()
            if attrs:
                entry["attributes"] = redact(attrs, self.sensitive_fields)

            extra = getattr(record, "event_data", None)
            if extra:
                entry["event"] = redact(extra, self.sensitive_fields)
            if getattr(record, "event_type", None):
                entry["event_type"] = record.event_type

            line = json.dumps(entry, ensure_ascii=False, default=str)
            if self._stream:
                self._stream.write(line + "\n")
                self._stream.flush()
            else:
                print(line, flush=True)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None
        super().close()


def setup_structured_logging(
    enabled: bool = True,
    file_path: Optional[str] = None,
    level: str = "INFO",
    sensitive_fields: Optional[list] = None,
) -> Optional[JsonLineHandler]:
    """
    挂载结构化 JSON 日志处理器（幂等）。

    Args:
        enabled: 是否启用
        file_path: JSON 行日志文件路径（None 则输出到 stdout）
        level: 最低日志级别
        sensitive_fields: 敏感字段名列表

    Returns:
        挂载的 handler（未启用时返回 None）
    """
    global _setup_done, _handler
    if _setup_done:
        return _handler
    _setup_done = True

    if not enabled:
        return None

    handler = JsonLineHandler(file_path=file_path, sensitive_fields=sensitive_fields)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    structured_logger = logging.getLogger(_STRUCTURED_LOGGER_NAME)
    structured_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    structured_logger.addHandler(handler)
    structured_logger.propagate = False

    _handler = handler
    logger.info(
        "Structured JSON logging enabled: file=%s level=%s",
        file_path or "(stdout)",
        level,
    )
    return handler


def log_structured_event(
    event_type: str,
    data: Dict[str, Any],
    level: int = logging.INFO,
) -> None:
    """
    输出一条结构化事件日志（携带当前 traceId/spanId）。

    Args:
        event_type: 事件类型（llm_call / tool_call / agent_reasoning / answer ...）
        data: 事件数据（会递归脱敏）
        level: 日志级别
    """
    structured_logger = logging.getLogger(_STRUCTURED_LOGGER_NAME)
    structured_logger.log(
        level,
        event_type,
        extra={"event_type": event_type, "event_data": data},
    )


def get_structured_logger() -> logging.Logger:
    """获取结构化日志 logger（供自定义记录）。"""
    return logging.getLogger(_STRUCTURED_LOGGER_NAME)


def configure_from_settings() -> None:
    """按全局配置挂载结构化日志（应用启动时调用）。"""
    try:
        from app.config import settings

        cfg = settings.telemetry.structured_log
        setup_structured_logging(
            enabled=cfg.enabled,
            file_path=cfg.file_path,
            level=cfg.level,
            sensitive_fields=cfg.sensitive_fields,
        )
    except Exception as e:
        logger.warning("Failed to configure structured logging: %s", e)
