from __future__ import annotations


# app/telemetry/trace.py
"""
traceId 全链路追踪上下文

通过 contextvars 在调用栈中传递 trace_id / span，无需修改函数签名。
- trace_id: 一次用户请求的唯一 ID（可从 HTTP 头注入或自动生成）
- span: 调用链中的一个环节（如 LLM 调用、工具调用、压缩），形成调用链

用法::

    from app.telemetry.trace import trace, span, get_trace_id

    with trace(trace_id="req-123"):
        with span("llm_call", model="gpt-4o"):
            ...
        log_structured_event("answer", {...})
"""


import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

_trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_span_stack_var: ContextVar[list] = ContextVar("span_stack", default=[])
_attributes_var: ContextVar[dict] = ContextVar("trace_attributes", default={})


@dataclass
class Span:
    """调用链中的一个环节"""

    name: str
    span_id: str
    parent_id: Optional[str]
    started_at: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "started_at": self.started_at,
            "attributes": self.attributes,
        }


def new_trace_id() -> str:
    """生成新的 traceId。"""
    return uuid.uuid4().hex


def get_trace_id() -> Optional[str]:
    """获取当前 traceId。"""
    return _trace_id_var.get()


def set_trace_id(trace_id: str) -> None:
    """设置当前 traceId。"""
    _trace_id_var.set(trace_id)


def clear_trace_id() -> None:
    """清除当前 traceId。"""
    _trace_id_var.set(None)


def get_span_stack() -> list:
    return _span_stack_var.get()


def get_current_span() -> Optional[Span]:
    stack = _span_stack_var.get()
    return stack[-1] if stack else None


def get_span_id() -> Optional[str]:
    span = get_current_span()
    return span.span_id if span else None


def set_attribute(key: str, value: Any) -> None:
    """在 trace 上挂载自定义属性（如 user_id、session_id）。"""
    attrs = dict(_attributes_var.get())
    attrs[key] = value
    _attributes_var.set(attrs)


def get_attributes() -> dict:
    return dict(_attributes_var.get())


@contextmanager
def trace(trace_id: Optional[str] = None, **attributes: Any) -> Iterator[str]:
    """
    traceId 上下文管理器。

    用法::

        with trace(trace_id="req-1", user_id="u1") as tid:
            ...

    Args:
        trace_id: 显式指定 traceId（否则自动生成）
        **attributes: 附加属性（user_id / session_id 等）

    Yields:
        当前 traceId
    """
    tid = trace_id or new_trace_id()
    prev_tid = _trace_id_var.get()
    prev_stack = _span_stack_var.get()
    prev_attrs = _attributes_var.get()

    _trace_id_var.set(tid)
    _span_stack_var.set([])
    _attributes_var.set(dict(prev_attrs))
    for k, v in attributes.items():
        set_attribute(k, v)

    try:
        yield tid
    finally:
        _trace_id_var.set(prev_tid)
        _span_stack_var.set(prev_stack)
        _attributes_var.set(prev_attrs)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """
    调用链 span 上下文管理器。

    用法::

        with span("llm_call", model="gpt-4o", llm_type="fast") as sp:
            ...

    Args:
        name: span 名称
        **attributes: span 属性

    Yields:
        Span 实例
    """
    stack = list(_span_stack_var.get())
    parent = stack[-1] if stack else None
    sp = Span(
        name=name,
        span_id=uuid.uuid4().hex,
        parent_id=parent.span_id if parent else None,
        started_at=datetime.now(timezone.utc).isoformat(),
        attributes=dict(attributes),
    )
    stack.append(sp)
    _span_stack_var.set(stack)
    try:
        yield sp
    finally:
        stack = list(_span_stack_var.get())
        if stack:
            stack.pop()
        _span_stack_var.set(stack)
