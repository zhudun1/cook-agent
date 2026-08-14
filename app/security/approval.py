# app/security/approval.py
"""
人工介入审批（Human-in-the-Loop Approval）

危险工具（如删除、下单、发消息、生成外部内容）调用前需用户审批：

流程:
1. Agent 请求调用危险工具 -> ApprovalManager.request() 生成 approval_id（pending）
2. SSE 事件 approval_requested 通知前端（含工具名、参数、approval_id）
3. 用户通过 API approve/reject
4. Agent 侧轮询审批结果（带超时）：
   - approved  -> 执行工具
   - rejected  -> 返回结构化错误 APPROVAL_DENIED（Agent 可告知用户/换方案）
   - timeout   -> 返回结构化错误 APPROVAL_TIMEOUT

线程安全：asyncio 单事件循环内使用 dict + 轮询；决策写入与读取均为同步。
"""

from __future__ import annotations

import fnmatch
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from app.config.security_config import ApprovalConfig

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ApprovalRequest:
    """一次审批请求。"""

    approval_id: str
    tool_name: str
    arguments: Dict[str, Any]
    user_id: Optional[str]
    session_id: Optional[str]
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    decided_at: Optional[str] = None
    decided_by: Optional[str] = None
    timeout_seconds: float = 120.0

    def to_dict(self) -> dict:
        return {
            "approval_id": self.approval_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "decided_by": self.decided_by,
            "timeout_seconds": self.timeout_seconds,
        }


class ApprovalManager:
    """
    审批管理器。

    用法::

        mgr = ApprovalManager()
        req = mgr.request("image_generator", {"prompt": "..."}, user_id="u1", session_id="s1")
        # 前端展示 req.to_dict()，用户 approve/reject:
        mgr.decide(req.approval_id, approve=True, by_user="u1")
        # Agent 侧轮询:
        status = mgr.get_status(req.approval_id)  # 超时自动判 TIMEOUT
    """

    def __init__(self, config: Optional[ApprovalConfig] = None):
        self.config = config or _default_config()
        self._requests: Dict[str, ApprovalRequest] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def requires_approval(self, tool_name: str) -> bool:
        """判断工具是否需要审批（glob 模式匹配）。"""
        if not self.config.enabled:
            return False
        patterns = self.config.required_tool_patterns or []
        return any(fnmatch.fnmatch(tool_name, p) for p in patterns)

    def auto_approve(self, user_id: Optional[str]) -> bool:
        """admin 用户是否自动通过。"""
        return (
            self.config.auto_approve_admin
            and user_id is not None
            and user_id in (self.config.admin_users or [])
        )

    def request(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> ApprovalRequest:
        """发起审批请求（返回 pending 状态）。"""
        req = ApprovalRequest(
            approval_id=uuid.uuid4().hex,
            tool_name=tool_name,
            arguments=arguments or {},
            user_id=user_id,
            session_id=session_id,
            timeout_seconds=timeout_seconds or self.config.timeout_seconds,
        )
        with self._lock:
            self._requests[req.approval_id] = req
        logger.info(
            "Approval requested: %s (%s) by user %s",
            req.approval_id,
            tool_name,
            user_id,
        )
        return req

    def decide(self, approval_id: str, approve: bool, by_user: Optional[str] = None) -> bool:
        """用户决策：批准/拒绝。"""
        with self._lock:
            req = self._requests.get(approval_id)
            if req is None or req.status != ApprovalStatus.PENDING:
                return False
            req.status = (
                ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
            )
            req.decided_at = datetime.now(timezone.utc).isoformat()
            req.decided_by = by_user
        logger.info(
            "Approval %s %s by %s",
            approval_id,
            "approved" if approve else "rejected",
            by_user,
        )
        return True

    def get_status(self, approval_id: str) -> Optional[ApprovalRequest]:
        """
        查询审批状态；超时未决策的自动标记 TIMEOUT（懒判定）。

        注意：pending 请求在 Agent 轮询侧应以 `timeout_seconds` 为准自行判断，
        此处仅将超过 `timeout_seconds` 的 pending 置为 timeout（用于展示）。
        """
        with self._lock:
            req = self._requests.get(approval_id)
            if req is None:
                return None
            if req.status == ApprovalStatus.PENDING:
                elapsed = _now_ts() - _iso_to_ts(req.created_at)
                if elapsed > req.timeout_seconds:
                    req.status = ApprovalStatus.TIMEOUT
                    req.decided_at = datetime.now(timezone.utc).isoformat()
            return req

    def wait_for_decision(
        self,
        approval_id: str,
        timeout_seconds: Optional[float] = None,
        poll_interval: float = 0.5,
    ) -> ApprovalRequest:
        """
        阻塞等待审批决策（Agent 侧使用）。

        Args:
            approval_id: 审批 ID
            timeout_seconds: 等待上限（默认用请求自身的 timeout）
            poll_interval: 轮询间隔（秒）

        Returns:
            最终状态（approved/rejected/timeout）
        """
        req = self._requests.get(approval_id)
        if req is None:
            raise KeyError(f"Approval not found: {approval_id}")
        timeout = timeout_seconds or req.timeout_seconds
        deadline = _now_ts() + timeout
        while True:
            current = self.get_status(approval_id)
            if current is None:
                return req
            if current.status != ApprovalStatus.PENDING:
                return current
            if _now_ts() >= deadline:
                with self._lock:
                    current.status = ApprovalStatus.TIMEOUT
                    current.decided_at = datetime.now(timezone.utc).isoformat()
                return current
            time.sleep(poll_interval)

    async def await_decision(
        self,
        approval_id: str,
        timeout_seconds: Optional[float] = None,
        poll_interval: float = 0.5,
    ) -> ApprovalRequest:
        """异步等待审批决策（不阻塞事件循环）。"""
        req = self._requests.get(approval_id)
        if req is None:
            raise KeyError(f"Approval not found: {approval_id}")
        timeout = timeout_seconds or req.timeout_seconds
        import asyncio

        deadline = _now_ts() + timeout
        while True:
            current = self.get_status(approval_id)
            if current is None:
                return req
            if current.status != ApprovalStatus.PENDING:
                return current
            if _now_ts() >= deadline:
                with self._lock:
                    current.status = ApprovalStatus.TIMEOUT
                    current.decided_at = datetime.now(timezone.utc).isoformat()
                return current
            await asyncio.sleep(poll_interval)

    def list_pending(self, session_id: Optional[str] = None, limit: int = 50) -> List[dict]:
        """列出待审批请求（前端展示）。"""
        with self._lock:
            reqs = [
                r
                for r in self._requests.values()
                if r.status == ApprovalStatus.PENDING
                and (session_id is None or r.session_id == session_id)
            ]
            reqs.sort(key=lambda r: r.created_at)
            return [r.to_dict() for r in reqs[-limit:]]


def _default_config() -> ApprovalConfig:
    try:
        from app.config import settings

        return settings.security.approval
    except Exception:
        return ApprovalConfig()


def _now_ts() -> float:
    return time.time()


def _iso_to_ts(iso: str) -> float:
    try:
        dt = datetime.fromisoformat(iso)
        return dt.timestamp()
    except Exception:
        return _now_ts()


# 全局单例
approval_manager = ApprovalManager()
