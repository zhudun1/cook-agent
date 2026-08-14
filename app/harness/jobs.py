# app/harness/jobs.py
"""
后台作业管理

借鉴 DeepSeek Harness 的 background job 模式：
- 长任务以 job_id 追踪，状态机：running -> completed | failed | killed
- 输出按需收集（job_output），支持 wait（阻塞至终态）
- 可随时 job_kill 取消

用法::

    manager = JobManager()
    job_id = manager.start("evaluation", my_coro, arg1, arg2)
    status = manager.status(job_id)          # 非阻塞
    output = await manager.output(job_id)    # 收集输出
    await manager.wait(job_id, timeout=30)   # 阻塞至终态
    manager.kill(job_id)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class BackgroundJob:
    """后台作业。"""

    job_id: str
    name: str
    status: JobStatus = JobStatus.RUNNING
    output: Any = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status.value,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "has_output": self.output is not None,
        }


class JobManager:
    """进程内后台作业管理器。"""

    def __init__(self):
        self._jobs: Dict[str, BackgroundJob] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def start(
        self,
        name: str,
        coro_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        启动一个后台作业。

        Args:
            name: 作业名称
            coro_fn: 协程函数（必须 async）
            *args / **kwargs: 传给 coro_fn 的参数

        Returns:
            job_id
        """
        job_id = uuid.uuid4().hex
        job = BackgroundJob(job_id=job_id, name=name)
        with self._lock:
            self._jobs[job_id] = job

        async def _runner() -> None:
            try:
                result = await coro_fn(*args, **kwargs)
                self._finish(job_id, JobStatus.COMPLETED, output=result)
            except asyncio.CancelledError:
                self._finish(job_id, JobStatus.KILLED, error="cancelled")
                raise
            except Exception as e:
                logger.error("Job %s (%s) failed: %s", job_id, name, e, exc_info=True)
                self._finish(job_id, JobStatus.FAILED, error=str(e))

        task = asyncio.create_task(_runner(), name=f"harness-job-{name}")
        with self._lock:
            self._tasks[job_id] = task
        logger.info("Job started: %s (%s)", job_id, name)
        return job_id

    def _finish(self, job_id: str, status: JobStatus, output: Any = None, error: Optional[str] = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            if status == JobStatus.COMPLETED:
                job.output = output
            else:
                job.error = error
            job.finished_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    def status(self, job_id: str) -> Optional[dict]:
        """查询作业状态（非阻塞）。"""
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def list(self, status: Optional[JobStatus] = None) -> List[dict]:
        """列出作业。"""
        with self._lock:
            jobs = [j.to_dict() for j in self._jobs.values()]
        if status:
            jobs = [j for j in jobs if j["status"] == status.value]
        return jobs

    async def output(self, job_id: str, wait: bool = False, timeout_ms: Optional[int] = None) -> dict:
        """
        读取作业输出（含实际 output / error 负载）。

        Args:
            job_id: 作业 ID
            wait: 是否阻塞至终态
            timeout_ms: 最大等待毫秒（wait=True 时有效）

        Returns:
            {status, output, error, ...}

        Raises:
            asyncio.TimeoutError: wait=True 且超时未终态
        """
        if wait:
            # 轮询等待终态（比 asyncio.Event 跨版本更健壮，无时序竞态）
            deadline = time.monotonic() + (timeout_ms or 30000) / 1000
            while True:
                with self._lock:
                    job = self._jobs.get(job_id)
                if job is None:
                    return {"job_id": job_id, "status": "unknown"}
                if job.status != JobStatus.RUNNING:
                    break
                if time.monotonic() >= deadline:
                    raise asyncio.TimeoutError(
                        f"Job {job_id} not finished within {(timeout_ms or 30000)/1000}s"
                    )
                await asyncio.sleep(0.02)

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"job_id": job_id, "status": "unknown"}
            data = job.to_dict()
            data["output"] = job.output
            data["error"] = job.error
            return data

    def kill(self, job_id: str, reason: Optional[str] = None) -> bool:
        """
        终止作业（仅运行中有效）。

        注意：状态置为 KILLED 是**同步**完成的（不依赖协程的取消处理），
        task.cancel() 仅作为底层清理。这是因为 Python 3.9 中取消一个
        尚未启动的任务不会执行协程体内的异常处理。
        """
        with self._lock:
            task = self._tasks.get(job_id)
            job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return False
        # 同步标记 KILLED（幂等；task 后续若完成会覆盖为实际终态）
        self._finish(job_id, JobStatus.KILLED, error=reason or "killed")
        if task and not task.done():
            task.cancel()
        logger.info("Job killed: %s (%s)", job_id, reason or "")
        return True

    async def shutdown(self) -> None:
        """取消所有运行中作业（应用退出时调用）。"""
        with self._lock:
            tasks = list(self._tasks.values())
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# 全局单例
job_manager = JobManager()
