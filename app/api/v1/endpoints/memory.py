# app/api/v1/endpoints/memory.py
"""
P2 长期记忆 API：查看 / 手动添加 / 删除 / 清空。
前端个人中心"长期记忆"面板数据源。
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["Memory"])


class MemoryCreate(BaseModel):
    """手动添加记忆。"""

    content: str = Field(..., min_length=2, max_length=500, description="记忆内容")
    memory_type: str = Field(
        default="fact",
        pattern="^(preference|goal|restriction|fact)$",
        description="类型: preference/goal/restriction/fact",
    )
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="重要性 0-1")


@router.get("")
async def list_user_memories(
    request: Request,
    memory_type: Optional[str] = Query(None, pattern="^(preference|goal|restriction|fact)$"),
    limit: int = Query(100, ge=1, le=500),
):
    """列出当前用户的长期记忆。"""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="需要登录")

    from app.memory.store import memory_store

    memories = await memory_store.list(user_id, memory_type=memory_type, limit=limit)
    return {"memories": memories, "total": len(memories)}


@router.post("", status_code=201)
async def add_user_memory(body: MemoryCreate, request: Request):
    """手动添加一条长期记忆。"""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="需要登录")

    from app.memory.store import memory_store

    memory = await memory_store.add(
        user_id=user_id,
        content=body.content,
        memory_type=body.memory_type,
        importance=body.importance,
        source="manual",
    )
    if memory is None:
        # 重复内容返回现有记忆
        existing = await memory_store.list(user_id, limit=1)
        return {"message": "记忆已存在（跳过重复）", "memory": existing[0] if existing else None}
    return {"message": "记忆已添加", "memory": memory.to_dict()}


@router.delete("/{memory_id}")
async def delete_user_memory(memory_id: str, request: Request):
    """删除一条长期记忆。"""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="需要登录")

    from app.memory.store import memory_store

    ok = await memory_store.delete(user_id, memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"message": "记忆已删除"}


@router.delete("")
async def clear_user_memories(request: Request):
    """清空当前用户的全部长期记忆。"""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="需要登录")

    from app.memory.store import memory_store

    count = await memory_store.clear(user_id)
    return {"message": f"已清空 {count} 条记忆"}
