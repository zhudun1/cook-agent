# app/storage/__init__.py
"""
统一存储后端（Storage Backend）

把进程内组件的数据面抽象为可插拔后端：
- MemoryBackend（单机默认）
- RedisBackend（生产多实例）

接入组件:
- cost_guard      -> hash 计数（cost:usage:{session}）
- approval        -> kv 对象（approval:req:{id}）
- event_stream    -> list 流（event:stream:{session}）
- tool_metrics    -> list 记录（toolmetrics:records）

用法::

    from app.storage.backend import get_storage_backend
    backend = get_storage_backend()
    await backend.hincrby("counter", "field", 1)
"""
