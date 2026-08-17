# 统一存储后端（Storage Backend）

> 生产级 Agent 的水平扩展基础：把进程内组件的数据面抽象为可插拔后端，
> 支撑多实例部署时状态一致。

## 为什么需要

以下组件原本是**进程内内存状态**（dict / deque），单实例没问题，但**多实例部署即失效**：

| 组件 | 进程内数据 | 多实例风险 |
|---|---|---|
| 成本熔断（`cost_guard`） | 会话 token 计数 | 各实例各记各的，熔断失效 |
| 人工审批（`approval`） | 审批请求对象 | 请求在 A 实例，决策请求打到 B 实例找不到 |
| 事件流（`event_stream`） | 会话 SSE 事件 + seq | 断点恢复跨实例/重启丢流 |
| 工具 SLO（`tool_metrics`） | 调用记录窗口 | 指标只有单实例视角 |

## 架构

```
┌─────────────────────────────────────────────────────┐
│  StorageBackend（app/storage/backend.py）           │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │ MemoryBackend │    │ RedisBackend             │   │
│  │ 单机/默认     │    │ 生产多实例共享状态        │   │
│  └──────────────┘    └──────────────────────────┘   │
│  原语: hash / kv / list / ttl（值一律字符串）        │
└───────────────┬─────────────────────────────────────┘
                │ get_storage_backend()（单例工厂）
    ┌───────────┼───────────┬───────────┬──────────┐
    ▼           ▼           ▼           ▼
 cost_guard  approval   event_stream  tool_metrics
 (hash计数)  (kv对象)    (list流+seq)  (list窗口)
```

设计原则：

- **接口最小**：hash（`hincrby/hgetall/hset/hdel`）、kv（`get/set/delete/exists`）、
  list（`rpush/lrange/llen/ltrim`）、ttl（`expire`），值一律字符串
- **后端权威 / 本地镜像**：审批 `get_status` 以后端数据为准，进程内 dict 仅作镜像，
  保证任意实例的决策对其他实例立即可见
- **组件无感知切换**：组件通过 `get_storage_backend()` 获取后端，不关心具体实现

## 组件数据面映射

| 组件 | 存储原语 | Key 模式 |
|---|---|---|
| cost_guard | hash 自增 | `cost:usage:{session_id}` |
| approval | kv JSON + TTL | `approval:req:{approval_id}` |
| event_stream | list + seq 元数据 + TTL | `event:stream:{sid}` / `event:meta:{sid}` |
| tool_metrics | list 窗口 + 裁剪 | `toolmetrics:records` |

## 配置切换

```yaml
# config.yml
storage:
  backend: "memory"    # 单机默认（测试友好）
  # backend: "redis"   # 生产多实例（自动复用 database.redis 连接）
  event_stream_ttl_seconds: 3600   # 事件流 TTL（断点恢复窗口）
  tool_metrics_max_records: 1000   # SLO 记录窗口
```

Redis 连接默认复用 `database.redis` 配置；如需单独指定：

```yaml
storage:
  backend: "redis"
  redis_host: "localhost"
  redis_port: 6379
  redis_db: 0
  redis_password: null
```

## 快速开始（本地 Redis + redis 后端）

```bash
# 1. 一键启动本地 Redis（缺失时自动编译安装）
./scripts/start_redis.sh

# 2. 切换后端
#    config.yml -> storage.backend: "redis"

# 3. 重启后端
DATABASE_URL="sqlite+aiosqlite:///./cook_agent.db" uvicorn app.main:app --port 8000
```

验证后端生效（启动日志会出现 `Storage backend: redis`）。

## 验证

### 单元测试（无需 Redis）
```bash
pytest tests/test_storage_backend.py tests/test_storage_redis_backend.py -q
# - MemoryBackend 原语、TTL、list 边界
# - RedisBackend 语义（fakeredis 模拟）
# - 事件流 / 成本熔断在 Redis 语义下的行为
```

### 真实 Redis 端到端（需 Redis 运行）
四个组件在**两个独立实例**间验证跨实例一致性：

```
实例 A 写/发起         实例 B 读/决策
  cost_guard.record ──► cost_guard.check（读到计数）
  approval.request ───► approval.decide ──► A 的 get_status 读到 approved
  event_stream.append ► event_stream.get_events（增量 seq）
  tool_metrics.record ► tool_metrics.get_stats（聚合）
```

## 设计边界

- **AgentHub 会话级工具实例缓存保持进程内**：工具实例持有连接（如 MCP 客户端），
  跨进程共享无意义；已加大小上限治理（默认 2000）防止内存泄漏
- **告警标记（cost_guard._warned）保持进程内**：仅影响日志频率，无需共享
- 超时懒判定（审批 TIMEOUT）仅影响本地镜像展示；权威超时判定在 Agent 轮询侧
  （`await_decision` 的 deadline）
