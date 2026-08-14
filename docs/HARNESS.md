# Harness 源码思想在 CookHero 中的落地

> 本文档说明如何将 **DeepSeek Harness（DSH）** 的 Agent 运行时思想应用到
> CookHero 的 Agent 系统中。代码位于 `app/harness/`。

## 1. 目标工具：持久化目标 + 版本号乐观并发（`goals.py`）

**DSH 思想**：长任务以持久化 goal 追踪，跨自动续跑轮次存续；每次更新基于
`get_goal` 返回的精确 revision（乐观并发控制）；会话恢复后目标自动解除武装
（disarmed），需显式 `resume` 重新武装，防止误续跑；状态机
`active -> paused/completed/blocked`，`blocked` 需要给出持续阻塞的具体条件。

**CookHero 落地**：`GoalStore`（JSON 文件持久化 + 进程锁）：

| DSH 概念 | CookHero 实现 |
| --- | --- |
| create_goal | `GoalStore.create_goal(objective, max_goal_rounds)` |
| get_goal（含 revision） | `GoalStore.get_goal(goal_id)` |
| update_goal（edit/pause/resume/complete/blocked） | `GoalStore.update_goal(goal_id, revision, action, ...)` |
| revision 冲突检测 | `GoalConflictError`（并发更新检测） |
| 续跑轮次上限 | `rounds_started / max_goal_rounds`（达上限自动 complete） |
| armed/disarmed | `activation` 字段，resume 时重新 armed |

## 2. 任务清单：整体替换语义（`todos.py`）

**DSH 思想**：`todo_write` 每次提交**完整清单**（无部分更新、无逐项编辑），
状态机 `pending -> in_progress -> completed`；允许多项 in_progress（并行）；
未完成时至少一项 in_progress。

**CookHero 落地**：`TodoStore.replace(scope, items)` 整体替换，
`TodoStore.mark(scope, content, status)` 状态推进，按 trace_id/goal_id 作用域
持久化（`data/harness/todos/<scope>.json`），可与轨迹回放联动排查执行计划。

## 3. 结构化结果：schema 校验，失败即失败（`schema.py`）

**DSH 思想**：子代理/工具返回对象时可带 JSON Schema，运行时校验；
校验失败的结果不进入后续阶段（fail fast），防止脏数据扩散。

**CookHero 落地**：`validate_result(obj, schema)` 实现 JSON Schema 子集
（type / properties / required / additionalProperties / items / enum /
const / oneOf），返回 `ValidationResult(valid, errors)`，errors 带路径定位
（如 `properties.name: 期望 string, 实际 integer`）。`workflow.run_structured`
将校验集成进子任务执行。

## 4. 后台作业：ID 追踪 + 输出收集 + 终止（`jobs.py`）

**DSH 思想**：长任务以 background job 运行，返回 job_id；非阻塞轮询状态、
按需收集输出（wait 可阻塞至终态）、可随时 kill。

**CookHero 落地**：`JobManager`（asyncio 任务注册表）——
`start(name, coro_fn, ...) -> job_id`，`status(job_id)`，
`await output(job_id, wait=True, timeout_ms=...)`，`kill(job_id, reason)`。
适用于评测跑批、压缩、图片生成等长任务。

## 5. 多阶段编排（`workflow.py`）

**DSH 思想**：workflow 以 phase 组织；阶段内并发 fan-out，阶段间 barrier；
`pipeline` 流水线（stage 间无 barrier，item 独立失败降级为 null）。

**CookHero 落地**：

- `run_phases(phases, on_phase_start, on_phase_end)`：顺序执行阶段，阶段内
  `parallel` 并发，支持阶段结果 schema 校验
- `pipeline(items, *stages)`：item 逐级流式处理，失败 item 置 None 不阻塞
- `parallel(thunks)`：并发 + barrier，失败 thunk 降级为 None

## 6. 轨迹回放调试（`app/telemetry/trajectory.py`）

**DSH 思想**：Agent turn 轨迹 JSON 持久化，支持回放调试；结合 traceId
全链路串联推理 → 工具调用 → 最终回答。

**CookHero 落地**：`TrajectoryRecorder` 将每次 Agent 执行的完整调用链落盘为
`data/trajectories/<trace_id>.json`；`load_trajectory / list_trajectories /
render_replay` 提供回放；结合 `window_stats` 可诊断**长对话记忆退化**
（滑动窗口丢弃了哪些历史消息）。

## 7. 配置外部化（`config.yml`）

**DSH 思想**：配置以 YAML/补丁层外部化管理，运行时可改，不侵入代码。

**CookHero 落地**：新增 `tokenizer:` / `resilience:` / `telemetry:` 配置段，
经 `app/config/config_loader.py` 加载进 `settings`，与既有 `llm:` / `evaluation:`
等段落统一管理。

## 与四项目标的关系

| 目标 | 对应 Harness 组件 |
| --- | --- |
| 1. Token 计数/滑动窗口/摘要压缩 | `tokenizer`/`window` 配置 + 轨迹 `window_stats` 回放诊断 |
| 2. 韧性调用/结构化日志/轨迹回放/YAML 配置 | `resilience` 配置、telemetry、trajectory、config.yml |
| 3. RAG grounding truth 评测 | `workflow.run_phases` 编排批量评测 + `jobs` 后台跑批 |
| 4. Harness 思想落地 | 本目录全部组件 |
