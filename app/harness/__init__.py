# app/harness/__init__.py
"""
Harness 模式组件（借鉴 DeepSeek Harness 源码思想）

将 Agent 运行时常见的编排原语落地为可复用组件：
- goals:     持久化目标（带版本号乐观并发控制、暂停/恢复/完成/阻塞语义）
- todos:     结构化任务清单（replace 语义 + 状态机）
- schema:    结构化结果校验（JSON Schema 子集），保证工具/子代理输出可靠
- jobs:      后台作业管理（ID、状态、输出收集、终止）
- workflow:  多阶段编排（phase / pipeline / parallel / barrier）

设计原则（与 DSH 一致）:
1. 持久化优先：目标/清单/作业状态可跨重启恢复
2. 显式状态机：每个原语有明确状态与转换规则
3. 结构化结果：校验失败即失败，不让脏数据流入后续阶段
4. 幂等操作：重复调用不产生副作用
"""
