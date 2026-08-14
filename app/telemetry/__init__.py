# app/telemetry/__init__.py
"""
可观测性模块：traceId 全链路追踪 + 结构化 JSON 日志 + Agent 轨迹回放。

子模块:
- trace: traceId/span 上下文（contextvars 传递，无需改函数签名）
- logger: 结构化 JSON 行日志（含 traceId、脱敏）
- trajectory: Agent turn 轨迹 JSON 持久化与回放调试
"""
