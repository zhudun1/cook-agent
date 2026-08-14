# Cook Agent

**生产级 LLM Agent 系统** · 从 demo 到可上线的完整工程实践

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.122-009688.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1.1-green.svg)](https://www.langchain.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/License-APACHE%202.0-blue.svg)](LICENSE)

---

## 📖 项目定位

一个**以饮食管理为落地场景的生产级 Agent 系统**。它不只是"能对话的 demo"，而是把
**可靠性、可观测性、评测闭环、长期记忆、安全治理** 等生产级能力完整落地的一整套工程实践——
适合作为 Agent 系统架构的参考实现。

核心能力：

- 🤖 **Agent 智能管家**：ReAct 推理 + 工具调用 + Subagent 专家 + MCP 动态扩展
- 🔍 **LLM + RAG 混合检索**：向量 + BM25 + Reranker + 多级缓存
- 🧠 **长期记忆**：跨会话偏好/目标/限制自动沉淀与回忆注入
- 🛡️ **P0 安全治理**：成本熔断 / 人工审批 / 工具权限矩阵 / 注入纵深防御
- 📊 **P1 评测闭环**：工具 SLO / 任务完成率 / RAG grounding truth / 回归拦截
- 🔭 **全链路可观测**：traceId / 结构化日志 / Agent 轨迹回放
- 🍽️ **饮食管理**：计划、记录、营养分析、个性化推荐

---

## 🏗️ 核心架构：生产级 Agent 七层模型

整个系统围绕七层架构构建，每一层都有明确的职责与可运行的代码：

```
┌─────────────────────────────────────────────────────────────┐
│ 7. 安全合规层  P0: 成本熔断 / HITL 人工审批 / 工具权限矩阵 /   │
│               注入纵深防御（用户输入 + 工具返回双层检测）        │
├─────────────────────────────────────────────────────────────┤
│ 6. 评测闭环层  P1: 工具级 SLO（成功率/P95）/ 黄金任务集端到端    │
│               完成率 / RAG grounding truth / 回归拦截          │
├─────────────────────────────────────────────────────────────┤
│ 5. 可观测性层  traceId 全链路 / 结构化 JSON 日志 /             │
│               Agent turn 轨迹持久化与回放调试                   │
├─────────────────────────────────────────────────────────────┤
│ 4. 编排层      ReAct / Subagent / Workflow（phase+pipeline） / │
│               持久化 Goal / 后台作业 / 审批挂起                 │
├─────────────────────────────────────────────────────────────┤
│ 3. 工具执行层  MCP 协议 / 权限检查 / 独立超时 / 结构化错误 /     │
│               会话级实例隔离（多租户）                          │
├─────────────────────────────────────────────────────────────┤
│ 2. 上下文层    tiktoken 实时计数 / 滑动窗口截断 / 摘要压缩 /     │
│               跨会话长期记忆回忆注入                           │
├─────────────────────────────────────────────────────────────┤
│ 1. 模型调用层  指数退避重试 / 模型降级链 / 成本熔断 /            │
│               Prompt 版本灰度 / 多模型路由 / token 记账          │
└─────────────────────────────────────────────────────────────┘
```

### 各层核心设计

| 层 | 核心问题 | 解决方案 |
|---|---|---|
| **1 模型层** | API 不稳定、成本失控 | 指数退避重试（防雪崩）+ 模型降级链（fast→normal）+ 会话/单轮 token 熔断 + Prompt 用户稳定灰度 |
| **2 上下文层** | 长对话记忆退化、token 爆炸 | tiktoken 实时计数 + 滑动窗口（预算约束）+ 摘要压缩（语义保留）+ 跨会话长期记忆（偏好/目标/限制） |
| **3 工具层** | 工具不可靠、多租户串数据 | 独立超时 + 结构化错误（`error_code/retryable/suggestion` 供 Agent 自主恢复）+ 有状态工具会话级克隆隔离 |
| **4 编排层** | 长任务不可控、无法恢复 | 持久化 Goal（版本号乐观并发）+ 审批挂起（HITL）+ 后台作业（ID/状态/kill）+ 多阶段 workflow |
| **5 可观测层** | 非确定性系统无法调试 | traceId 全链路串联推理→工具→回答 + 轨迹 JSON 落盘 + `load_trajectory/render_replay` 回放 |
| **6 评测层** | "看起来不错"≠可上线 | 工具 SLO（实时采集）+ 黄金任务集完成率（LLM/启发式判定）+ RAG reference 指标 + 基线回归拦截 |
| **7 安全层** | 上线即被攻击 | 成本熔断 / 危险操作人工审批 / 工具×用户权限矩阵 / 双层注入检测（输入 + 工具返回内容） |

---

## ✨ 关键特性

### 🛡️ P0 安全治理（`app/security/`）
- **成本熔断** `cost_guard.py`：单会话/单轮 token 预算，超预算 degrade/refuse/warn
- **人工介入审批** `approval.py`：危险工具调用挂起 → SSE `approval_requested` → 前端审批卡片 → 批准后自动重执行
- **工具权限矩阵** `permissions.py`：admin > 显式 deny > allow > 默认拒绝（敏感工具）> 默认允许，支持 glob
- **注入纵深防御** `injection.py`：第二层检测工具返回内容（指令覆盖/提示词泄露/越狱/`<|im_start|>`）

### 📊 P1 评测闭环（`app/evaluation/`）
- **工具级 SLO** `tool_metrics.py`：每次调用成功率/延迟/P50/P95/错误码分布，`GET /evaluation/tool-metrics`
- **任务级完成率** `task_runner.py`：黄金任务集（`testsets/agent_tasks.jsonl`）跑完整 Agent 流程，LLM 判定 + 启发式回退
- **RAG grounding truth** `runner.py`：`context_precision / context_recall / answer_correctness` 离线批量评测 + 报告
- **回归拦截**：`--baseline` 对比完成率，下降超阈值标记 REGRESSION（可接入 CI）

### 🧠 P2 长期记忆（`app/memory/`）
- 对话自动提取偏好/目标/限制/事实（LLM + 启发式回退）
- 关键词 + 类型加权回忆检索；**限制类安全兜底**（泛化查询也召回过敏/忌口）
- 前端个人中心可视化查看/编辑/删除

### 🏗️ P3 架构演进（`app/prompts/` `app/agent/event_stream.py`）
- **多租户隔离**：`BaseTool.is_stateful` + `clone_for_session()`，有状态工具（MCP）会话级独立实例
- **SSE 断点恢复**：事件流按会话缓存（带 seq），断线重连增量回放
- **Prompt 版本灰度**：多版本注册 + 用户稳定分流（md5 哈希）+ 内容哈希校验

### ⚙️ 基础能力
- **Token 上下文工程**：tiktoken 计数 + 滑动窗口 + token 感知压缩
- **LLM 韧性**：指数退避重试 + 模型降级链 + 流式首块重试
- **可观测**：traceId 中间件 + 结构化 JSON 日志（脱敏）+ 轨迹回放
- **Harness 编排原语**：持久化 Goal / Todo 清单 / Schema 校验 / 后台作业 / Workflow
- **MCP 协议**：远程工具标准化接入 + 鉴权头
- **多模态**：图片识别与饮食记录联动

---

## 📁 模块地图

```
app/
├── agent/          # Agent 核心（ReAct 循环 / 工具注册表 / 会话级隔离 / 审批挂起 / 事件流）
├── security/       # P0 安全（成本熔断 / 审批 / 权限矩阵 / 注入检测）
├── evaluation/     # P1 评测（工具 SLO / 任务完成率 / RAG grounding truth）
├── memory/         # P2 长期记忆（提取 / 存储 / 回忆检索）
├── prompts/        # P3 Prompt 版本管理与灰度
├── harness/        # 编排原语（Goal / Todo / Schema / Jobs / Workflow）
├── telemetry/      # 可观测（traceId / 结构化日志 / 轨迹回放）
├── llm/            # 模型层（韧性重试降级 / tokenizer / 滑动窗口）
├── context/        # 上下文组装与压缩
├── rag/            # 检索 / 重排 / 缓存 / 嵌入
├── tools/          # Web 搜索（Tavily）
├── config/         # YAML 外部化配置（pydantic 模型 + loader）
└── api/v1/         # REST API（agent / evaluation / memory / auth ...）

frontend/src/
├── components/agent/    # Agent 对话 / 审批卡片 / 工具选择器
├── components/layout/   # 个人中心（含长期记忆面板）
├── components/evaluation/ # 评测面板（任务完成率）
├── hooks/useAgent.ts    # SSE 流式 / 断点恢复 / 审批决策
└── services/api/        # REST 客户端
```

---

## 🚀 快速开始

### 方式一：本地一键启动（SQLite，无需基础设施）

```bash
# 1. 安装依赖
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置（必填 JWT_SECRET_KEY；LLM key 见 .env.example）
cp .env.example .env

# 3. 启动后端（SQLite 自动建库，embedding/Milvus/Redis 缺失时优雅降级）
DATABASE_URL="sqlite+aiosqlite:///./cook_agent.db" uvicorn app.main:app --port 8000

# 4. 启动前端
cd frontend && npm install && npm run dev
```

- 前端：http://localhost:5173 · API 文档：http://localhost:8000/docs

### 方式二：Docker 完整部署（PostgreSQL + Redis + Milvus）

```bash
cd deployments && docker-compose up -d
cd .. && DATABASE_URL="" uvicorn app.main:app --port 8000   # 默认走 config.yml 的 PG 配置
```

---

## ⚙️ 配置（config.yml，全部 YAML 外部化）

```yaml
llm:          # 分层模型（fast/normal/vision）+ model_names 负载均衡与降级
tokenizer:    # tiktoken 计数 + token 预算 + 压缩阈值
resilience:   # 重试（指数退避/抖动/可重试分类）+ 降级链 + 工具超时
telemetry:    # traceId / 结构化日志 / 轨迹落盘
security:     # P0: cost_guard / approval / permissions / injection_guard
memory:       # P2: 长期记忆（提取 / 召回 / 上限）
evaluation:   # P1: testsets / grounding truth 指标 / 报告目录
prompt_registry:  # P3: Prompt 多版本灰度（按用户稳定分流）
```

---

## 🧪 评测体系

```bash
# RAG grounding truth 离线评测
python -m scripts.run_evaluation --dry-run          # 校验测试集
python -m scripts.run_evaluation                     # 评测 + 报告

# Agent 任务级端到端评测（完成率 + 回归拦截）
python -m scripts.run_agent_evaluation --dry-run
python -m scripts.run_agent_evaluation --save-baseline data/evaluation_reports/task_baseline.json
python -m scripts.run_agent_evaluation --baseline data/evaluation_reports/task_baseline.json --fail-on-regression

# 单元测试
pytest tests/ -q          # 121 个测试（tokenizer / resilience / security / memory / evaluation / harness ...）
```

---

## 🛠️ 技术栈

FastAPI · LangChain · SQLAlchemy · PostgreSQL/SQLite · Redis · Milvus · React 19 · Vite ·
tiktoken · tenacity · RAGAS · MCP · pydantic

## 📄 License

[Apache 2.0](LICENSE)
