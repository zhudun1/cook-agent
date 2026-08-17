# Cook Agent

**生产级 LLM Agent 系统 · 以饮食管理为落地场景的完整工程实践**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.122-009688.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1.1-green.svg)](https://www.langchain.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/License-APACHE%202.0-blue.svg)](LICENSE)

---

## 📖 项目简介

**Cook Agent** 是一个融合 LLM、RAG、Agent、多模态与营养数据分析的**生产级智能饮食管理平台**。它不仅是菜谱库，更是一位能陪你做计划、做记录、看数据、给建议的"饮食管理助手"。

与普通 demo 不同，本项目把 **可靠性工程、可观测性、评测闭环、长期记忆、安全治理** 等生产级能力完整落地，是从"能对话的玩具"到"可上线系统"的完整工程实践。

- 🤖 **Agent 智能管家**：ReAct 推理 + 工具调用 + Subagent 专家体系 + MCP 动态扩展
- 🍽️ **个性化推荐**：结合用户画像与长期记忆，提供更贴合的菜品选择
- 🗓️ **饮食计划**：按周规划三餐与加餐，形成可执行的饮食节奏
- 🧾 **AI 记录**：文字/图片一键记录，自动估算热量与宏量营养
- 📊 **营养分析**：每日/每周统计与计划偏差分析，持续优化习惯
- 🛡️ **生产级安全**：成本熔断、危险操作人工审批、工具权限矩阵、注入纵深防御
- 🔭 **全链路可观测**：traceId 追踪 + Agent 轨迹回放 + 结构化日志
- 🧠 **长期记忆**：跨会话沉淀用户偏好/目标/限制，越用越懂你
- 📈 **评测闭环**：工具 SLO、任务完成率、RAG grounding truth、回归拦截

---

## 🏗️ 架构设计：生产级 Agent 七层模型

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
| **2 上下文层** | 长对话记忆退化、token 爆炸 | tiktoken 实时计数 + 滑动窗口（预算约束）+ 摘要压缩（语义保留）+ 跨会话长期记忆 |
| **3 工具层** | 工具不可靠、多租户串数据 | 独立超时 + 结构化错误（`error_code/retryable/suggestion` 供 Agent 自主恢复）+ 有状态工具会话级克隆隔离 |
| **4 编排层** | 长任务不可控、无法恢复 | 持久化 Goal（版本号乐观并发）+ 审批挂起（HITL）+ 后台作业（ID/状态/kill）+ 多阶段 workflow |
| **5 可观测层** | 非确定性系统无法调试 | traceId 全链路串联推理→工具→回答 + 轨迹 JSON 落盘 + 回放调试 |
| **6 评测层** | "看起来不错"≠可上线 | 工具 SLO + 黄金任务集完成率 + RAG reference 指标 + 基线回归拦截 |
| **7 安全层** | 上线即被攻击 | 成本熔断 / 危险操作人工审批 / 工具×用户权限矩阵 / 双层注入检测 |

---

## ✨ 核心功能

### 1. Agent 智能饮食管家（`app/agent/`）
- **ReAct 模式**：推理 + 行动循环，支持自主决策和工具调用
- **多模态支持**：图片自动上传持久化，识别食材/菜品
- **用户画像 + 长期记忆**：自动读取画像与跨会话记忆，提供个性化服务
- **Subagent 子代理**：内置/自定义专家，独立 system prompt 与工具集
- **内置工具**：饮食计划、饮食记录、营养分析、知识库检索、Web 搜索、图片生成、计算器、日期时间
- **MCP 协议支持**：用户自定义 MCP 服务器 + 鉴权头
- **上下文压缩**：token 感知的自动压缩，减少 Token 消耗
- **实时反馈**：SSE 事件流，实时展示工具调用过程与结果
- **执行追踪**：主 Agent 与 Subagent 轨迹分层展示

### 2. 生产级可靠性（`app/llm/resilience.py`、`app/agent/tools/base.py`）
- **指数退避重试 + 模型降级链**：LLM 调用失败自动重试（带抖动防雪崩），主模型全挂自动降级到低成本层级
- **工具独立超时 + 结构化错误**：每个工具独立超时配置；失败返回 `error_code / retryable / suggestion`，让 Agent **自主决策恢复路径**
- **JSON 自动修复兜底**：解析流水线修复尾逗号/单引号/未加引号 key/截断等常见坏 JSON
- **成本熔断**：单会话/单轮 token 预算，超预算自动 degrade/refuse/warn

### 3. 全链路可观测性（`app/telemetry/`）
- **traceId 全链路追踪**：HTTP 中间件注入/回传 `X-Trace-Id`，串联推理→工具调用→最终回答
- **结构化 JSON 日志**：JSON 行输出，自动携带 traceId/spanId，敏感字段递归脱敏
- **Agent 轨迹回放**：每次执行完整落盘 `data/trajectories/<trace_id>.json`，`load_trajectory / render_replay` 支持回放调试

### 4. P0 安全防护（`app/security/`）
- **人工介入审批（HITL）**：危险工具调用挂起 → SSE `approval_requested` → 前端审批卡片 → 批准后自动重执行
- **工具权限矩阵**：admin > 显式 deny > allow > 默认拒绝（敏感工具）> 默认允许，支持 glob 模式
- **注入纵深防御**：第一层用户输入检查 + **第二层工具返回内容检测**（指令覆盖/提示词泄露/越狱/`<|im_start|>`）
- **成本熔断**：防止单会话/单轮 token 失控

### 5. 评测闭环（`app/evaluation/`）
- **工具级 SLO**：每次调用成功率/延迟/P50/P95/错误码分布，`GET /evaluation/tool-metrics`
- **任务级完成率**：黄金任务集（`testsets/agent_tasks.jsonl`）跑完整 Agent 流程，LLM 判定 + 启发式回退
- **RAG grounding truth**：`context_precision / context_recall / answer_correctness` 离线批量评测 + 报告
- **回归拦截**：`--baseline` 对比完成率，下降超阈值标记 REGRESSION（可接入 CI）

### 6. 长期记忆与个性化（`app/memory/`）
- **跨会话记忆**：对话自动提取偏好/目标/限制/事实（LLM + 启发式回退）
- **回忆检索注入**：关键词 + 类型加权检索，限制类安全兜底
- **可视化维护**：前端个人中心查看/编辑/删除记忆

### 7. 饮食计划与记录
- 周视图管理早/午/晚餐与加餐，餐次自动汇总热量与宏量营养
- 一键标记已吃，计划自动转化为真实记录
- AI 解析文字/图片饮食描述，自动估算营养信息

### 8. 营养分析与目标追踪
- 每日/每周营养总览（热量、蛋白、脂肪、碳水）
- 计划 vs 实际偏差分析，识别饮食习惯波动
- 目标管理：卡路里/蛋白/脂肪/碳水目标

### 9. 智能对话式菜谱查询
- 自然语言理解需求，支持多轮对话
- 自动识别用户意图（查询、推荐、闲聊等）
- 流式响应，实时显示生成内容

### 10. 个性化知识库与混合检索
- 个人食谱上传自动索引，与全局食谱库融合查询
- **向量 + BM25 + Reranker** 混合检索，多级缓存提速

### 11. 多模态支持
- 图片识别：食材/菜品/饮食记录多场景
- 最多 4 张、单张 10MB，支持 OpenAI 兼容视觉模型

### 12. Prompt 版本管理与灰度（`app/prompts/`）
- 同一 prompt 多版本注册，**按用户稳定分流**（md5 哈希 → 权重区间）
- 内容哈希校验，支持 A/B 实验与逐步灰度

### 13. 统一存储后端（`app/storage/`）
- **数据面可插拔**：成本熔断计数 / 审批请求 / 事件流 / 工具 SLO 统一走 `StorageBackend` 抽象
- **MemoryBackend**（单机默认）+ **RedisBackend**（生产多实例共享状态），`storage:` 配置一键切换
- 多实例部署时成本熔断、审批决策、断点恢复、SLO 聚合跨实例一致
- 📖 详细说明：[docs/STORAGE_BACKEND.md](docs/STORAGE_BACKEND.md) · 本地 Redis 一键启动：`./scripts/start_redis.sh`

### 14. Harness 编排原语（`app/harness/`）
- **持久化 Goal**：版本号乐观并发、暂停/恢复/完成/阻塞状态机
- **Todo 清单**：整体替换语义 + 状态机
- **Schema 校验**：结构化结果 fail-fast
- **后台作业**：ID 追踪、输出收集、kill
- **Workflow**：phase 编排 / parallel barrier / pipeline

---

## 🚀 快速开始

### 前置要求
- **Python** >= 3.12
- **Node.js** >= 18

### 方式一：本地一键启动（SQLite，无需基础设施）

```bash
# 1. 安装依赖
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置（必填 JWT_SECRET_KEY；LLM key 见 .env.example）
cp .env.example .env

# 3. 启动后端（SQLite 自动建库；embedding/Milvus/Redis 缺失时优雅降级）
DATABASE_URL="sqlite+aiosqlite:///./cook_agent.db" uvicorn app.main:app --port 8000

# 4. 启动前端
cd frontend && npm install && npm run dev
```

- 前端：http://localhost:5173
- API 文档：http://localhost:8000/docs

### 方式二：Docker 完整部署（PostgreSQL + Redis + Milvus）

```bash
cd deployments && docker-compose up -d
cd .. && uvicorn app.main:app --port 8000   # 默认走 config.yml 的 PostgreSQL 配置
```

---

## ⚙️ 配置说明

### 1. 环境变量（`.env`）

```env
# ==================== LLM API 配置 ====================
LLM_API_KEY=your_main_api_key               # 主 API Key（normal 层）
FAST_LLM_API_KEY=your_fast_model_api_key    # 快速模型 Key（意图识别/查询改写）
VISION_API_KEY=your_vision_model_api_key    # 视觉模型 Key
RERANKER_API_KEY=your_reranker_api_key      # 重排序 Key

# ==================== 安全认证 ====================
JWT_SECRET_KEY=your_secure_jwt_secret_key

# ==================== 数据库 ====================
# 生产默认使用 config.yml 的 PostgreSQL；本地可用 SQLite 一键启动:
# DATABASE_URL=sqlite+aiosqlite:///./cook_agent.db
```

### 2. 主配置（`config.yml`，全部 YAML 外部化）

```yaml
llm:              # 分层模型（fast/normal/vision），model_names 负载均衡与降级
tokenizer:        # tiktoken 计数模型 + token 预算 + 压缩阈值
resilience:       # 重试（指数退避/抖动/可重试分类）+ 降级链 + 工具超时
telemetry:        # traceId / 结构化日志文件 / 轨迹落盘目录
security:         # P0: cost_guard / approval / permissions / injection_guard
memory:           # P2: 长期记忆（提取 LLM 层级 / 召回条数 / 上限）
evaluation:       # P1: testsets 目录 / grounding truth 指标 / 报告目录
prompt_registry:  # P3: Prompt 多版本灰度（按用户稳定分流）
storage:          # 统一存储后端：memory（单机）| redis（多实例）
```

---

## 🧪 测试与评测

```bash
# 单元测试（121 个：tokenizer / resilience / security / memory / evaluation / harness ...）
pytest tests/ -q

# RAG grounding truth 离线评测
python -m scripts.run_evaluation --dry-run          # 校验测试集
python -m scripts.run_evaluation                     # 评测 + 报告

# Agent 任务级端到端评测（完成率 + 回归拦截）
python -m scripts.run_agent_evaluation --dry-run
python -m scripts.run_agent_evaluation --save-baseline data/evaluation_reports/task_baseline.json
python -m scripts.run_agent_evaluation --baseline data/evaluation_reports/task_baseline.json --fail-on-regression
```

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
├── config/         # YAML 外部化配置（pydantic 模型 + loader）
└── api/v1/         # REST API（agent / evaluation / memory / auth ...）

frontend/src/
├── components/agent/       # Agent 对话 / 审批卡片 / 工具选择器
├── components/layout/      # 个人中心（含长期记忆面板）
├── components/evaluation/  # 评测面板（任务完成率）
├── hooks/useAgent.ts       # SSE 流式 / 断点恢复 / 审批决策
└── services/api/           # REST 客户端
```

---

## 🛠️ 技术栈

FastAPI · LangChain · SQLAlchemy · PostgreSQL/SQLite · Redis · Milvus · React 19 · Vite ·
tiktoken · tenacity · RAGAS · MCP · pydantic

## 📄 License

[Apache 2.0](LICENSE)
