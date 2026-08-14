# app/prompts/__init__.py
"""
P3 Prompt 版本管理与灰度路由

- registry: PromptRegistry（多版本注册 + 用户稳定灰度分流 + 哈希校验）
- 接入点: AgentContextBuilder.build 解析 agent system_prompt

config.yml 的 prompt_registry 段可在启动时预注册版本化 prompt。
"""
