# app/memory/__init__.py
"""
P2 长期记忆模块（跨会话记忆 + 画像沉淀）

组件:
- store:     记忆存取（去重/上限裁剪/关键词+类型加权回忆检索）
- extractor: 对话 -> 记忆提取（LLM 提取 + 启发式回退）
- manager:   编排（build_context 注入 / extract_and_store 后台沉淀）

接入点: AgentService.chat
- 构建上下文前: memory_manager.build_context(user_id, message) -> 注入 system prompt
- 对话结束后: asyncio.create_task(memory_manager.extract_and_store(...))
"""
