# app/context/compress.py
"""
Context Compressor for CookHero.
Responsible for compressing older conversation history into summaries.

Key responsibilities:
1. Determine when compression is needed
2. Generate summaries of older messages via LLM
3. Support incremental/rolling compression
4. Persist compressed summaries to database

Compression rule:
- When uncompressed messages >= COMPRESSION_THRESHOLD + RECENT_MESSAGES_LIMIT:
  - Compress the first COMPRESSION_THRESHOLD messages from uncompressed ones
  - This ensures uncompressed count stays in range [COMPRESSION_THRESHOLD, COMPRESSION_THRESHOLD + RECENT_MESSAGES_LIMIT)
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings, LLMType
from app.llm import LLMProvider, llm_context

from app.database.conversation_repository import ConversationRepository

logger = logging.getLogger(__name__)


COMPRESSION_SYSTEM_PROMPT = """
你是 CookHero 的「对话上下文摘要助手」，专门负责将较早的对话历史压缩为**简洁、结构清晰、信息完整的长期摘要**，用于后续烹饪推荐与饮食决策。你的目标不是复述对话，而是**提炼对后续推荐最有价值的信息**。

【必须重点保留的信息】
1. 用户的明确需求与目标  
   - 想做什么菜 / 想解决什么问题（如“快手晚餐”“减脂餐”“招待朋友”）
   - 使用场景（早餐 / 午餐 / 晚餐 / 聚会 / 健身后等）

2. 与烹饪和饮食强相关的事实信息  
   - 提到的**食材、菜品名称、菜系**
   - 饮食偏好（清淡 / 重口 / 川菜 / 粤菜等）
   - 饮食限制或禁忌（过敏、忌口、素食、减脂、高蛋白等）
   - 人数、预算、时间限制、厨具条件

3. 助手已经给出的**重要结论或建议**  
   - 已推荐过的菜品名称
   - 明确给出的做法思路、搭配建议、替代方案
   - 已被用户认可、采纳或明确否定的建议

【可以弱化或忽略的内容】
- 闲聊、寒暄、情绪性表达
- 重复出现、但不影响决策的信息
- 已被明确推翻或放弃的方案细节

【摘要表达要求】
1. 使用**第三人称客观描述**（如“用户希望…”，“系统已推荐…”）
2. 语言简洁、信息密集、偏事实性总结
3. 允许使用条目化或自然段落，但不要像聊天记录
4. 不要加入任何新的建议或推测，只能基于已有对话
5. 摘要长度根据对话内容灵活调整，确保信息完整即可

【增量摘要规则】
- 如果提供了「之前的对话摘要」，请将**新增对话内容与已有摘要进行融合**
- 输出应是一个**完整的、可直接使用的综合摘要**
- 不要提及“之前摘要 / 新摘要”等元信息

你的输出将作为后续对话的系统上下文，请确保**信息准确、稳定、可长期使用**。
"""


class ContextCompressor:
    """
    Compresses conversation history into summaries using LLM.

    Compression strategy:
    - Triggered when: uncompressed_count >= compression_threshold + recent_messages_limit
    - Compresses: first compression_threshold messages from uncompressed ones
    - Result: uncompressed_count reduced by compression_threshold
    - Invariant: every message is either compressed (in summary) or in context (original)
    """

    MODULE_NAME = "context_compression"

    def __init__(
        self,
        llm_type: LLMType | str = LLMType.NORMAL,
        compression_threshold: int = 6,
        recent_messages_limit: int = 10,
        max_messages_per_compression: int = 200,
        history_text_max_len: int = 8096,
        provider: LLMProvider | None = None,
    ):
        """
        Initialize ContextCompressor.

        Args:
            llm_type: Which LLM tier to use (fast/normal)
            compression_threshold: Number of messages to compress each time
            recent_messages_limit: Number of recent uncompressed messages to keep
            max_messages_per_compression: Max messages to compress in one call
        """
        self._llm_type = llm_type
        self.compression_threshold = compression_threshold
        self.recent_messages_limit = recent_messages_limit
        self.max_messages_per_compression = max_messages_per_compression
        self.history_text_max_len = history_text_max_len

        self._provider = provider or LLMProvider(settings.llm)
        # Use tracked invoker for usage statistics
        self._llm = self._provider.create_invoker(llm_type, temperature=0.3)

    async def maybe_compress(
        self,
        conversation_id: str,
        repository: ConversationRepository,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Check if compression is needed and perform it if so.

        This is the main entry point for compression logic.
        Handles: decision making, compression, and persistence.

        Compression rule:
        - When uncompressed_count >= compression_threshold + recent_messages_limit:
          - Compress first compression_threshold messages from uncompressed ones

        Args:
            conversation_id: The conversation ID
            repository: ConversationRepository for data access and persistence
            user_id: User ID for tracking (optional)

        Returns:
            True if compression was performed, False otherwise
        """
        try:
            # Get current state
            total_count = await repository.get_message_count(conversation_id)
            (
                existing_summary,
                compressed_count,
            ) = await repository.get_compressed_summary(conversation_id)

            # Calculate uncompressed count
            uncompressed_count = total_count - compressed_count

            # Check if compression is needed
            trigger_threshold = self.compression_threshold + self.recent_messages_limit
            count_trigger = uncompressed_count >= trigger_threshold

            # Token-aware trigger: uncompressed messages exceed token budget ratio
            # 解决长对话中单条消息过大但条数不多导致上下文爆炸的问题
            token_trigger = False
            try:
                from app.config import settings
                from app.llm.tokenizer import get_token_counter

                budget = settings.tokenizer.token_budget
                if budget > 0:
                    token_history = await repository.get_history(
                        conversation_id, limit=1000
                    ) or []
                    token_candidates = token_history[compressed_count:]
                    if token_candidates:
                        token_dicts = [
                            {"role": h["role"], "content": h["content"]}
                            for h in token_candidates
                        ]
                        tokens = get_token_counter().count_messages(token_dicts)
                        threshold = int(
                            budget * settings.tokenizer.compression_token_ratio
                        )
                        token_trigger = tokens > threshold
                        if token_trigger:
                            logger.info(
                                "Token-based compression trigger for %s: "
                                "uncompressed tokens=%d > threshold=%d",
                                conversation_id,
                                tokens,
                                threshold,
                            )
            except Exception as e:
                logger.debug("Token-based compression check failed: %s", e)

            if not count_trigger and not token_trigger:
                logger.debug(
                    "Compression not needed for %s: uncompressed=%d, threshold=%d",
                    conversation_id,
                    uncompressed_count,
                    trigger_threshold,
                )
                return False

            logger.info(
                "Triggering compression for %s: total=%d, compressed=%d, uncompressed=%d",
                conversation_id,
                total_count,
                compressed_count,
                uncompressed_count,
            )

            # Get full history for compression
            full_history = (
                await repository.get_history(conversation_id, limit=1000) or []
            )
            history_dicts = [
                {"role": h["role"], "content": h["content"]} for h in full_history
            ]

            # Get messages to compress (first COMPRESSION_THRESHOLD from uncompressed)
            uncompressed_messages = history_dicts[compressed_count:]
            messages_to_compress = uncompressed_messages[: self.compression_threshold]

            if not messages_to_compress:
                return False

            # Perform compression
            new_summary = await self._compress(
                messages_to_compress,
                existing_summary=existing_summary,
                user_id=user_id,
                conversation_id=conversation_id,
            )

            if new_summary:
                # Update compressed count
                new_compressed_count = compressed_count + len(messages_to_compress)

                # Persist
                await repository.update_compressed_summary(
                    conversation_id,
                    new_summary,
                    new_compressed_count,
                )

                logger.info(
                    "Compressed %d messages for %s, new compressed_count=%d",
                    len(messages_to_compress),
                    conversation_id,
                    new_compressed_count,
                )
                return True

            return False

        except Exception as e:
            logger.error(
                "Failed to compress context for %s: %s",
                conversation_id,
                e,
                exc_info=True,
            )
            return False

    async def _compress(
        self,
        messages: List[Dict[str, str]],
        existing_summary: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> str:
        """
        Compress messages into a summary (internal method).

        If existing_summary is provided, performs incremental compression
        by integrating new messages with the existing summary.

        Args:
            messages: List of message dicts to compress
            existing_summary: Optional existing summary to build upon
            user_id: User ID for tracking (optional)
            conversation_id: Conversation ID for tracking (optional)

        Returns:
            Compressed summary string
        """
        if not messages:
            return existing_summary or ""

        # Limit messages per compression to avoid context overflow
        messages_to_process = messages[-self.max_messages_per_compression :]

        # Format messages for compression
        messages_text = self._format_messages_for_compression(messages_to_process)

        # Build compression prompt
        if existing_summary:
            user_prompt = (
                f"【之前的对话摘要】\n{existing_summary}\n\n"
                f"【新增的对话内容】\n{messages_text}\n\n"
                "请将新增的对话内容与之前的摘要整合，生成一个更新后的综合摘要。"
            )
        else:
            user_prompt = (
                f"【对话内容】\n{messages_text}\n\n请为上述对话生成一个简洁的摘要。"
            )

        try:
            # Use llm_context for usage tracking
            with llm_context(self.MODULE_NAME, user_id, conversation_id):
                response = await self._llm.ainvoke(
                    [
                        SystemMessage(content=COMPRESSION_SYSTEM_PROMPT),
                        HumanMessage(content=user_prompt),
                    ]
                )

            # Extract content from response
            content = response.content
            if isinstance(content, str):
                summary = content.strip()
            else:
                # Handle case where content might be a list
                summary = str(content).strip()

            logger.info(
                "Compressed %d messages into summary (len=%d)",
                len(messages_to_process),
                len(summary),
            )
            return summary

        except Exception as e:
            logger.error("Failed to compress messages: %s", e, exc_info=True)
            # Return existing summary on failure, or empty string
            return existing_summary or ""

    def _format_messages_for_compression(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Format messages as text for compression prompt."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate very long messages
            if len(content) > self.history_text_max_len:
                content = content[: self.history_text_max_len] + "..."
            role_label = "用户" if role == "user" else "助手"
            parts.append(f"{role_label}: {content}")
        return "\n".join(parts)
