# app/context/manager.py
"""
Context Manager for CookHero.
Responsible for building and assembling conversation context for LLM consumption.

Context structure:
1. System Message - The base system prompt
2. Compressed Summary - Summary of already compressed messages (if exists)
3. Uncompressed Messages - Original messages that haven't been compressed yet (history[compressed_count:])
4. Extra System Prompt - Additional context (e.g., RAG retrieved content)

Key invariant:
- Every historical message is either:
  a) Included in compressed_summary (semantically preserved), OR
  b) Present as an original message in the context
- No message should ever be "lost" (neither compressed nor in context)

This module does NOT call LLM directly. Compression is handled by ContextCompressor.
"""

from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


class ContextManager:
    """
    Builds and manages conversation context for LLM consumption.
    
    Responsibilities:
    - Combine system prompt, compressed history, and uncompressed messages
    - Format messages for LLM input
    - Provide unified interface for context building
    
    Context assembly uses compressed_count to slice history:
    - history[:compressed_count] -> already summarized in compressed_summary
    - history[compressed_count:] -> original messages to include in context
    """

    def __init__(
        self,
        system_prompt: str,
        history_text_max_len: int = 8096,
    ):
        """
        Initialize ContextManager.
        
        Args:
            system_prompt: The base system prompt for the LLM
            history_text_max_len: Max length for individual message content in text format
        """
        self.system_prompt = system_prompt
        self.history_text_max_len = history_text_max_len

    def build_llm_messages(
        self,
        history: List[Dict[str, str]],
        compressed_count: int = 0,
        compressed_summary: Optional[str] = None,
        extra_prompt: Optional[str] = None,
        user_profile: Optional[str] = None,
        user_instruction: Optional[str] = None,
    ) -> List[BaseMessage]:
        """
        Build LLM messages with proper context assembly.
        
        Context structure:
        1. User personalization context (profile + instruction) - high priority
        2. System Message (base prompt)
        3. System Message with compressed summary (if exists)
        4. Uncompressed Messages (history[compressed_count:])
        5. Extra system prompt (e.g., RAG context) - appended at end
        
        Args:
            history: Full conversation history as list of dicts with 'role' and 'content'
            compressed_count: Number of messages already compressed (from start of history)
            compressed_summary: Summary of compressed messages (from ContextCompressor)
            extra_system_prompt: Additional context (e.g., RAG retrieved content)
            user_profile: User's personal information and preferences
            user_instruction: User's custom instructions for the LLM
            
        Returns:
            List of LangChain BaseMessage objects ready for LLM
        """
        result: List[BaseMessage] = []
        
        # Add user personalization context (high priority, right after base system prompt)
        if user_profile or user_instruction:
            personalization_prompt = self._format_user_personalization(user_profile, user_instruction)
            result.append(SystemMessage(content=personalization_prompt))

        # Add base system prompt
        result.append(SystemMessage(content=self.system_prompt))
        
        # Add compressed summary if available
        if compressed_summary:
            compress_prompt = self._format_compressed_summary(compressed_summary)
            result.append(SystemMessage(content=compress_prompt))
        
        # Get uncompressed messages (messages not yet summarized)
        uncompressed_messages = history[compressed_count:]
        
        # 滑动窗口截断：token 预算内保留最近的未压缩消息，旧消息从旧到新丢弃
        # system/摘要消息永不丢弃；extra_prompt（如 RAG 上下文）也保留
        from app.config import settings
        from app.llm.tokenizer import get_token_counter

        budget = getattr(settings.tokenizer, "token_budget", 16000)
        windowed = get_token_counter().fit_in_budget(
            uncompressed_messages,
            budget,
            keep_first=0,
        )
        if len(windowed) < len(uncompressed_messages):
            logger = __import__("logging").getLogger(__name__)
            logger.info(
                "ContextManager sliding window truncated %d messages (budget=%d)",
                len(uncompressed_messages) - len(windowed),
                budget,
            )
        uncompressed_messages = windowed

        # Add uncompressed messages
        for msg in uncompressed_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                result.append(HumanMessage(content=content))
            else:
                result.append(AIMessage(content=content))
        
        # Add extra system prompt (e.g., RAG context) at the end
        if extra_prompt:
            result.append(AIMessage(content=extra_prompt))
        
        return result

    def build_history_text(
        self,
        history: List[Dict[str, str]],
        compressed_count: int = 0,
        compressed_summary: Optional[str] = None,
        empty_placeholder: str = "(无历史对话)",
    ) -> str:
        """
        Build formatted history text for intent detection, query rewriting, etc.
        
        Args:
            history: Full conversation history
            compressed_count: Number of messages already compressed
            compressed_summary: Optional compressed summary of older history
            limit: Optional limit for uncompressed messages (for intent detection, etc.)
            empty_placeholder: Text to return if no history
            
        Returns:
            Formatted string representation of conversation history
        """
        # Get uncompressed messages
        uncompressed = history[compressed_count:]
        
        if not uncompressed and not compressed_summary:
            return empty_placeholder
        
        parts: List[str] = []
        
        # Add compressed summary first if available
        if compressed_summary:
            parts.append(f"[历史对话摘要]\n{compressed_summary}\n")
        
        # Add uncompressed messages
        if uncompressed:
            parts.append("[最近对话]")
            for msg in uncompressed[:-1]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if len(content) > self.history_text_max_len:
                    content = content[:self.history_text_max_len] + "..."
                parts.append(f"{role}:\n {content}\n")
            # Last message is current user query
            last_msg = uncompressed[-1]
            role = last_msg.get("role", "")
            content = last_msg.get("content", "")
            parts.append(f"{role} (**当前问题**):\n {content}\n")
        
        return "\n".join(parts)

    def _format_compressed_summary(self, summary: str) -> str:
        """Format compressed summary as a system message."""
        return (
            "以下是之前对话的摘要,请在回答时参考这些背景信息:\n\n"
            f"{summary}"
        )
    
    def _format_user_personalization(
        self, 
        user_profile: Optional[str], 
        user_instruction: Optional[str]
    ) -> str:
        """Format user personalization context as a system message."""
        parts = []
        
        if user_profile:
            parts.append(f"## 用户个人信息 (User Profile)\n{user_profile}")
        
        if user_instruction:
            parts.append(f"## 用户自定义指令 (User Instruction)\n{user_instruction}")
        
        if not parts:
            return ""
        
        header = "以下是用户的个人信息和自定义指令,请在所有回答中遵循这些设定:\n"
        return header + "\n\n".join(parts)
