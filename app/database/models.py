# app/database/models.py
"""
SQLAlchemy ORM models for CookHero.
Defines database schema for conversations, messages, user profiles,
long-term memory, and conversation summaries.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
)
from sqlalchemy import Uuid as _SqlUuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class UUID(_SqlUuid):
    """
    宽松 UUID 类型（跨 PostgreSQL / SQLite）。

    SQLAlchemy 的 Uuid 类型在绑定 str 输入时（如 `WHERE id = 'abc...'`）
    在 SQLite 下会报 "'str' object has no attribute 'hex'"（PostgreSQL 的
    asyncpg 驱动会自动转换所以无感）。本类型在绑定处理器中先把 str 转成
    uuid.UUID，兼容两种数据库与两种输入形式。
    """

    def bind_processor(self, dialect):
        impl = super().bind_processor(dialect)

        def process(value):
            if value is None:
                return None
            if isinstance(value, str):
                try:
                    value = uuid.UUID(value)
                except (ValueError, AttributeError):
                    return value  # 非 UUID 字符串原样透传，交由数据库报错
            if impl is not None:
                return impl(value)
            return value

        return process


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ==================== User Model ====================

class UserModel(Base):
    """ORM model for application users."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    occupation: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    bio: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # User personalization fields
    profile: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_instruction: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_users_username", "username", unique=True),
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "username": self.username,
            "occupation": self.occupation,
            "bio": self.bio,
            "profile": self.profile,
            "user_instruction": self.user_instruction,
            "created_at": self.created_at.isoformat(),
        }


class ConversationModel(Base):
    """ORM model for conversation sessions."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    # Optional user identifier for multi-user support
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    # Optional title/summary for the conversation
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # Metadata for extensibility (e.g., tags, preferences)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)
    
    # Compressed context summary for older messages
    compressed_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Number of messages included in the compressed summary
    compressed_message_count: Mapped[int] = mapped_column(default=0, nullable=False)

    # Relationship to messages
    messages: Mapped[List["MessageModel"]] = relationship(
        "MessageModel",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="MessageModel.created_at",
    )

    __table_args__ = (
        Index("ix_conversations_user_updated", "user_id", "updated_at"),
    )

    def to_dict(self) -> dict:
        """Serialize conversation to dict for API responses."""
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "user_id": self.user_id,
            "title": self.title,
            "message_count": len(self.messages) if self.messages else 0,
            "last_message_preview": (
                self.messages[-1].content[:80] if self.messages else None
            ),
            "compressed_summary": self.compressed_summary,
            "compressed_message_count": self.compressed_message_count,
        }


class MessageModel(Base):
    """ORM model for individual messages in a conversation."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False  # "user" or "assistant"
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    # Optional fields for RAG metadata
    sources: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    intent: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    thinking: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    
    # Duration metrics for response timing (in milliseconds)
    thinking_duration_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    answer_duration_ms: Mapped[Optional[int]] = mapped_column(nullable=True)

    # Relationship to conversation
    conversation: Mapped["ConversationModel"] = relationship(
        "ConversationModel", back_populates="messages"
    )

    __table_args__ = (
        Index("ix_messages_conv_created", "conversation_id", "created_at"),
    )

    def to_dict(self) -> dict:
        """Serialize message to dict for API responses."""
        return {
            "id": str(self.id),
            "role": self.role,
            "content": self.content,
            "timestamp": self.created_at.isoformat(),
            "sources": self.sources,
            "intent": self.intent,
            "thinking": self.thinking,
            "thinking_duration_ms": self.thinking_duration_ms,
            "answer_duration_ms": self.answer_duration_ms,
        }


class KnowledgeDocumentModel(Base):
    """
    Unified knowledge document model for all document types.
    Stores both public documents (HowToCook recipes/tips) and personal documents.
    For public documents, user_id is NULL.
    """

    __tablename__ = "knowledge_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # NULL for public documents (GLOBAL), actual user_id for personal documents
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=True
    )
    dish_name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    difficulty: Mapped[str] = mapped_column(String(50), nullable=False)
    # "recipes" for public recipes/tips, "personal" for user-owned documents
    data_source: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # "recipes", "tips", "personal" - more specific type
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # Original file path or identifier
    source: Mapped[str] = mapped_column(String(512), nullable=False)
    # Whether this is a dish index document
    is_dish_index: Mapped[bool] = mapped_column(default=False, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )

    __table_args__ = (
        Index("ix_knowledge_docs_user_category", "user_id", "category"),
        Index("ix_knowledge_docs_data_source", "data_source"),
        Index("ix_knowledge_docs_source_type", "source_type"),
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else "GLOBAL",
            "dish_name": self.dish_name,
            "category": self.category,
            "difficulty": self.difficulty,
            "data_source": self.data_source,
            "source_type": self.source_type,
            "source": self.source,
            "is_dish_index": self.is_dish_index,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_metadata(self) -> dict:
        """Convert to metadata dict for vector store."""
        return {
            "source": self.source,
            "parent_id": None,  # Will be set when creating chunks
            "dish_name": self.dish_name,
            "category": self.category,
            "difficulty": self.difficulty,
            "is_dish_index": self.is_dish_index,
            "data_source": self.data_source,
            "user_id": str(self.user_id) if self.user_id else "GLOBAL",
            "source_type": self.source_type,
        }


# ==================== RAG Evaluation Model ====================

class RAGEvaluationModel(Base):
    """
    ORM model for RAG evaluation results.
    Stores RAGAS evaluation metrics for each RAG-enabled response.
    """

    __tablename__ = "rag_evaluations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)

    # Original data for evaluation
    query: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    context: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)

    # Grounding truth（离线评测）
    reference_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reference_contexts: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # Retrieval quality metrics
    context_precision: Mapped[Optional[float]] = mapped_column(nullable=True)
    context_recall: Mapped[Optional[float]] = mapped_column(nullable=True)

    # Generation quality metrics
    faithfulness: Mapped[Optional[float]] = mapped_column(nullable=True)
    answer_relevancy: Mapped[Optional[float]] = mapped_column(nullable=True)
    answer_correctness: Mapped[Optional[float]] = mapped_column(nullable=True)

    # Evaluation metadata
    evaluation_status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False
    )  # pending/completed/failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evaluation_duration_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    evaluated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_rag_evaluations_conv_created", "conversation_id", "created_at"),
        Index("ix_rag_evaluations_status", "evaluation_status"),
        Index("ix_rag_evaluations_user_created", "user_id", "created_at"),
    )

    def to_dict(self) -> dict:
        """Serialize evaluation to dict for API responses."""
        return {
            "id": str(self.id),
            "message_id": str(self.message_id),
            "conversation_id": str(self.conversation_id),
            "user_id": self.user_id,
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "context": self.context,  # Full context for alerts
            "response": self.response,  # Full response for alerts
            "context_preview": self.context[:200] + "..." if len(self.context) > 200 else self.context,
            "response_preview": self.response[:200] + "..." if len(self.response) > 200 else self.response,
            "faithfulness": self.faithfulness,  # Direct field for alerts
            "answer_relevancy": self.answer_relevancy,  # Direct field for alerts
            "metrics": {
                "context_precision": self.context_precision,
                "context_recall": self.context_recall,
                "faithfulness": self.faithfulness,
                "answer_relevancy": self.answer_relevancy,
                "answer_correctness": self.answer_correctness,
            },
            "reference_answer": self.reference_answer,
            "evaluation_status": self.evaluation_status,
            "error_message": self.error_message,
            "evaluation_duration_ms": self.evaluation_duration_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
        }


# ==================== Long-term Memory Model ====================

class UserMemoryModel(Base):
    """
    ORM model for cross-session long-term memory entries.
    Stores user preferences / goals / restrictions extracted from conversations.
    """

    __tablename__ = "user_memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    memory_type: Mapped[str] = mapped_column(
        String(32), default="fact", nullable=False
    )  # preference | goal | restriction | fact
    importance: Mapped[float] = mapped_column(default=0.5, nullable=False)
    source: Mapped[str] = mapped_column(String(32), default="extracted", nullable=False)
    source_session_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    access_count: Mapped[int] = mapped_column(default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_user_memories_user_created", "user_id", "created_at"),
        Index("ix_user_memories_user_type", "user_id", "memory_type"),
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "source": self.source,
            "source_session_id": self.source_session_id,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed_at": self.last_accessed_at.isoformat()
            if self.last_accessed_at
            else None,
        }


# ==================== LLM Usage Log Model ====================

class LLMUsageLogModel(Base):
    """
    ORM model for LLM usage statistics.
    Records token usage, timing, and context for each LLM invocation.
    Does NOT store input/output content for privacy.
    """

    __tablename__ = "llm_usage_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Unique request identifier for tracing
    request_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    # Module that made the LLM call
    module_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    # User context (optional)
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    # Conversation context (optional)
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    # Model information
    model_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    # Tool information (if this LLM call involved tool usage)
    tool_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    # Token usage statistics
    input_tokens: Mapped[Optional[int]] = mapped_column(nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(nullable=True)
    # Performance metrics
    duration_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    __table_args__ = (
        # Time-series query optimization
        Index("ix_llm_usage_created_at_desc", created_at.desc()),
        # User + time query
        Index("ix_llm_usage_user_created", "user_id", "created_at"),
        # Module + time query
        Index("ix_llm_usage_module_created", "module_name", "created_at"),
        # Conversation query
        Index("ix_llm_usage_conversation", "conversation_id"),
        # Model + time query
        Index("ix_llm_usage_model_created", "model_name", "created_at"),
        # Tool + time query
        Index("ix_llm_usage_tool_created", "tool_name", "created_at"),
    )

    def to_dict(self) -> dict:
        """Serialize to dict for API responses."""
        return {
            "id": str(self.id),
            "request_id": self.request_id,
            "module_name": self.module_name,
            "user_id": self.user_id,
            "conversation_id": str(self.conversation_id) if self.conversation_id else None,
            "model_name": self.model_name,
            "tool_name": self.tool_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
