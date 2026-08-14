from __future__ import annotations


# app/config/config.py
"""
Unified configuration module for CookHero.
Provides a single entry point for all application configuration.

Design:
- Settings: Top-level configuration class containing global and module configs
- All configs loaded from config.yml + .env secrets
- Environment variables are loaded via load_dotenv in config_loader
"""

import os

from pydantic import BaseModel

from app.config.database_config import DatabaseConfig
from app.config.llm_config import LLMConfig
from app.config.rag_config import RAGConfig
from app.config.web_search_config import WebSearchConfig
from app.config.vision_config import VisionConfig, ImageGenerationConfig, ImageStorageConfig
from app.config.evaluation_config import EvaluationConfig
from app.config.mcp_config import MCPConfig
from app.config.tokenizer_config import TokenizerConfig
from app.config.resilience_config import ResilienceConfig
from app.config.telemetry_config import TelemetryConfig
from app.config.security_config import SecurityConfig
from app.config.memory_config import MemoryConfig
from app.config.config_loader import (
    load_database_config,
    load_llm_config,
    load_rag_config,
    load_web_search_config,
    load_vision_config,
    load_evaluation_config,
    load_mcp_config,
    load_image_generation_config,
    load_image_storage_config,
    load_tokenizer_config,
    load_resilience_config,
    load_telemetry_config,
    load_security_config,
    load_memory_config,
)


class Settings(BaseModel):
    """
    Top-level application settings.
    
    Contains:
    1. Global configuration (API prefix, project name, etc.)
    2. Global LLM provider configuration
    3. Database configurations (PostgreSQL, Redis, Milvus)
    4. Module-specific configurations (RAG, Web Search, etc.)
    """
    # ==========================================================================
    # Global Configuration
    # ==========================================================================
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "CookHero"
    DEBUG: bool = False

    # ==========================================================================
    # Auth / Security
    # Note: Environment variables are already loaded via load_dotenv in config_loader
    # SECURITY: JWT_SECRET_KEY must be set via environment variable in production
    # ==========================================================================
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    # Default to 60 minutes (was 10080 = 7 days, too long for security)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    # Refresh token expiration (7 days)
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    # ==========================================================================
    # Rate Limiting Configuration (Strict Mode)
    # ==========================================================================
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
    RATE_LIMIT_LOGIN_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_LOGIN_PER_MINUTE", "5"))
    RATE_LIMIT_CONVERSATION_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_CONVERSATION_PER_MINUTE", "30"))
    RATE_LIMIT_GLOBAL_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_GLOBAL_PER_MINUTE", "100"))

    # ==========================================================================
    # Security Configuration
    # ==========================================================================
    # Login security: lock account after N failed attempts
    LOGIN_MAX_FAILED_ATTEMPTS: int = int(os.getenv("LOGIN_MAX_FAILED_ATTEMPTS", "5"))
    LOGIN_LOCKOUT_MINUTES: int = int(os.getenv("LOGIN_LOCKOUT_MINUTES", "15"))
    # Input validation limits
    MAX_MESSAGE_LENGTH: int = int(os.getenv("MAX_MESSAGE_LENGTH", "10000"))
    MAX_IMAGE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "5"))
    # Prompt injection protection
    PROMPT_GUARD_ENABLED: bool = os.getenv("PROMPT_GUARD_ENABLED", "true").lower() == "true"
    
    # ==========================================================================
    # Module Configurations
    # ==========================================================================
    # Global LLM provider configuration (layered: fast/normal)
    llm: LLMConfig = load_llm_config()

    # Database configurations (PostgreSQL, Redis, Milvus)
    database: DatabaseConfig = load_database_config()

    # RAG configuration loaded from config.yml
    # Note: RAG reranker api_key may fall back to normal LLM api_key
    rag: RAGConfig = load_rag_config(llm.normal)
    
    # Web Search configuration loaded from config.yml
    web_search: WebSearchConfig = load_web_search_config()
    
    # Vision/Multimodal configuration loaded from config.yml
    vision: VisionConfig = load_vision_config()

    # RAG Evaluation configuration loaded from config.yml
    evaluation: EvaluationConfig = load_evaluation_config()

    # MCP configuration loaded from config.yml
    mcp: MCPConfig = load_mcp_config()

    # Image Generation configuration loaded from config.yml
    image_generation: ImageGenerationConfig = load_image_generation_config()

    # Image Storage configuration loaded from config.yml (imgbb)
    image_storage: ImageStorageConfig = load_image_storage_config()

    # Tokenizer & sliding window configuration loaded from config.yml
    tokenizer: TokenizerConfig = load_tokenizer_config()

    # LLM resilience configuration (retry / fallback / tool timeout) loaded from config.yml
    resilience: ResilienceConfig = load_resilience_config()

    # Telemetry configuration (trace / structured log / trajectory) loaded from config.yml
    telemetry: TelemetryConfig = load_telemetry_config()

    # P0 Security configuration (cost guard / approval / permissions / injection)
    security: SecurityConfig = load_security_config()

    # P2 Long-term memory configuration
    memory: MemoryConfig = load_memory_config()

    class Config:
        arbitrary_types_allowed = True


# Single global settings instance
settings = Settings()

# Convenience alias for backward compatibility
DefaultRAGConfig = settings.rag
