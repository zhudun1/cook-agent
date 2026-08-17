from __future__ import annotations


# app/config/config_loader.py
"""
Configuration loader for CookHero.
Loads from config.yml and merges with secrets from environment variables.

Environment variable loading:
- Uses load_dotenv() to load .env file into os.environ
- All sensitive params are read from os.getenv()
- Supports inheritance (e.g., RERANKER_API_KEY falls back to LLM_API_KEY)
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from app.config.database_config import (
    DatabaseConfig,
    MilvusConfig,
    PostgresConfig,
    RedisConfig,
)
from app.config.llm_config import LLMConfig
from app.config.rag_config import RAGConfig
from app.config.web_search_config import WebSearchConfig
from app.config.vision_config import VisionConfig, ImageGenerationConfig, ImageStorageConfig
from app.config.evaluation_config import EvaluationConfig, AlertThresholds
from app.config.mcp_config import MCPConfig, MCPServerConfig
from app.config.tokenizer_config import TokenizerConfig
from app.config.resilience_config import ResilienceConfig
from app.config.telemetry_config import TelemetryConfig
from app.config.security_config import SecurityConfig
from app.config.memory_config import MemoryConfig
from app.config.storage_config import StorageConfig


# Load .env file into environment variables at module import
load_dotenv()


def _load_config_data() -> Dict[str, Any]:
    """Load raw YAML config as a dict."""
    config_path = Path("config.yml")
    if not config_path.exists():
        raise FileNotFoundError("config.yml not found in the project root.")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_llm_config() -> LLMConfig:
    """
    Load global LLM provider configuration.

    Environment variables:
    - LLM_API_KEY: Normal LLM API key
    - FAST_LLM_API_KEY / LLM_FAST_API_KEY: Fast LLM API key (fallback to LLM_API_KEY)
    - VISION_API_KEY: Vision LLM API key (fallback to LLM_API_KEY)
    """
    config_data = _load_config_data()
    llm_root = config_data.get("llm", {}) or {}
    llm_data = dict(llm_root)

    # Inject API keys from environment (with inheritance for convenience)
    normal_api_key = os.getenv("LLM_API_KEY")
    fast_api_key = os.getenv("FAST_LLM_API_KEY") or os.getenv("LLM_FAST_API_KEY")
    vision_api_key = os.getenv("VISION_API_KEY") or os.getenv("LLM_API_KEY")

    normal_data = dict(llm_data.get("normal", {}) or {})
    fast_data = dict(llm_data.get("fast", {}) or {})
    vision_data = dict(llm_data.get("vision", {}) or {})

    if normal_api_key:
        normal_data["api_key"] = normal_api_key

    if fast_api_key:
        fast_data["api_key"] = fast_api_key

    if vision_api_key:
        vision_data["api_key"] = vision_api_key

    llm_data["normal"] = normal_data
    llm_data["fast"] = fast_data
    llm_data["vision"] = vision_data

    return LLMConfig.model_validate(llm_data)


def load_database_config() -> DatabaseConfig:
    """
    Load database configuration for PostgreSQL, Redis, and Milvus.
    
    Environment variables:
    - DATABASE_PASSWORD: PostgreSQL password
    - REDIS_PASSWORD: Redis password
    - MILVUS_USER: Milvus username
    - MILVUS_PASSWORD: Milvus password
    """
    config_data = _load_config_data()
    db_root = config_data.get("database", {}) or {}

    # PostgreSQL config from database.postgres
    pg_data = dict(db_root.get("postgres", {}) or {})
    db_password = os.getenv("DATABASE_PASSWORD")
    if db_password:
        pg_data["password"] = db_password
    postgres_config = PostgresConfig.model_validate(pg_data)

    # Redis config from database.redis
    redis_data = dict(db_root.get("redis", {}) or {})
    redis_password = os.getenv("REDIS_PASSWORD")
    if redis_password:
        redis_data["password"] = redis_password
    redis_config = RedisConfig.model_validate(redis_data)

    # Milvus config from database.milvus
    milvus_data = dict(db_root.get("milvus", {}) or {})
    milvus_user = os.getenv("MILVUS_USER")
    milvus_password = os.getenv("MILVUS_PASSWORD")
    if milvus_user:
        milvus_data["user"] = milvus_user
    if milvus_password:
        milvus_data["password"] = milvus_password
    milvus_config = MilvusConfig.model_validate(milvus_data)

    return DatabaseConfig(
        postgres=postgres_config,
        redis=redis_config,
        milvus=milvus_config,
    )


def load_rag_config(llm_config: Any | None = None) -> RAGConfig:
    """
    Load RAG configuration from YAML + environment variables.
    
    Environment variables:
    - RERANKER_API_KEY: Dedicated reranker API key (falls back to LLM_API_KEY)
    
    Args:
        llm_config: Global LLM config (normal profile) for API key fallback
    """
    config_data = _load_config_data()

    # Build RAG config data (excluding database sections)
    rag_data: Dict[str, Any] = {}

    # Copy RAG-specific sections
    for key in ["paths", "embedding", "retrieval", "data_source"]:
        if key in config_data:
            rag_data[key] = config_data[key]

    # Vector store config (without host/port, those are in DatabaseConfig)
    vs_data = config_data.get("vector_store", {}) or {}
    rag_data["vector_store"] = {
        "type": vs_data.get("type", "milvus"),
        "collection_names": vs_data.get("collection_names", {}),
    }

    # Reranker config with API key inheritance
    reranker_data = config_data.get("reranker", {}) or {}
    
    # API key priority: RERANKER_API_KEY > reranker.api_key in yaml > LLM_API_KEY
    reranker_api_key = os.getenv("RERANKER_API_KEY")
    if reranker_api_key:
        reranker_data["api_key"] = reranker_api_key
    
    rag_data["reranker"] = reranker_data

    # Cache config (without connection details, those are in DatabaseConfig)
    cache_data = config_data.get("cache", {}) or {}
    rag_data["cache"] = {
        "enabled": cache_data.get("enabled", True),
        "ttl": cache_data.get("ttl", 3600),
        "l2_enabled": cache_data.get("l2_enabled", True),
        "similarity_threshold": cache_data.get("similarity_threshold", 0.92),
        "vector_collection": cache_data.get("vector_collection", "cookhero_retrieval_cache"),
    }

    return RAGConfig.model_validate(rag_data)


def load_web_search_config() -> WebSearchConfig:
    """
    Load web search configuration from YAML + environment variables.
    
    Environment variables:
    - WEB_SEARCH_API_KEY: API key for web search provider
    """
    config_data = _load_config_data()
    ws_data = dict(config_data.get("web_search", {}) or {})
    
    # Load API key from environment
    api_key = os.getenv("WEB_SEARCH_API_KEY")
    if api_key:
        ws_data["api_key"] = api_key
    
    return WebSearchConfig.model_validate(ws_data)


def load_vision_config() -> VisionConfig:
    """
    Load vision configuration from YAML (domain keywords only).

    Note: Vision model configuration is now handled by LLMConfig.vision.
    This function only loads domain-specific settings like food_related_keywords.
    """
    config_data = _load_config_data()
    vision_data = dict(config_data.get("vision", {}) or {})

    # Only keep domain-related settings, model config is now in LLMConfig
    domain_data = {}
    if "food_related_keywords" in vision_data:
        domain_data["food_related_keywords"] = vision_data["food_related_keywords"]

    return VisionConfig.model_validate(domain_data)


def load_evaluation_config() -> EvaluationConfig:
    """
    Load RAG evaluation configuration from YAML.

    No environment variables needed - evaluation uses existing LLM config.
    """
    config_data = _load_config_data()
    eval_data = dict(config_data.get("evaluation", {}) or {})

    # Parse alert thresholds if present
    thresholds_data = eval_data.pop("alert_thresholds", None)
    if thresholds_data:
        eval_data["alert_thresholds"] = AlertThresholds(**thresholds_data)

    # Build config with defaults for missing fields
    # Note: context_precision and context_recall require 'reference' (ground truth)
    # which is not available in real-time evaluation scenarios.
    return EvaluationConfig(
        enabled=eval_data.get("enabled", True),
        async_mode=eval_data.get("async_mode", True),
        sample_rate=eval_data.get("sample_rate", 1.0),
        metrics=eval_data.get("metrics", [
            "faithfulness",
            "answer_relevancy",
        ]),
        llm_type=eval_data.get("llm_type", "fast"),
        timeout_seconds=eval_data.get("timeout_seconds", 60),
        alert_thresholds=eval_data.get("alert_thresholds", AlertThresholds()),
        testsets_dir=eval_data.get("testsets_dir", "testsets"),
        ground_truth_metrics=eval_data.get("ground_truth_metrics", [
            "context_precision",
            "context_recall",
            "answer_correctness",
        ]),
        report_dir=eval_data.get("report_dir", "data/evaluation_reports"),
        max_question_chars=eval_data.get("max_question_chars", 2000),
    )


def load_mcp_config() -> MCPConfig:
    """
    Load MCP configuration from YAML + environment variables.

    Environment variables:
    - AMAP_API_KEY: Amap (高德地图) API key for MCP integration
    """
    config_data = _load_config_data()
    mcp_data = dict(config_data.get("mcp", {}) or {})

    # Load AMAP API key from environment
    amap_api_key = os.getenv("AMAP_API_KEY")
    if amap_api_key:
        mcp_data["amap_api_key"] = amap_api_key

    # Parse amap server config if present
    amap_data = mcp_data.pop("amap", None)
    if amap_data:
        mcp_data["amap"] = MCPServerConfig(**amap_data)

    return MCPConfig.model_validate(mcp_data)


def load_image_generation_config() -> ImageGenerationConfig:
    """
    Load image generation configuration from YAML + environment variables.

    Environment variables:
    - IMAGE_GENERATION_API_KEY: OpenAI API key for DALL-E image generation
    """
    config_data = _load_config_data()
    ig_data = dict(config_data.get("image_generation", {}) or {})

    # Load API key from environment
    api_key = os.getenv("IMAGE_GENERATION_API_KEY")
    if api_key:
        ig_data["api_key"] = api_key

    return ImageGenerationConfig.model_validate(ig_data)


def load_image_storage_config() -> ImageStorageConfig:
    """
    Load image storage configuration from YAML + environment variables.

    Environment variables:
    - IMGBB_STORAGE_API_KEY: imgbb API key for image storage
    """
    config_data = _load_config_data()
    is_data = dict(config_data.get("image_storage", {}) or {})

    # Load API key from environment
    api_key = os.getenv("IMGBB_STORAGE_API_KEY")
    if api_key:
        is_data["api_key"] = api_key

    return ImageStorageConfig.model_validate(is_data)


def load_tokenizer_config() -> TokenizerConfig:
    """
    Load tokenizer/sliding-window configuration from YAML.

    Section: tokenizer
    """
    config_data = _load_config_data()
    data = dict(config_data.get("tokenizer", {}) or {})
    return TokenizerConfig.model_validate(data)


def load_resilience_config() -> ResilienceConfig:
    """
    Load LLM resilience configuration from YAML.

    Section: resilience (retry / fallback / tools)
    """
    config_data = _load_config_data()
    data = dict(config_data.get("resilience", {}) or {})
    return ResilienceConfig.model_validate(data)


def load_telemetry_config() -> TelemetryConfig:
    """
    Load telemetry configuration from YAML.

    Section: telemetry (trace / structured_log / trajectory)
    """
    config_data = _load_config_data()
    data = dict(config_data.get("telemetry", {}) or {})
    return TelemetryConfig.model_validate(data)


def load_security_config() -> SecurityConfig:
    """
    Load P0 security configuration from YAML.

    Section: security (cost_guard / approval / permissions / injection_guard)
    """
    config_data = _load_config_data()
    data = dict(config_data.get("security", {}) or {})
    return SecurityConfig.model_validate(data)


def load_memory_config() -> MemoryConfig:
    """
    Load P2 long-term memory configuration from YAML.

    Section: memory
    """
    config_data = _load_config_data()
    data = dict(config_data.get("memory", {}) or {})
    return MemoryConfig.model_validate(data)


def load_storage_config() -> StorageConfig:
    """
    Load storage backend configuration from YAML.

    Section: storage
    """
    config_data = _load_config_data()
    data = dict(config_data.get("storage", {}) or {})
    return StorageConfig.model_validate(data)
