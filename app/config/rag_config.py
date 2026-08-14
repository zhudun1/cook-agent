from __future__ import annotations


# app/config/rag_config.py
"""
RAG (Retrieval-Augmented Generation) configuration models.
These models define the structure for the config.yml file.

Design principles:
1. Database connection configs are defined in DatabaseConfig and passed in
2. RAG-specific configs only define their unique fields
3. LLM config uses the global LLMConfig (layered: fast/normal)
"""

from pydantic import BaseModel
from typing import List, Literal, Optional, Dict


# =============================================================================
# RAG-Specific Configurations
# =============================================================================

class PathsConfig(BaseModel):
    """Data paths configuration."""
    base_data_path: str = "data/HowToCook"


class VectorStoreConfig(BaseModel):
    """
    Vector store configuration.
    Note: Connection settings (host, port, credentials) are in DatabaseConfig.milvus
    """
    type: Literal["milvus"] = "milvus"
    collection_names: Dict[str, str] = {
        "recipes": "cook_hero_recipes",
        "personal": "cook_hero_personal_docs",
    }


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_name: str = "BAAI/bge-small-zh-v1.5"


class RetrievalConfig(BaseModel):
    """Retrieval pipeline configuration."""
    top_k: int = 9
    score_threshold: float = 0.2
    ranker_type: Literal["rrf", "weighted"] = "weighted"
    ranker_weights: List[float] = [0.8, 0.2]  # [dense, sparse]


class RerankerConfig(BaseModel):
    """
    Reranker configuration.
    Note: api_key defaults to global LLM API key if not set.
    """
    enabled: bool = True
    type: Literal["siliconflow"] = "siliconflow"
    model_name: str = "Qwen/Qwen3-Reranker-8B"
    base_url: Optional[str] = "https://api.siliconflow.cn/v1/rerank"
    api_key: Optional[str] = None  # Falls back to global LLM API key
    temperature: float = 0.0
    max_tokens: int = 8192
    score_threshold: float = 0.1


class CacheConfig(BaseModel):
    """
    Cache configuration for RAG retrieval results.
    
    Cache strategy:
    - L1: Exact match (Redis) - fast lookup for identical queries
    - L2: Semantic match (Milvus) - handles similar queries
    
    Note: 
    - Connection settings are in DatabaseConfig (redis/milvus)
    - Only caches Query -> Retrieved Documents, NOT LLM responses.
    """
    enabled: bool = True
    # TTL for both L1 and L2
    ttl: int = 3600  # 1 hour
    # L2 semantic cache
    l2_enabled: bool = True
    similarity_threshold: float = 0.92
    vector_collection: str = "cookhero_retrieval_cache"


class HowToCookConfig(BaseModel):
    """HowToCook recipe data source configuration (includes tips)."""
    path_suffix: str = "dishes"
    tips_path_suffix: str = "tips"  # Tips are now loaded together
    headers_to_split_on: List[List[str]] = [["#", "header_1"], ["##", "header_2"]]


class DataSourceConfig(BaseModel):
    """Data sources configuration."""
    howtocook: HowToCookConfig = HowToCookConfig()


# =============================================================================
# Main RAG Configuration
# =============================================================================

class RAGConfig(BaseModel):
    """
    Main RAG configuration model.
    
    Note: 
    - Database connections are in DatabaseConfig and passed separately
    - LLM configuration uses global LLMConfig (layered: fast/normal)
    """
    # Module configurations
    paths: PathsConfig = PathsConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    cache: CacheConfig = CacheConfig()
    data_source: DataSourceConfig = DataSourceConfig()


