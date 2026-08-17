# app/rag/embeddings/embedding_factory.py
"""
Embedding 模型工厂。

优先使用配置的本地模型（HuggingFace）；本地模型不可用时
（依赖缺失 / 模型下载失败 / 推理后端故障）降级为不可用占位模型，
保证服务仍可启动（RAG 检索降级，Agent/对话主链路不受影响）。
"""
import logging
from typing import List

from langchain_core.embeddings import Embeddings
from app.config import RAGConfig

logger = logging.getLogger(__name__)


class UnavailableEmbeddings(Embeddings):
    """
    占位 embedding：底层模型不可用时的降级实现。
    任何调用记录 warning 并抛出明确异常（由调用方降级处理）。
    """

    def __init__(self, reason: str = "embedding model unavailable"):
        super().__init__()
        self.reason = reason

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.warning("Embedding unavailable: %s", self.reason)
        raise RuntimeError(f"Embedding model unavailable: {self.reason}")

    def embed_query(self, text: str) -> List[float]:
        logger.warning("Embedding unavailable: %s", self.reason)
        raise RuntimeError(f"Embedding model unavailable: {self.reason}")


def get_embedding_model(config: RAGConfig) -> Embeddings:
    """
    Factory function to create and return an embedding model based on the config.

    Args:
        config: The RAG configuration object.

    Returns:
        An instance of an embedding model（本地模型不可用时为降级占位）。
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Initializing local embedding model: %s", config.embedding.model_name)
        return HuggingFaceEmbeddings(
            model_name=config.embedding.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        logger.warning(
            "Local embedding model unavailable (%s); "
            "falling back to unavailable placeholder. RAG retrieval will be degraded.",
            e,
        )
        return UnavailableEmbeddings(reason=str(e))
