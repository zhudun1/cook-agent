# app/services/rag_service.py
"""
RAG Service - Orchestrates the RAG pipeline for knowledge retrieval and response generation.
All documents are stored in PostgreSQL and Milvus, no in-memory document storage.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.config import (
    DefaultRAGConfig,
    RAGConfig,
    settings,
)
from app.database.document_repository import document_repository
from app.rag.embeddings.embedding_factory import get_embedding_model
from app.rag.vector_stores.vector_store_factory import get_vector_store
from app.rag.pipeline.document_processor import document_processor
from app.rag.pipeline.retrieval import RetrievalOptimizationModule
from app.rag.pipeline.generation import GenerationIntegrationModule
from app.rag.pipeline.metadata_filter import MetadataFilterExtractor
from app.rag.rerankers.base import BaseReranker
from app.rag.cache import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of RAG retrieval operation."""

    original_query: str
    rewritten_query: str
    context: str
    documents: List[Document]
    sources: List[Dict]


class RAGService:
    """
    Orchestrates the entire RAG pipeline.

    Provides the main interface:
    - retrieve() - Query rewriting and retrieval, returns context
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: RAGConfig | None = None):
        if hasattr(self, "_initialized") and self._initialized:
            return

        logger.info("Initializing RAGService")
        self.config = config or DefaultRAGConfig
        self.db_config = settings.database
        self.embeddings = get_embedding_model(self.config)

        self.retrieval_modules: Dict[str, RetrievalOptimizationModule] = {}
        self.reranker: BaseReranker | None = None

        # Initialize vector store connections (no local file loading)
        self._init_vector_stores()

        self.generation_module = GenerationIntegrationModule(llm_type="fast")

        self.metadata_filter_extractor = MetadataFilterExtractor(llm_type="fast")

        if self.config.reranker.enabled:
            if self.config.reranker.type == "siliconflow":
                from app.rag.rerankers.siliconflow_reranker import SiliconFlowReranker

                self.reranker = SiliconFlowReranker(self.config.reranker)
            else:
                logger.warning(
                    f"Reranker type '{self.config.reranker.type}' not recognized. Reranking disabled."
                )

        # Initialize cache manager if enabled
        self.cache_manager: CacheManager | None = None
        if self.config.cache.enabled:
            self.cache_manager = CacheManager(
                redis_host=self.db_config.redis.host,
                redis_port=self.db_config.redis.port,
                redis_db=self.db_config.redis.db,
                redis_password=self.db_config.redis.password,
                ttl=self.config.cache.ttl,
                similarity_threshold=self.config.cache.similarity_threshold,
                embeddings=self.embeddings,
                l2_enabled=self.config.cache.l2_enabled,
                vector_host=self.db_config.milvus.host,
                vector_port=self.db_config.milvus.port,
                vector_collection=self.config.cache.vector_collection,
                vector_user=self.db_config.milvus.user,
                vector_password=self.db_config.milvus.password,
                vector_secure=self.db_config.milvus.secure,
            )
            logger.info("Cache manager enabled")
        else:
            logger.info("Caching disabled")

        # Markdown splitter for personal documents
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "header_1"), ("##", "header_2")],
            strip_headers=False,
        )

        self._initialized = True
        logger.info("RAGService initialized")

    def _init_vector_stores(self):
        """Initialize connections to Milvus vector stores."""
        logger.info("Initializing vector store connections")

        for name in ("recipes", "personal"):
            collection = self.config.vector_store.collection_names.get(name)
            if not collection:
                continue
            try:
                vector_store = get_vector_store(
                    milvus_config=self.db_config.milvus,
                    collection_name=collection,
                    embeddings=self.embeddings,
                    chunks=[],
                    force_rebuild=False,
                )
                retrieval_module = RetrievalOptimizationModule(
                    vectorstore=vector_store,
                    score_threshold=self.config.retrieval.score_threshold,
                    default_ranker_type=self.config.retrieval.ranker_type,
                    default_ranker_weights=self.config.retrieval.ranker_weights,
                )
                self.retrieval_modules[name] = retrieval_module
                logger.info("Connected to %s collection: %s", name, collection)
            except Exception as e:
                logger.warning("Failed to connect to %s collection: %s", name, e)

    # =========================================================================
    # Public API
    # =========================================================================

    async def retrieve(
        self,
        query: str,
        use_intelligent_ranker: bool = True,
        skip_rewrite: bool = False,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> RetrievalResult:
        """Perform query rewriting and retrieval only, without LLM generation."""
        if not self.retrieval_modules:
            raise RuntimeError("RAG Service is not properly initialized.")

        logger.info("retrieval start query='%s'", query[:80])

        # Query rewriting
        rewritten_query = (
            query
            if skip_rewrite
            else await self.generation_module.rewrite_query(
                query, user_id=user_id, conversation_id=conversation_id
            )
        )

        # Metadata filter extraction (from cached metadata, no DB access)
        filter_catalog = document_repository.get_metadata_for_filter(user_id)
        metadata_expression = (
            await self.metadata_filter_extractor.build_filter_expression(
                query, filter_catalog, user_id, conversation_id
            )
        )

        # Execute retrieval across all sources
        all_retrieved_docs = await self._execute_retrieval(
            rewritten_query,
            self.config.retrieval.top_k,
            use_intelligent_ranker,
            metadata_expression,
            user_id,
        )

        # Rerank if enabled
        reranked_docs = await self._rerank_if_needed(
            rewritten_query, all_retrieved_docs
        )

        # Post-process: fetch parent documents from database
        processed_docs = await document_processor.post_process_retrieval(reranked_docs)

        # Build context string
        context_parts = [doc.page_content for doc in processed_docs]
        context = "\n\n".join(context_parts) if context_parts else ""

        # Extract sources for frontend display
        sources = self._extract_sources(processed_docs)
        self._log_retrieval_summary("processed", processed_docs)

        return RetrievalResult(
            original_query=query,
            rewritten_query=rewritten_query,
            context=context,
            documents=processed_docs,
            sources=sources,
        )

    # =========================================================================
    # Retrieval execution
    # =========================================================================

    async def _execute_retrieval(
        self,
        rewritten_query: str,
        top_k: int,
        use_intelligent_ranker: bool,
        metadata_expression: Optional[str],
        user_id: Optional[str] = None,
    ) -> List[Document]:
        """Execute retrieval across all configured sources."""
        all_docs: List[Document] = []

        for source_name, module in self.retrieval_modules.items():
            docs = await self._retrieve_from_source(
                source_name=source_name,
                retrieval_module=module,
                rewritten_query=rewritten_query,
                top_k=top_k,
                use_intelligent_ranker=use_intelligent_ranker,
                metadata_expression=metadata_expression,
                user_id=user_id,
            )
            all_docs.extend(docs)

            self._log_retrieval_summary(f"source={source_name}", docs)

        logger.info("aggregated retrieved docs=%d", len(all_docs))
        return all_docs

    async def _retrieve_from_source(
        self,
        source_name: str,
        retrieval_module: RetrievalOptimizationModule,
        rewritten_query: str,
        top_k: int,
        use_intelligent_ranker: bool,
        metadata_expression: Optional[str],
        user_id: Optional[str] = None,
    ) -> List[Document]:
        """Retrieve documents from a single source with caching."""
        logger.info("retrieving source=%s", source_name)

        # Check cache
        if self.cache_manager:
            cached_docs = await self.cache_manager.get(
                source_name,
                rewritten_query,
                user_id if source_name == "personal" else None,
            )
            if cached_docs:
                logger.info(
                    "Using cached results for source '%s': %d documents",
                    source_name,
                    len(cached_docs),
                )
                for doc in cached_docs:
                    if "retrieval_score" not in doc.metadata:
                        doc.metadata["retrieval_score"] = 1.0
                    doc.metadata["data_source"] = source_name
                return cached_docs

        # Determine ranker configuration
        ranker_type = ranker_weights = None
        if use_intelligent_ranker:
            ranker_type, ranker_weights = retrieval_module.intelligent_ranker_selection(
                rewritten_query
            )

        # Execute search
        try:
            expr = self._build_filter_expr(metadata_expression, source_name, user_id)
            retrieved_docs, retrieved_scores = await retrieval_module.hybrid_search(
                rewritten_query,
                top_k=top_k,
                ranker_type=ranker_type,
                ranker_weights=ranker_weights,
                expr=expr,
            )
        except Exception as exc:
            logger.error(
                "Error during retrieval from source '%s': %s", source_name, exc
            )
            return []

        # Annotate documents with scores and source
        for doc, score in zip(retrieved_docs, retrieved_scores):
            existing_source = doc.metadata.get("data_source")
            if existing_source and existing_source != source_name:
                logger.warning(
                    "Data source mismatch (metadata=%s, expected=%s). Overriding.",
                    existing_source,
                    source_name,
                )
            doc.metadata["data_source"] = existing_source or source_name
            doc.metadata["retrieval_score"] = score

        # Deduplicate by parent_id, keep highest score
        unique_docs: Dict[Optional[str], Document] = {}
        for doc in retrieved_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id not in unique_docs:
                unique_docs[parent_id] = doc
            elif (
                doc.metadata["retrieval_score"]
                > unique_docs[parent_id].metadata["retrieval_score"]
            ):
                unique_docs[parent_id] = doc

        final_docs = sorted(
            unique_docs.values(),
            key=lambda d: d.metadata.get("retrieval_score", 0.0),
            reverse=True,
        )

        # Cache results
        if self.cache_manager:
            await self.cache_manager.set(
                source_name,
                rewritten_query,
                final_docs,
                user_id if source_name == "personal" else None,
            )

        return final_docs

    @staticmethod
    def _build_filter_expr(
        metadata_expression: Optional[str], source_name: str, user_id: Optional[str]
    ) -> Optional[str]:
        """Build filter expression with user scoping for personal documents."""
        if source_name != "personal" or not user_id:
            return metadata_expression

        user_filter = f'user_id == "{user_id}"'
        if metadata_expression:
            return f"({metadata_expression}) and ({user_filter})"
        return user_filter

    # =========================================================================
    # Personal document management
    # =========================================================================

    async def add_personal_document(
        self,
        *,
        user_id: str,
        document_id: str,
        dish_name: str,
        category: str,
        difficulty: str,
        data_source: str,
        content: str,
    ) -> None:
        """Add a user-owned document into the personal Milvus collection."""
        if not user_id:
            raise ValueError("user_id is required for personal documents")

        if "personal" not in self.retrieval_modules:
            raise RuntimeError("Personal retriever not initialized")

        metadata = {
            "source": f"personal::{user_id}",
            "parent_id": document_id,
            "dish_name": dish_name,
            "category": category,
            "difficulty": difficulty,
            "is_dish_index": False,
            "data_source": "personal",
            "user_id": user_id,
            "source_type": data_source,
        }

        # Create chunks
        chunks = document_processor.create_chunks(document_id, content, metadata)

        if chunks:
            retrieval_module = self.retrieval_modules["personal"]
            await asyncio.to_thread(retrieval_module.vectorstore.add_documents, chunks)
            logger.info("Indexed %d personal chunks for user %s", len(chunks), user_id)

    async def update_personal_document(
        self,
        *,
        user_id: str,
        document_id: str,
        dish_name: str,
        category: str,
        difficulty: str,
        data_source: str,
        content: str,
    ) -> None:
        """Update an existing personal document."""
        await self.delete_personal_document(user_id=user_id, document_id=document_id)
        await self.add_personal_document(
            user_id=user_id,
            document_id=document_id,
            dish_name=dish_name,
            category=category,
            difficulty=difficulty,
            data_source=data_source,
            content=content,
        )
        logger.info("Updated personal document id=%s user=%s", document_id, user_id)

    async def delete_personal_document(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> None:
        """Delete a personal document's chunks from the vector store."""
        if "personal" not in self.retrieval_modules:
            logger.warning("Personal retriever not initialized, skipping delete")
            return

        retrieval_module = self.retrieval_modules["personal"]
        expr = f'parent_id == "{document_id}" and user_id == "{user_id}"'

        try:
            await asyncio.to_thread(
                retrieval_module.vectorstore.col.delete,  # type: ignore
                expr,
            )
            logger.info(
                "Deleted personal document chunks id=%s user=%s", document_id, user_id
            )
        except Exception as e:
            logger.warning("Failed to delete personal document chunks: %s", e)

    # =========================================================================
    # Helper methods
    # =========================================================================

    async def _rerank_if_needed(self, rewritten_query: str, docs_for_rerank):
        if self.reranker and self.config.reranker.enabled:
            logger.info("reranking docs=%d", len(docs_for_rerank))
            return await self.reranker.rerank(rewritten_query, docs_for_rerank)
        return docs_for_rerank

    def _log_retrieval_summary(self, stage: str, docs: List[Document]) -> None:
        """Log a summary of retrieved documents."""
        summaries = []
        for doc in docs:
            meta = doc.metadata or {}
            summaries.append(
                {
                    "rerank_score": meta.get("rerank_score"),
                    "dish_name": meta.get("dish_name"),
                    "difficulty": meta.get("difficulty"),
                    "category": meta.get("category"),
                    "parent_id": meta.get("parent_id"),
                    "retrieval_score": meta.get("retrieval_score"),
                }
            )
        logger.info("retrieval %s docs=%d", stage, len(docs))
        for summary in summaries:
            logger.info("  %s", summary)

    def _extract_sources(self, documents: List[Document]) -> List[Dict]:
        """
        Extract source information from documents for frontend display.

        Returns unified format: {"type": "rag", "info": str, "url": optional str}
        The 'info' field combines dish_name/title for display.
        """
        sources = []
        seen = set()

        for doc in documents:
            metadata = doc.metadata or {}
            # Build info from title or category
            title = (
                metadata.get("dish_name")
                or metadata.get("title")
                or metadata.get("source_title")
            )
            info = title or metadata.get("category") or "CookHero 知识库"

            # Build unified source dict
            source_info: Dict[str, str] = {
                "type": "rag",  # Always "rag" for knowledge base sources
                "info": info,
            }

            # Add optional URL if available
            if metadata.get("url"):
                source_info["url"] = str(metadata["url"])

            # Deduplicate by (type, info)
            key = (source_info["type"], source_info["info"])
            if key not in seen:
                seen.add(key)
                sources.append(source_info)

        return sources


# Instantiate the singleton service
rag_service_instance = RAGService()
