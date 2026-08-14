# app/database/document_repository.py
"""
Repository for knowledge document CRUD operations.
Provides async database access for documents stored in PostgreSQL.
"""

import logging
import uuid
from typing import Dict, List, Optional

from sqlalchemy import delete, select, func
from langchain_core.documents import Document

from app.database.models import KnowledgeDocumentModel
from app.database.session import get_session_context

logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    Repository for managing knowledge documents in PostgreSQL.
    Provides methods to:
    - Get parent documents by their IDs (for post_process_retrieval)
    - CRUD operations for documents
    - Batch operations for ingestion
    - Cached metadata options for efficient retrieval
    """
    
    # Class-level cache for metadata options (loaded once at startup)
    _global_cache: Dict[str, List[str]] = {}  # Global recipes metadata
    _user_cache: Dict[str, Dict[str, List[str]]] = {}  # user_id -> personal metadata
    _cache_initialized: bool = False

    @classmethod
    async def init_all_metadata_cache(cls) -> None:
        """
        Initialize all metadata caches from database. Call once at startup.
        Loads both global recipes and all users' personal documents metadata.
        """
        if cls._cache_initialized:
            return
        
        async with get_session_context() as session:
            # 方言感知：PostgreSQL 用 array_agg；SQLite 用 group_concat
            dialect = session.bind.dialect.name
            use_sqlite = dialect == "sqlite"

            # 1. Load global metadata (user_id is NULL)
            cls._global_cache = {"dish_name": [], "category": [], "difficulty": []}
            for field_name in cls._global_cache.keys():
                field = getattr(KnowledgeDocumentModel, field_name)
                stmt = select(func.distinct(field)).where(
                    KnowledgeDocumentModel.user_id.is_(None)
                )
                rows = (await session.execute(stmt)).scalars().all()
                cls._global_cache[field_name] = sorted([v for v in rows if v])
            
            # 2. Load all user metadata (grouped by user_id)
            cls._user_cache = {}
            for field_name in ("dish_name", "category", "difficulty"):
                field = getattr(KnowledgeDocumentModel, field_name)
                if use_sqlite:
                    agg = func.group_concat(func.distinct(field))
                else:
                    agg = func.array_agg(func.distinct(field))
                stmt = select(
                    KnowledgeDocumentModel.user_id,
                    agg,
                ).where(
                    KnowledgeDocumentModel.user_id.is_not(None)
                ).group_by(KnowledgeDocumentModel.user_id)
                
                rows = (await session.execute(stmt)).all()
                for user_uuid, values in rows:
                    user_id = str(user_uuid)
                    if user_id not in cls._user_cache:
                        cls._user_cache[user_id] = {
                            "dish_name": [],
                            "category": [],
                            "difficulty": [],
                        }
                    if use_sqlite:
                        # group_concat 返回逗号分隔字符串
                        values = [v.strip() for v in (values or "").split(",") if v.strip()]
                    cls._user_cache[user_id][field_name] = sorted([v for v in (values or []) if v])
        
        cls._cache_initialized = True
        logger.info(
            "Metadata cache initialized: global(%d dishes, %d categories, %d difficulties), %d users",
            len(cls._global_cache.get("dish_name", [])),
            len(cls._global_cache.get("category", [])),
            len(cls._global_cache.get("difficulty", [])),
            len(cls._user_cache),
        )

    @classmethod
    def _update_cache_on_create(
        cls,
        dish_name: str,
        category: str,
        difficulty: str,
        user_id: Optional[str] = None,
    ) -> None:
        """Incrementally update metadata cache when a document is created."""
        if user_id:
            # Update user-specific cache
            if user_id not in cls._user_cache:
                cls._user_cache[user_id] = {
                    "dish_name": [],
                    "category": [],
                    "difficulty": [],
                }
            user_cache = cls._user_cache[user_id]
            if dish_name and dish_name not in user_cache["dish_name"]:
                user_cache["dish_name"] = sorted(user_cache["dish_name"] + [dish_name])
            if category and category not in user_cache["category"]:
                user_cache["category"] = sorted(user_cache["category"] + [category])
            if difficulty and difficulty not in user_cache["difficulty"]:
                user_cache["difficulty"] = sorted(user_cache["difficulty"] + [difficulty])
        else:
            raise NotImplementedError(
                "Global metadata cache update on create not implemented."
            )

    @classmethod
    async def _update_cache_on_delete(
        cls,
        user_id: Optional[str] = None,
    ) -> None:
        """Update metadata cache when a document is deleted by reloading the affected cache."""
        if user_id:
            # Reload user cache
            async with get_session_context() as session:
                user_uuid = uuid.UUID(user_id)
                user_cache = {"dish_name": [], "category": [], "difficulty": []}
                for field_name in user_cache.keys():
                    field = getattr(KnowledgeDocumentModel, field_name)
                    stmt = select(func.distinct(field)).where(
                        KnowledgeDocumentModel.user_id == user_uuid
                    )
                    rows = (await session.execute(stmt)).scalars().all()
                    user_cache[field_name] = sorted([v for v in rows if v])
                print(user_uuid, user_cache)
                cls._user_cache[user_id] = user_cache
        else:
            raise NotImplementedError(
                "Global metadata cache reload on delete not implemented."
            )

    @classmethod
    def get_metadata_options(cls, user_id: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get merged metadata options (no database access).
        Merges global cache with user-specific cache if user_id is provided.
        """
        if not user_id:
            return cls._global_cache.copy()
        
        # Merge global and user metadata
        user_meta = cls._user_cache.get(user_id, {})
        merged = {}
        for key in ("dish_name", "category", "difficulty"):
            global_values = set(cls._global_cache.get(key, []))
            user_values = set(user_meta.get(key, []))
            merged[key] = sorted(global_values | user_values)
        
        return merged

    @classmethod
    def get_metadata_for_filter(cls, user_id: Optional[str] = None) -> Dict[str, Dict[str, List[str]]]:
        """
        Get metadata catalog for filter extraction (no database access).
        Returns data grouped by source: 'Global Recipes' and 'Personal Documents'.
        """
        result: Dict[str, Dict[str, List[str]]] = {}
        
        # Always include global recipes
        if cls._global_cache:
            result["Global Recipes"] = cls._global_cache.copy()
        
        # Include user's personal documents if provided
        if user_id:
            user_meta = cls._user_cache.get(user_id, {})
            if user_meta and any(user_meta.values()):
                result["Personal Documents"] = user_meta.copy()
        
        return result

    @staticmethod
    async def get_by_id(doc_id: str) -> Optional[KnowledgeDocumentModel]:
        """Get a single document by its ID."""
        try:
            doc_uuid = uuid.UUID(doc_id)
        except (TypeError, ValueError):
            return None

        async with get_session_context() as session:
            stmt = select(KnowledgeDocumentModel).where(
                KnowledgeDocumentModel.id == doc_uuid
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    @staticmethod
    async def get_by_id_for_user(doc_id: str, user_id: str) -> Optional[KnowledgeDocumentModel]:
        """Get a single document by ID, scoped to a specific user."""
        try:
            doc_uuid = uuid.UUID(doc_id)
            user_uuid = uuid.UUID(user_id)
        except (TypeError, ValueError):
            return None

        async with get_session_context() as session:
            stmt = select(KnowledgeDocumentModel).where(
                KnowledgeDocumentModel.id == doc_uuid,
                KnowledgeDocumentModel.user_id == user_uuid,
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    @staticmethod
    async def get_by_ids(doc_ids: List[str]) -> Dict[str, KnowledgeDocumentModel]:
        """
        Get multiple documents by their IDs.
        Returns a dict mapping doc_id -> document for easy lookup.
        """
        if not doc_ids:
            return {}

        try:
            doc_uuids = [uuid.UUID(doc_id) for doc_id in doc_ids]
        except (TypeError, ValueError):
            logger.warning("Invalid document ID format in get_by_ids")
            return {}

        async with get_session_context() as session:
            stmt = select(KnowledgeDocumentModel).where(
                KnowledgeDocumentModel.id.in_(doc_uuids)
            )
            result = await session.execute(stmt)
            docs = result.scalars().all()
            return {str(doc.id): doc for doc in docs}

    @staticmethod
    async def get_parent_documents(parent_ids: List[str]) -> Dict[str, Document]:
        """
        Get parent documents by their IDs and convert to LangChain Documents.
        Used by post_process_retrieval to restore full document content.
        """
        if not parent_ids:
            return {}

        doc_models = await DocumentRepository.get_by_ids(parent_ids)
        
        result = {}
        for doc_id, model in doc_models.items():
            result[doc_id] = Document(
                id=doc_id,
                page_content=model.content,
                metadata=model.to_metadata(),
            )
        return result

    @staticmethod
    async def create(
        *,
        doc_id: Optional[str] = None,
        user_id: Optional[str] = None,
        dish_name: str,
        category: str,
        difficulty: str,
        data_source: str,
        source_type: str,
        source: str,
        is_dish_index: bool = False,
        content: str,
    ) -> KnowledgeDocumentModel:
        """Create a new document."""
        async with get_session_context() as session:
            doc = KnowledgeDocumentModel(
                id=uuid.UUID(doc_id) if doc_id else uuid.uuid4(),
                user_id=uuid.UUID(user_id) if user_id else None,
                dish_name=dish_name.strip(),
                category=category.strip(),
                difficulty=difficulty.strip(),
                data_source=data_source,
                source_type=source_type,
                source=source,
                is_dish_index=is_dish_index,
                content=content,
            )
            session.add(doc)
            await session.flush()
            logger.info("Created document id=%s dish_name=%s", doc.id, dish_name)
            
            # Update metadata cache
            DocumentRepository._update_cache_on_create(
                dish_name=dish_name.strip(),
                category=category.strip(),
                difficulty=difficulty.strip(),
                user_id=user_id,
            )
            
            return doc

    @staticmethod
    async def create_batch(documents: List[Dict]) -> List[KnowledgeDocumentModel]:
        """
        Batch create multiple documents.
        Each dict should contain: doc_id, user_id (optional), dish_name, category,
        difficulty, data_source, source_type, source, is_dish_index, content
        """
        if not documents:
            return []

        async with get_session_context() as session:
            models = []
            for doc_data in documents:
                doc = KnowledgeDocumentModel(
                    id=uuid.UUID(doc_data["doc_id"]) if doc_data.get("doc_id") else uuid.uuid4(),
                    user_id=uuid.UUID(doc_data["user_id"]) if doc_data.get("user_id") else None,
                    dish_name=doc_data["dish_name"].strip(),
                    category=doc_data["category"].strip(),
                    difficulty=doc_data["difficulty"].strip(),
                    data_source=doc_data["data_source"],
                    source_type=doc_data["source_type"],
                    source=doc_data["source"],
                    is_dish_index=doc_data.get("is_dish_index", False),
                    content=doc_data["content"],
                )
                session.add(doc)
                models.append(doc)
            
            await session.flush()
            logger.info("Batch created %d documents", len(models))
            return models

    @staticmethod
    async def update(
        doc_id: str,
        user_id: Optional[str] = None,
        **updates
    ) -> Optional[KnowledgeDocumentModel]:
        """Update an existing document. Only updates provided fields."""
        try:
            doc_uuid = uuid.UUID(doc_id)
            user_uuid = uuid.UUID(user_id) if user_id else None
        except (TypeError, ValueError):
            return None

        async with get_session_context() as session:
            # Build query - for personal docs, also filter by user_id
            stmt = select(KnowledgeDocumentModel).where(
                KnowledgeDocumentModel.id == doc_uuid
            )
            if user_uuid:
                stmt = stmt.where(KnowledgeDocumentModel.user_id == user_uuid)
            
            result = await session.execute(stmt)
            doc = result.scalar_one_or_none()
            
            if not doc:
                return None
            
            # Update fields
            for key, value in updates.items():
                if hasattr(doc, key) and value is not None:
                    if isinstance(value, str):
                        value = value.strip()
                    setattr(doc, key, value)
            
            await session.flush()
            logger.info("Updated document id=%s", doc_id)
            return doc

    @staticmethod
    async def delete(doc_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a document by ID. For personal docs, also requires user_id."""
        try:
            doc_uuid = uuid.UUID(doc_id)
            user_uuid = uuid.UUID(user_id) if user_id else None
        except (TypeError, ValueError):
            return False

        async with get_session_context() as session:
            print(user_uuid, doc_uuid)
            stmt = delete(KnowledgeDocumentModel).where(
                KnowledgeDocumentModel.id == doc_uuid
            )
            if user_uuid:
                stmt = stmt.where(KnowledgeDocumentModel.user_id == user_uuid)
            
            result = await session.execute(stmt)
            deleted = result.rowcount > 0  # type: ignore
            
        if deleted:
            logger.info("Deleted document id=%s", doc_id)
            # Update metadata cache
            if user_id:
                await DocumentRepository._update_cache_on_delete(
                    user_id=user_id,
                )
            
        return deleted

    @staticmethod
    async def delete_by_data_source(data_source: str) -> int:
        """Delete all documents with a specific data_source. Used for re-ingestion."""
        async with get_session_context() as session:
            stmt = delete(KnowledgeDocumentModel).where(
                KnowledgeDocumentModel.data_source == data_source
            )
            result = await session.execute(stmt)
            count = result.rowcount  # type: ignore
            logger.info("Deleted %d documents with data_source=%s", count, data_source)
            return count

    @staticmethod
    async def list_by_user(
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[KnowledgeDocumentModel]:
        """List personal documents for a user."""
        try:
            user_uuid = uuid.UUID(user_id)
        except (TypeError, ValueError):
            return []

        async with get_session_context() as session:
            stmt = (
                select(KnowledgeDocumentModel)
                .where(KnowledgeDocumentModel.user_id == user_uuid)
                .order_by(KnowledgeDocumentModel.updated_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def count_by_data_source(data_source: str) -> int:
        """Count documents by data_source."""
        async with get_session_context() as session:
            stmt = select(func.count()).where(
                KnowledgeDocumentModel.data_source == data_source
            )
            result = await session.execute(stmt)
            return result.scalar() or 0


# Singleton instance for convenience
document_repository = DocumentRepository()
