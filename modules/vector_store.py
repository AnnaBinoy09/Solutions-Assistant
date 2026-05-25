

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple

from .document_loader import Document

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages a persistent ChromaDB collection for RAG document storage.

    Usage:
        store = VectorStoreManager(persist_dir="./chroma_db", collection_name="rag_docs")
        store.add_documents(chunks, embeddings)
        results = store.query(query_embedding, top_k=4)
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_documents",
    ):
        """
        Args:
            persist_dir: Directory for ChromaDB persistence.
            collection_name: Name of the ChromaDB collection.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None

        os.makedirs(persist_dir, exist_ok=True)
        logger.info(
            f"VectorStoreManager — collection='{collection_name}', "
            f"persist_dir='{persist_dir}'"
        )

    # ──────────────────────────────────────────
    # Lazy init
    # ──────────────────────────────────────────

    @property
    def client(self):
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError("Install chromadb: pip install chromadb")

            self._client = chromadb.Client()
            logger.info("ChromaDB PersistentClient initialized.")
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Cosine similarity
            )
            logger.info(
                f"Collection '{self.collection_name}' ready "
                f"(existing docs: {self._collection.count()})."
            )
        return self._collection

    # ──────────────────────────────────────────
    # Add documents
    # ──────────────────────────────────────────

    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
    ) -> int:
        """
        Store document chunks and their embeddings in ChromaDB.

        Args:
            documents: List of Document chunks.
            embeddings: Corresponding embedding vectors (same order).

        Returns:
            Number of documents successfully added.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents vs {len(embeddings)} embeddings."
            )

        if not documents:
            return 0

        ids, texts, metas, vecs = [], [], [], []

        for doc, emb in zip(documents, embeddings):
            doc_id = self._make_id(doc)

            # Sanitize metadata — ChromaDB only accepts str/int/float/bool values
            meta = self._sanitize_metadata(doc.metadata)

            ids.append(doc_id)
            texts.append(doc.page_content)
            metas.append(meta)
            vecs.append(emb)

        # Upsert (add or update) to handle re-ingestion gracefully
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vecs,
            metadatas=metas,
        )

        logger.info(f"Upserted {len(ids)} chunks into '{self.collection_name}'.")
        return len(ids)

    # ──────────────────────────────────────────
    # Query
    # ──────────────────────────────────────────

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        where_filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks for a query embedding.

        Args:
            query_embedding: Dense vector of the user's query.
            top_k: Number of results to return.
            where_filter: Optional ChromaDB metadata filter dict.

        Returns:
            List of result dicts with keys: text, metadata, distance, id.
        """
        count = self.collection.count()
        if count == 0:
            logger.warning("Collection is empty — no documents ingested yet.")
            return []

        effective_k = min(top_k, count)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": effective_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        raw = self.collection.query(**query_kwargs)

        results = []
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        ids = raw.get("ids", [[]])[0]

        for text, meta, dist, doc_id in zip(docs, metas, distances, ids):
            results.append({
                "text": text,
                "metadata": meta,
                "distance": dist,
                "similarity": round(1 - dist, 4),  # cosine: similarity = 1 - distance
                "id": doc_id,
            })

        logger.info(f"Retrieved {len(results)} chunks (top_k={effective_k}).")
        return results

    # ──────────────────────────────────────────
    # Management
    # ──────────────────────────────────────────

    def delete_by_source(self, source_name: str) -> int:
        """
        Delete all chunks belonging to a specific source document.

        Args:
            source_name: The 'source' metadata field value (filename).

        Returns:
            Number of chunks deleted.
        """
        try:
            results = self.collection.get(
                where={"source": source_name},
                include=["metadatas"],
            )
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for '{source_name}'.")
            return len(ids_to_delete)
        except Exception as e:
            logger.error(f"Failed to delete source '{source_name}': {e}")
            return 0

    def list_sources(self) -> List[str]:
        """Return unique source document names stored in the collection."""
        try:
            results = self.collection.get(include=["metadatas"])
            sources = {
                m.get("source", "unknown")
                for m in results.get("metadatas", [])
                if m
            }
            return sorted(sources)
        except Exception as e:
            logger.error(f"Failed to list sources: {e}")
            return []

    def document_count(self) -> int:
        """Return total number of chunks stored."""
        return self.collection.count()

    def is_source_indexed(self, source_name: str) -> bool:
        """Check whether a source file has already been ingested."""
        try:
            results = self.collection.get(
                where={"source": source_name},
                limit=1,
                include=["metadatas"],
            )
            return len(results.get("ids", [])) > 0
        except Exception:
            return False

    def reset_collection(self):
        """Delete and recreate the collection (destructive — use with care)."""
        self.client.delete_collection(self.collection_name)
        self._collection = None  # Force re-creation on next access
        logger.warning(f"Collection '{self.collection_name}' has been reset.")

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _make_id(doc: Document) -> str:
        """Generate a deterministic ID from source + page + chunk_index."""
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        chunk_idx = doc.metadata.get("chunk_index", 0)
        content_hash = hashlib.md5(doc.page_content[:200].encode()).hexdigest()[:8]
        return f"{source}__p{page}__c{chunk_idx}__{content_hash}"

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        """Ensure all metadata values are ChromaDB-compatible primitives."""
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized
