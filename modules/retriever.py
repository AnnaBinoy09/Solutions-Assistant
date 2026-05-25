"""
modules/retriever.py — Module 5: Semantic Retrieval
─────────────────────────────────────────────────────
Responsibilities:
  - Accept a user query and retrieve top-k relevant chunks
  - Embed the query using EmbeddingEngine
  - Query VectorStoreManager for nearest neighbors
  - De-duplicate results and apply similarity thresholds
  - Return structured RetrievalResult objects with source metadata

Design note:
  De-duplication works at chunk-text level. If multiple chunks
  share near-identical content (e.g., headers repeated across pages),
  only the highest-scoring one is kept.
"""

import logging
from typing import List, Optional
from dataclasses import dataclass, field

from .embedder import EmbeddingEngine
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved chunk with full provenance metadata."""
    text: str
    source: str
    page: int
    chunk_index: int
    similarity: float
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"RetrievalResult(source={self.source!r}, page={self.page}, "
            f"chunk={self.chunk_index}, similarity={self.similarity:.3f})"
        )

    def citation_label(self) -> str:
        """Human-readable citation label for UI display."""
        return f"{self.source} — Page {self.page}, Chunk {self.chunk_index + 1}"


class Retriever:
    """
    Orchestrates query embedding → vector search → de-duplication.

    Usage:
        retriever = Retriever(embedding_engine, vector_store, top_k=4)
        results = retriever.retrieve("What is the cancellation policy?")
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStoreManager,
        top_k: int = 4,
        similarity_threshold: float = 0.0,
        dedup_threshold: float = 0.95,
    ):
        """
        Args:
            embedding_engine: Initialized EmbeddingEngine instance.
            vector_store: Initialized VectorStoreManager instance.
            top_k: Number of chunks to retrieve.
            similarity_threshold: Minimum similarity score to include a result.
                                   Set 0.0 to disable filtering.
            dedup_threshold: Chunks with similarity to each other above this
                              value are considered duplicates (higher one kept).
        """
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.dedup_threshold = dedup_threshold

        logger.info(
            f"Retriever initialized — top_k={top_k}, "
            f"similarity_threshold={similarity_threshold}"
        )

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        source_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Embed a query and retrieve the most relevant document chunks.

        Args:
            query: User's natural language question.
            source_filter: Optional — restrict retrieval to a specific document name.

        Returns:
            List[RetrievalResult] sorted by similarity descending.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info(f"Retrieving for query: {query[:80]!r}")

        # 1. Embed the query
        query_embedding = self.embedding_engine.embed_query(query)

        # 2. Build optional metadata filter
        where_filter = {"source": source_filter} if source_filter else None

        # 3. Query the vector store (fetch more than top_k to allow post-filtering)
        fetch_k = min(self.top_k * 2, max(self.top_k, 10))
        raw_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=fetch_k,
            where_filter=where_filter,
        )

        if not raw_results:
            logger.warning("No results returned from vector store.")
            return []

        # 4. Convert to RetrievalResult objects
        results = [self._to_retrieval_result(r) for r in raw_results]

        # 5. Apply similarity threshold
        if self.similarity_threshold > 0.0:
            before = len(results)
            results = [r for r in results if r.similarity >= self.similarity_threshold]
            logger.debug(
                f"Threshold filter: {before} → {len(results)} results "
                f"(threshold={self.similarity_threshold})"
            )

        # 6. De-duplicate
        results = self._deduplicate(results)

        # 7. Return top_k
        results = results[: self.top_k]

        logger.info(
            f"Returned {len(results)} chunks for query. "
            f"Top similarity: {results[0].similarity:.3f}" if results else "No results."
        )
        return results

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _to_retrieval_result(raw: dict) -> RetrievalResult:
        """Convert a raw ChromaDB result dict to a RetrievalResult."""
        meta = raw.get("metadata", {})
        return RetrievalResult(
            text=raw.get("text", ""),
            source=meta.get("source", "unknown"),
            page=int(meta.get("page", 0)),
            chunk_index=int(meta.get("chunk_index", 0)),
            similarity=raw.get("similarity", 0.0),
            metadata=meta,
        )

    def _deduplicate(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Remove near-duplicate chunks by comparing text prefixes.
        When two chunks share >80% of their leading characters, keep the
        higher-similarity one.
        """
        seen_texts = []
        unique = []

        for result in results:
            is_dup = False
            for seen in seen_texts:
                if self._text_overlap(result.text, seen) > self.dedup_threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(result)
                seen_texts.append(result.text)

        if len(unique) < len(results):
            logger.debug(f"Dedup: {len(results)} → {len(unique)} results.")

        return unique

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        """
        Estimate overlap ratio between two texts using leading/trailing comparison.
        Simple but fast — avoids expensive full edit-distance computation.
        """
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if not shorter:
            return 0.0
        # Check how much of the shorter text appears at the start of the longer
        common_prefix = 0
        for c1, c2 in zip(shorter, longer):
            if c1 == c2:
                common_prefix += 1
            else:
                break
        return common_prefix / len(shorter)

    # ──────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────

    def format_results_for_display(self, results: List[RetrievalResult]) -> str:
        """Format retrieved results as a readable string (for debugging)."""
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r.citation_label()} (similarity={r.similarity:.3f})\n"
                f"    {r.text[:200].replace(chr(10), ' ')}..."
            )
        return "\n\n".join(lines)
