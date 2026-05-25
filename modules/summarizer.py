"""
modules/summarizer.py — Module 9: Multi-Document Summarizer
─────────────────────────────────────────────────────────────
Responsibilities:
  - Generate per-document summaries using the LLM
  - Generate a combined cross-document synthesis summary
  - Support different summary styles (concise, detailed, bullet-points)
  - Return structured SummaryResult objects for the UI

Design:
  For each document, we:
    1. Retrieve ALL chunks from the vector store (no query — full scan)
    2. Assemble a summarization prompt (chunked to stay under token limits)
    3. Call the LLM once per document
  Then optionally synthesize a cross-document summary.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────

@dataclass
class DocumentSummary:
    """Summary of a single document."""
    source: str
    summary: str
    style: str
    chunk_count: int
    elapsed_seconds: float
    success: bool = True
    error: Optional[str] = None


@dataclass
class MultiDocSummaryResult:
    """Aggregated result of summarizing multiple documents."""
    document_summaries: List[DocumentSummary]
    combined_summary: Optional[str]
    total_documents: int
    total_elapsed_seconds: float
    style: str
    success: bool = True
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Summarizer
# ──────────────────────────────────────────────

class DocumentSummarizer:
    """
    Generates summaries for one or more ingested documents.

    Usage:
        summarizer = DocumentSummarizer(vector_store, llm)
        result = summarizer.summarize(["doc1.pdf", "doc2.pdf"], style="bullet")
    """

    STYLES = {
        "concise": (
            "Write a concise 3-5 sentence summary capturing the core purpose "
            "and key conclusions of the document."
        ),
        "detailed": (
            "Write a detailed structured summary with sections for: "
            "Overview, Key Topics, Important Details, and Conclusions. "
            "Be thorough but avoid unnecessary repetition."
        ),
        "bullet": (
            "Summarize the document as a structured bullet-point list. "
            "Group related points under clear sub-headings. "
            "Use at most 10 top-level bullets."
        ),
    }

    # Max characters of chunk text to feed into the summarization prompt
    MAX_CONTEXT_CHARS = 8000

    def __init__(self, vector_store, llm_handler):
        """
        Args:
            vector_store: VectorStoreManager instance (shared with pipeline).
            llm_handler: LLMHandler instance (shared with pipeline).
        """
        self.vector_store = vector_store
        self.llm = llm_handler
        logger.info("DocumentSummarizer initialized.")

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def summarize(
        self,
        sources: List[str],
        style: str = "concise",
        generate_combined: bool = True,
    ) -> MultiDocSummaryResult:
        """
        Summarize one or more documents.

        Args:
            sources: List of document filenames (must be ingested).
            style: One of 'concise', 'detailed', 'bullet'.
            generate_combined: If True and len(sources) > 1, add a cross-doc synthesis.

        Returns:
            MultiDocSummaryResult with per-doc summaries + optional combined summary.
        """
        if style not in self.STYLES:
            style = "concise"

        overall_start = time.time()
        doc_summaries: List[DocumentSummary] = []

        for source in sources:
            summary = self._summarize_single(source, style)
            doc_summaries.append(summary)

        combined = None
        if generate_combined and len(doc_summaries) > 1:
            combined = self._combined_summary(doc_summaries, style)

        return MultiDocSummaryResult(
            document_summaries=doc_summaries,
            combined_summary=combined,
            total_documents=len(sources),
            total_elapsed_seconds=round(time.time() - overall_start, 2),
            style=style,
        )

    def summarize_all(
        self,
        style: str = "concise",
        generate_combined: bool = True,
    ) -> MultiDocSummaryResult:
        """
        Summarize every document currently in the vector store.

        Args:
            style: Summary style.
            generate_combined: Whether to add a cross-doc synthesis.

        Returns:
            MultiDocSummaryResult.
        """
        sources = self.vector_store.list_sources()
        if not sources:
            return MultiDocSummaryResult(
                document_summaries=[],
                combined_summary=None,
                total_documents=0,
                total_elapsed_seconds=0.0,
                style=style,
                success=False,
                error="No documents are currently ingested.",
            )
        return self.summarize(sources, style=style, generate_combined=generate_combined)

    # ──────────────────────────────────────────
    # Internal: single-document summary
    # ──────────────────────────────────────────

    def _summarize_single(self, source: str, style: str) -> DocumentSummary:
        start = time.time()
        try:
            # Fetch all chunks for this source from ChromaDB
            chunks = self._fetch_all_chunks(source)
            if not chunks:
                return DocumentSummary(
                    source=source,
                    summary=f"No content found for '{source}'.",
                    style=style,
                    chunk_count=0,
                    elapsed_seconds=round(time.time() - start, 2),
                    success=False,
                    error="No chunks found in vector store.",
                )

            context = self._assemble_context(chunks)
            prompt = self._build_summary_prompt(source, context, style)

            logger.info(f"Summarizing '{source}' ({len(chunks)} chunks, style={style})...")
            summary_text = self.llm.generate(prompt)

            return DocumentSummary(
                source=source,
                summary=summary_text,
                style=style,
                chunk_count=len(chunks),
                elapsed_seconds=round(time.time() - start, 2),
            )

        except Exception as e:
            logger.error(f"Failed to summarize '{source}': {e}")
            return DocumentSummary(
                source=source,
                summary=f"Summary generation failed: {e}",
                style=style,
                chunk_count=0,
                elapsed_seconds=round(time.time() - start, 2),
                success=False,
                error=str(e),
            )

    # ──────────────────────────────────────────
    # Internal: combined summary
    # ──────────────────────────────────────────

    def _combined_summary(
        self,
        doc_summaries: List[DocumentSummary],
        style: str,
    ) -> str:
        """Synthesize a cross-document overview from individual summaries."""
        try:
            parts = []
            for ds in doc_summaries:
                if ds.success:
                    parts.append(f"Document: {ds.source}\n{ds.summary}")

            if not parts:
                return "No successful summaries to combine."

            combined_context = "\n\n---\n\n".join(parts)
            style_instruction = self.STYLES[style]

            prompt = (
                "You are a document analysis expert. Below are individual summaries "
                "of multiple documents. Your task is to synthesize them into a single "
                "unified overview that:\n"
                "1. Identifies common themes and connections across documents\n"
                "2. Highlights important differences or complementary information\n"
                "3. Provides an integrated conclusion\n\n"
                f"Style instruction: {style_instruction}\n\n"
                "INDIVIDUAL DOCUMENT SUMMARIES:\n"
                f"{combined_context}\n\n"
                "CROSS-DOCUMENT SYNTHESIS:\n"
            )

            logger.info(f"Generating combined summary for {len(doc_summaries)} documents...")
            return self.llm.generate(prompt)

        except Exception as e:
            logger.error(f"Combined summary generation failed: {e}")
            return f"Combined summary generation failed: {e}"

    # ──────────────────────────────────────────
    # Internal: data helpers
    # ──────────────────────────────────────────

    def _fetch_all_chunks(self, source: str) -> List[Dict]:
        """
        Retrieve ALL stored chunks for a given source document.
        Uses ChromaDB's .get() API (not query) to fetch by metadata filter.
        """
        try:
            raw = self.vector_store.collection.get(
                where={"source": source},
                include=["documents", "metadatas"],
            )
            docs = raw.get("documents", [])
            metas = raw.get("metadatas", [])

            chunks = []
            for text, meta in zip(docs, metas):
                chunks.append({
                    "text": text,
                    "page": meta.get("page", 0),
                    "chunk_index": meta.get("chunk_index", 0),
                })

            # Sort by page then chunk_index for reading order
            chunks.sort(key=lambda c: (c["page"], c["chunk_index"]))
            return chunks

        except Exception as e:
            logger.error(f"Failed to fetch chunks for '{source}': {e}")
            return []

    def _assemble_context(self, chunks: List[Dict]) -> str:
        """
        Join chunk texts into a single context string, capped at MAX_CONTEXT_CHARS.
        Distributes budget evenly across chunks so no single page dominates.
        """
        if not chunks:
            return ""

        per_chunk_budget = self.MAX_CONTEXT_CHARS // max(len(chunks), 1)
        parts = []
        total = 0

        for chunk in chunks:
            text = chunk["text"].strip()
            if not text:
                continue
            excerpt = text[:per_chunk_budget]
            if len(text) > per_chunk_budget:
                excerpt += "..."
            parts.append(f"[Page {chunk['page']}] {excerpt}")
            total += len(excerpt)
            if total >= self.MAX_CONTEXT_CHARS:
                break

        return "\n\n".join(parts)

    def _build_summary_prompt(self, source: str, context: str, style: str) -> str:
        """Assemble the LLM prompt for a single-document summary."""
        style_instruction = self.STYLES[style]
        return (
            "You are a precise document summarization assistant.\n"
            "Summarize the following document excerpts ONLY based on the content provided.\n"
            f"Style instruction: {style_instruction}\n\n"
            f"DOCUMENT: {source}\n\n"
            "<content>\n"
            f"{context}\n"
            "</content>\n\n"
            "SUMMARY:\n"
        )

    # ──────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────

    @staticmethod
    def available_styles() -> List[str]:
        return ["concise", "detailed", "bullet"]
