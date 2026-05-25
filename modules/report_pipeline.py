"""
modules/report_pipeline.py — Presales Report Generation Pipeline
──────────────────────────────────────────────────────────────────
Responsibilities:
  - Separate from the RAG chat pipeline
  - Accept one or more already-ingested document names (or raw file paths)
  - Retrieve ALL chunks for selected documents (not query-based)
  - Build a report-type-specific prompt via ReportPromptBuilder
  - Call LLMHandler to generate the structured report
  - Return a ReportResult with markdown content + metadata

Flow:
  Selected Documents
        ↓
  VectorStoreManager.get_all_chunks_for_source()
        ↓
  ReportPromptBuilder.build_context_from_texts()
        ↓
  ReportPromptBuilder.build(context, config)
        ↓
  LLMHandler.generate(prompt)
        ↓
  ReportResult (markdown, metadata, timing)

This module is intentionally decoupled from RAGPipeline.
It can be used independently and never touches the retriever.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import config
from .vector_store import VectorStoreManager
from .llm_handler import LLMHandler
from .report_prompt_builder import ReportPromptBuilder, ReportPromptConfig, REPORT_TYPES
from .document_loader import DocumentLoader

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────

@dataclass
class ReportResult:
    """Complete result from a report generation operation."""
    success: bool
    report_type: str
    report_type_label: str
    markdown: str
    source_documents: List[str]
    elapsed_seconds: float
    word_count: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

class ReportPipeline:
    """
    Presales report generation pipeline.

    Usage:
        pipeline = ReportPipeline(vector_store, llm_handler)

        result = pipeline.generate_report(
            source_documents=["rfp.pdf", "requirements.docx"],
            report_type="executive_summary",
            custom_instruction="Focus on cloud infrastructure requirements.",
            audience="VP of Sales",
        )
        print(result.markdown)
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        llm_handler: LLMHandler,
    ):
        """
        Args:
            vector_store: Shared VectorStoreManager instance (from RAGPipeline).
            llm_handler: Shared LLMHandler instance (from RAGPipeline).
        """
        self.vector_store = vector_store
        self.llm = llm_handler
        self.prompt_builder = ReportPromptBuilder()
        logger.info("ReportPipeline initialized.")

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def generate_report(
        self,
        source_documents: List[str],
        report_type: str = "executive_summary",
        custom_instruction: str = "",
        audience: str = "presales team",
        max_context_chars: int = 8000,
    ) -> ReportResult:
        """
        Generate a structured presales report from indexed documents.

        Args:
            source_documents: List of document source names to include.
            report_type: One of the keys in REPORT_TYPES dict.
            custom_instruction: Optional focus or custom prompt addition.
            audience: Target audience label (used in prompt framing).
            max_context_chars: Max chars of document context to send to LLM.

        Returns:
            ReportResult with markdown report content.
        """
        start = time.time()
        label = REPORT_TYPES.get(report_type, "Custom Analysis")

        if not source_documents:
            return self._error_result(
                report_type, label, [], start,
                "No source documents selected."
            )

        try:
            # Step 1: Retrieve all chunks for selected documents
            logger.info(
                f"[1/3] Fetching all chunks for: {source_documents}"
            )
            texts, sources = self._fetch_all_chunks(
                source_documents, max_context_chars
            )

            if not texts:
                return self._error_result(
                    report_type, label, source_documents, start,
                    "No indexed content found for the selected documents. "
                    "Please ensure they are ingested first."
                )

            # Step 2: Build prompt
            logger.info(f"[2/3] Building {label} prompt...")
            context_text = self.prompt_builder.build_context_from_texts(
                texts, sources, max_chars=max_context_chars
            )

            prompt_config = ReportPromptConfig(
                report_type=report_type,
                documents=source_documents,
                custom_instruction=custom_instruction,
                audience=audience,
                max_context_chars=max_context_chars,
            )
            prompt = self.prompt_builder.build(context_text, prompt_config)

            # Step 3: Generate
            logger.info(f"[3/3] Generating {label} with LLM...")
            markdown = self.llm.generate(prompt)

            elapsed = round(time.time() - start, 2)
            word_count = len(markdown.split())
            logger.info(
                f"Report generated: {label} — "
                f"{word_count} words in {elapsed}s."
            )

            return ReportResult(
                success=True,
                report_type=report_type,
                report_type_label=label,
                markdown=markdown,
                source_documents=source_documents,
                elapsed_seconds=elapsed,
                word_count=word_count,
                metadata={
                    "chunks_used": len(texts),
                    "context_chars": len(context_text),
                    "audience": audience,
                },
            )

        except ConnectionError as e:
            return self._error_result(
                report_type, label, source_documents, start,
                f"LLM Connection Error: {e}"
            )
        except Exception as e:
            logger.exception("Unexpected error during report generation.")
            return self._error_result(
                report_type, label, source_documents, start,
                f"Unexpected error: {e}"
            )

    def available_report_types(self) -> Dict[str, str]:
        """Return the registry of available report types."""
        return dict(REPORT_TYPES)

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _fetch_all_chunks(
        self,
        source_documents: List[str],
        max_chars: int,
    ) -> tuple[List[str], List[str]]:
        """
        Retrieve all stored text chunks for the given source documents.
        Returns (texts, sources) lists aligned by index.
        Stops adding chunks once max_chars is reached.
        """
        texts = []
        sources = []
        total_chars = 0

        for source_name in source_documents:
            try:
                # Fetch all metadata for this source
                results = self.vector_store.collection.get(
                    where={"source": source_name},
                    include=["documents", "metadatas"],
                )

                docs = results.get("documents", [])
                metas = results.get("metadatas", [])

                # Sort by page then chunk_index for coherent ordering
                paired = list(zip(docs, metas))
                paired.sort(key=lambda x: (
                    int(x[1].get("page", 0)),
                    int(x[1].get("chunk_index", 0)),
                ))

                for text, meta in paired:
                    if not text or not text.strip():
                        continue
                    if total_chars + len(text) > max_chars:
                        # Truncate the last chunk to fit
                        remaining = max_chars - total_chars
                        if remaining > 200:
                            texts.append(text[:remaining] + "... [truncated]")
                            sources.append(source_name)
                        break
                    texts.append(text)
                    sources.append(source_name)
                    total_chars += len(text)

                logger.info(
                    f"Fetched chunks for '{source_name}': "
                    f"{len(docs)} chunks, {total_chars} total chars so far."
                )

            except Exception as e:
                logger.error(f"Failed to fetch chunks for '{source_name}': {e}")
                continue

        return texts, sources

    @staticmethod
    def _error_result(
        report_type: str,
        label: str,
        docs: List[str],
        start: float,
        error: str,
    ) -> ReportResult:
        logger.error(f"Report generation failed: {error}")
        return ReportResult(
            success=False,
            report_type=report_type,
            report_type_label=label,
            markdown="",
            source_documents=docs,
            elapsed_seconds=round(time.time() - start, 2),
            error=error,
        )
