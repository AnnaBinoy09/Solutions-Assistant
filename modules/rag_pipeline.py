"""
modules/rag_pipeline.py — Module 8: End-to-End RAG Pipeline
────────────────────────────────────────────────────────────
Responsibilities:
  - Orchestrate all modules into a clean, unified interface
  - Expose two primary operations:
      1. ingest(file_path) — process and store a document
      2. query(question)   — retrieve and answer a question
  - Provide error isolation: module failures are caught and surfaced clearly
  - Return structured RAGResponse objects with answer + citations

This module is the single entry point for the Streamlit UI (app.py).
It wires together:
  DocumentLoader → DocumentChunker → EmbeddingEngine → VectorStoreManager
                                                      ↓
                      PromptBuilder ← Retriever ← EmbeddingEngine
                           ↓
                       LLMHandler → RAGResponse
"""

"""
modules/rag_pipeline.py — Module 8: End-to-End RAG Pipeline
────────────────────────────────────────────────────────────
Responsibilities:
  - Orchestrate all modules into a clean, unified interface
  - Expose three primary operations:
      1. ingest(file_path)                       — process and store a document
      2. query(question)                         — retrieve and answer a question
      3. summarize(sources, style)               — summarize one or more documents
  - Provide error isolation: module failures are caught and surfaced clearly
  - Return structured result objects with answer + citations

This module is the single entry point for the Streamlit UI (app.py).
It wires together:
  DocumentLoader → DocumentChunker → EmbeddingEngine → VectorStoreManager
                                                      ↓
                      PromptBuilder ← Retriever ← EmbeddingEngine
                           ↓
                       LLMHandler → RAGResponse

  DocumentSummarizer uses VectorStoreManager + LLMHandler directly.
"""

import os
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import config
from .document_loader import DocumentLoader
from .chunker import DocumentChunker
from .embedder import EmbeddingEngine
from .vector_store import VectorStoreManager
from .retriever import Retriever, RetrievalResult
from .llm_handler import LLMHandler
from .prompt_builder import PromptBuilder
from .summarizer import DocumentSummarizer, MultiDocSummaryResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Response containers
# ──────────────────────────────────────────────

@dataclass
class IngestionResult:
    """Result of a document ingestion operation."""
    success: bool
    file_name: str
    chunks_created: int
    elapsed_seconds: float
    error: Optional[str] = None


@dataclass
class RAGResponse:
    """Complete response from a RAG query."""
    answer: str
    citations: List[dict]
    retrieved_chunks: List[RetrievalResult]
    query: str
    elapsed_seconds: float
    success: bool = True
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline: ingestion, querying, and summarization.

    Usage:
        pipeline = RAGPipeline()

        # Ingest a document
        result = pipeline.ingest("/path/to/guide.pdf")

        # Query
        response = pipeline.query("What is the return policy?")
        print(response.answer)

        # Summarize
        summary = pipeline.summarize(["guide.pdf", "terms.pdf"], style="bullet")
        print(summary.combined_summary)
    """

    def __init__(self):
        logger.info("Initializing RAG Pipeline...")

        # Module 1: Loader
        self.loader = DocumentLoader()

        # Module 2: Chunker
        self.chunker = DocumentChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.SEPARATORS,
        )

        # Module 3: Embedder (shared between ingestion and retrieval)
        self.embedder = EmbeddingEngine(model_name=config.EMBEDDING_MODEL_NAME)

        # Module 4: Vector Store
        self.vector_store = VectorStoreManager(
            persist_dir=config.CHROMA_DB_PATH,
            collection_name=config.CHROMA_COLLECTION_NAME,
        )

        # Module 5: Retriever
        self.retriever = Retriever(
            embedding_engine=self.embedder,
            vector_store=self.vector_store,
            top_k=config.TOP_K_RESULTS,
            similarity_threshold=config.SIMILARITY_THRESHOLD,
        )

        # Module 6: LLM
        self.llm = LLMHandler(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )

        # Module 7: Prompt builder
        self.prompt_builder = PromptBuilder()

        # Module 9: Summarizer (shares vector_store + llm with pipeline)
        self.summarizer = DocumentSummarizer(
            vector_store=self.vector_store,
            llm_handler=self.llm,
        )

        logger.info("RAG Pipeline ready.")

    # ──────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────

    def ingest(self, file_path: str, force_reingest: bool = False) -> IngestionResult:
        """
        Load, chunk, embed, and store a document.

        Args:
            file_path: Path to PDF or DOCX file.
            force_reingest: If True, re-process even if already indexed.

        Returns:
            IngestionResult with success status and metadata.
        """
        start = time.time()
        file_name = os.path.basename(file_path)

        try:
            # Check if already indexed
            if not force_reingest and self.vector_store.is_source_indexed(file_name):
                logger.info(f"'{file_name}' already indexed — skipping.")
                return IngestionResult(
                    success=True,
                    file_name=file_name,
                    chunks_created=0,
                    elapsed_seconds=round(time.time() - start, 2),
                    error="Already indexed (use force_reingest=True to re-process).",
                )

            # If force re-ingest, remove old data first
            if force_reingest:
                self.vector_store.delete_by_source(file_name)

            # Step 1: Load
            logger.info(f"[1/4] Loading '{file_name}'...")
            documents = self.loader.load(file_path)
            if not documents:
                raise ValueError(f"No text could be extracted from '{file_name}'.")

            # Step 2: Chunk
            logger.info(f"[2/4] Chunking {len(documents)} pages/sections...")
            chunks = self.chunker.split(documents)
            if not chunks:
                raise ValueError("Chunking produced zero chunks.")

            stats = self.chunker.stats(chunks)
            logger.info(f"       Chunk stats: {stats}")

            # Step 3: Embed
            logger.info(f"[3/4] Embedding {len(chunks)} chunks...")
            embeddings = self.embedder.embed_documents(chunks)

            # Step 4: Store
            logger.info(f"[4/4] Storing in ChromaDB...")
            stored = self.vector_store.add_documents(chunks, embeddings)

            elapsed = round(time.time() - start, 2)
            logger.info(
                f"Ingestion complete: '{file_name}' — "
                f"{stored} chunks stored in {elapsed}s."
            )

            return IngestionResult(
                success=True,
                file_name=file_name,
                chunks_created=stored,
                elapsed_seconds=elapsed,
            )

        except FileNotFoundError as e:
            return self._ingestion_error(file_name, str(e), start)
        except ValueError as e:
            return self._ingestion_error(file_name, str(e), start)
        except Exception as e:
            logger.exception(f"Unexpected error during ingestion of '{file_name}'")
            return self._ingestion_error(file_name, f"Unexpected error: {e}", start)

    # ──────────────────────────────────────────
    # Query
    # ──────────────────────────────────────────

    def query(
        self,
        question: str,
        source_filter: Optional[str] = None,
    ) -> RAGResponse:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: Natural language question from the user.
            source_filter: Optional — restrict retrieval to one document.

        Returns:
            RAGResponse with answer, citations, and retrieved chunks.
        """
        start = time.time()

        try:
            # Guard: empty vector store
            if self.vector_store.document_count() == 0:
                prompt = self.prompt_builder.build_no_context_prompt(question)
                answer = self._safe_llm_call(prompt)
                return RAGResponse(
                    answer=answer,
                    citations=[],
                    retrieved_chunks=[],
                    query=question,
                    elapsed_seconds=round(time.time() - start, 2),
                )

            # Step 1: Retrieve
            logger.info(f"[1/3] Retrieving context for: {question[:80]!r}")
            results = self.retriever.retrieve(question, source_filter=source_filter)

            if not results:
                return RAGResponse(
                    answer=(
                        "I could not find any relevant content in the documents "
                        "for your question. Please try rephrasing or uploading "
                        "additional documents."
                    ),
                    citations=[],
                    retrieved_chunks=[],
                    query=question,
                    elapsed_seconds=round(time.time() - start, 2),
                )

            # Step 2: Build prompt
            logger.info(f"[2/3] Building prompt with {len(results)} chunks...")
            prompt = self.prompt_builder.build(results, question)

            # Step 3: Generate answer
            logger.info("[3/3] Generating answer with LLM...")
            answer = self._safe_llm_call(prompt)

            # Format citations
            citations = self.prompt_builder.format_citations(results)

            elapsed = round(time.time() - start, 2)
            logger.info(f"Query answered in {elapsed}s.")

            return RAGResponse(
                answer=answer,
                citations=citations,
                retrieved_chunks=results,
                query=question,
                elapsed_seconds=elapsed,
            )

        except ConnectionError as e:
            return RAGResponse(
                answer=f"⚠️ LLM Connection Error: {e}",
                citations=[],
                retrieved_chunks=[],
                query=question,
                elapsed_seconds=round(time.time() - start, 2),
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.exception("Unexpected error during query.")
            return RAGResponse(
                answer=f"⚠️ An unexpected error occurred: {e}",
                citations=[],
                retrieved_chunks=[],
                query=question,
                elapsed_seconds=round(time.time() - start, 2),
                success=False,
                error=str(e),
            )

    # ──────────────────────────────────────────
    # Summarization
    # ──────────────────────────────────────────

    def summarize(
        self,
        sources: Optional[List[str]] = None,
        style: str = "concise",
        generate_combined: bool = True,
    ) -> MultiDocSummaryResult:
        """
        Summarize one or more ingested documents.

        Args:
            sources: Document filenames to summarize. None = all documents.
            style: 'concise' | 'detailed' | 'bullet'
            generate_combined: If True and multiple docs, add cross-doc synthesis.

        Returns:
            MultiDocSummaryResult with per-doc summaries + optional combined summary.
        """
        if sources is None:
            return self.summarizer.summarize_all(
                style=style,
                generate_combined=generate_combined,
            )
        return self.summarizer.summarize(
            sources=sources,
            style=style,
            generate_combined=generate_combined,
        )

    # ──────────────────────────────────────────
    # Utility / Status
    # ──────────────────────────────────────────

    def list_documents(self) -> List[str]:
        """Return names of all ingested documents."""
        return self.vector_store.list_sources()

    def document_count(self) -> int:
        """Return total number of stored chunks."""
        return self.vector_store.document_count()

    def llm_status(self) -> dict:
        """Check LLM availability."""
        available = self.llm.is_available()
        return {
            "available": available,
            "model": config.OLLAMA_MODEL,
            "base_url": config.OLLAMA_BASE_URL,
            "available_models": self.llm.list_models() if not available else [],
        }

    def delete_document(self, source_name: str) -> int:
        """Remove a document from the vector store by name."""
        return self.vector_store.delete_by_source(source_name)

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _safe_llm_call(self, prompt: str) -> str:
        """Call the LLM with graceful fallback."""
        try:
            return self.llm.generate(prompt)
        except ConnectionError:
            raise  # Re-raise for caller to handle
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error generating response: {e}"

    @staticmethod
    def _ingestion_error(file_name: str, error: str, start: float) -> IngestionResult:
        logger.error(f"Ingestion failed for '{file_name}': {error}")
        return IngestionResult(
            success=False,
            file_name=file_name,
            chunks_created=0,
            elapsed_seconds=round(time.time() - start, 2),
            error=error,
        )