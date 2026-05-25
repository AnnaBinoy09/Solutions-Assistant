"""
modules/__init__.py
Exposes the public API of each module for clean imports.
"""

from .document_loader import DocumentLoader
from .chunker import DocumentChunker
from .embedder import EmbeddingEngine
from .vector_store import VectorStoreManager
from .retriever import Retriever
from .llm_handler import LLMHandler
from .prompt_builder import PromptBuilder
from .rag_pipeline import RAGPipeline
from .report_prompt_builder import ReportPromptBuilder, ReportPromptConfig, REPORT_TYPES
from .report_pipeline import ReportPipeline, ReportResult

__all__ = [
    # Core RAG
    "DocumentLoader",
    "DocumentChunker",
    "EmbeddingEngine",
    "VectorStoreManager",
    "Retriever",
    "LLMHandler",
    "PromptBuilder",
    "RAGPipeline",
    # Report generation
    "ReportPromptBuilder",
    "ReportPromptConfig",
    "REPORT_TYPES",
    "ReportPipeline",
    "ReportResult",
]