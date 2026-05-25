"""
modules/prompt_builder.py — Module 7: Prompt Engineering
──────────────────────────────────────────────────────────
Responsibilities:
  - Construct a clear, grounded RAG prompt from:
      • Retrieved context chunks
      • User's question
  - Enforce strict grounding: LLM must not hallucinate
  - Instruct LLM to acknowledge when the answer is not in the documents
  - Keep the prompt compact and token-efficient

Prompt Design Principles:
  1. Delimit context blocks clearly (XML-style tags)
  2. State the grounding rule explicitly and early
  3. Ask for concise answers with references
  4. Prevent hallucination with a fallback instruction
"""

import logging
from typing import List, Optional
from .retriever import RetrievalResult

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Assembles a RAG prompt from retrieved context and a user question.

    Usage:
        builder = PromptBuilder()
        prompt = builder.build(results, "What are the refund conditions?")
    """

    # ──────────────────────────────────────────
    # Prompt templates
    # ──────────────────────────────────────────

    _SYSTEM_INSTRUCTIONS = """You are a precise document assistant. Your role is to answer questions strictly based on the provided context excerpts from uploaded documents.

STRICT RULES:
1. Answer ONLY using information found in the <context> blocks below.
2. Do NOT use any external knowledge or make assumptions beyond the provided text.
3. If the answer cannot be found in the context, respond exactly: "I could not find the answer to this question in the provided documents."
4. Keep answers clear, concise, and well-structured.
5. When quoting or paraphrasing, attribute the source (document name and page)."""

    _CONTEXT_HEADER = "CONTEXT FROM DOCUMENTS:"
    _QUESTION_HEADER = "QUESTION:"
    _ANSWER_HEADER = "ANSWER:"

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def build(
        self,
        retrieved_results: List[RetrievalResult],
        query: str,
        max_context_chars: int = 6000,
    ) -> str:
        """
        Build the complete prompt string for the LLM.

        Args:
            retrieved_results: Chunks from the retriever, sorted by similarity.
            query: The user's question.
            max_context_chars: Hard cap on total context length (prevents token overflow).

        Returns:
            Formatted prompt string ready to send to the LLM.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        context_block = self._build_context_block(retrieved_results, max_context_chars)
        prompt = self._assemble_prompt(context_block, query)

        logger.debug(f"Built prompt ({len(prompt)} chars, {len(retrieved_results)} chunks).")
        return prompt

    def build_no_context_prompt(self, query: str) -> str:
        """
        Build a prompt for when no context is available (empty vector store).

        Args:
            query: The user's question.

        Returns:
            Prompt that instructs the LLM to say no documents are loaded.
        """
        return (
            f"{self._SYSTEM_INSTRUCTIONS}\n\n"
            f"Note: No documents have been loaded into the system yet.\n\n"
            f"{self._QUESTION_HEADER}\n{query}\n\n"
            f"{self._ANSWER_HEADER}\n"
        )

    # ──────────────────────────────────────────
    # Internal builders
    # ──────────────────────────────────────────

    def _build_context_block(
        self,
        results: List[RetrievalResult],
        max_chars: int,
    ) -> str:
        """
        Format retrieval results into a structured context block.
        Truncates if total context exceeds max_chars.
        """
        if not results:
            return "<context>\nNo relevant context found.\n</context>"

        sections = []
        total_chars = 0

        for i, result in enumerate(results, 1):
            header = (
                f"[Excerpt {i}] Source: {result.source} | "
                f"Page: {result.page} | "
                f"Chunk: {result.chunk_index + 1} | "
                f"Relevance: {result.similarity:.2f}"
            )
            body = result.text.strip()

            section = f"{header}\n{body}"
            section_len = len(section)

            if total_chars + section_len > max_chars:
                # Truncate to fit
                remaining = max_chars - total_chars - len(header) - 20
                if remaining > 100:
                    body = body[:remaining] + "... [truncated]"
                    section = f"{header}\n{body}"
                    sections.append(section)
                break

            sections.append(section)
            total_chars += section_len

        context_text = "\n\n---\n\n".join(sections)
        return f"<context>\n{context_text}\n</context>"

    def _assemble_prompt(self, context_block: str, query: str) -> str:
        """Assemble the final prompt from all parts."""
        return (
            f"{self._SYSTEM_INSTRUCTIONS}\n\n"
            f"{self._CONTEXT_HEADER}\n"
            f"{context_block}\n\n"
            f"{self._QUESTION_HEADER}\n"
            f"{query.strip()}\n\n"
            f"{self._ANSWER_HEADER}\n"
        )

    # ──────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────

    @staticmethod
    def format_citations(results: List[RetrievalResult]) -> List[dict]:
        """
        Format retrieval results as citation objects for UI display.

        Returns:
            List of dicts: {label, source, page, chunk, similarity, excerpt}
        """
        citations = []
        for i, r in enumerate(results, 1):
            citations.append({
                "label": f"Source {i}",
                "source": r.source,
                "page": r.page,
                "chunk": r.chunk_index + 1,
                "similarity": r.similarity,
                "excerpt": r.text[:300] + ("..." if len(r.text) > 300 else ""),
            })
        return citations

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate (≈4 chars per token for English text)."""
        return len(text) // 4
