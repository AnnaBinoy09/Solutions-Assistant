
import logging

from typing import List
from .document_loader import Document

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Splits Document objects into smaller chunks using recursive text splitting.

    Usage:
        chunker = DocumentChunker(chunk_size=700, chunk_overlap=120)
        chunks = chunker.split(documents)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None,
    ):
        """
        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between adjacent chunks (preserves context).
            separators: Ordered list of separators for recursive splitting.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

        logger.info(
            f"DocumentChunker initialized — chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of Document objects into smaller chunks.

        Args:
            documents: Documents from DocumentLoader.

        Returns:
            List[Document] — each chunk has page_content + inherited metadata
                             plus chunk_index and chunk_total keys.
        """
        if not documents:
            logger.warning("No documents provided to chunker.")
            return []

        all_chunks = []
        for doc in documents:
            chunks = self._split_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunking complete: {len(documents)} documents → {len(all_chunks)} chunks"
        )
        return all_chunks

    # ──────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────

    def _split_document(self, doc: Document) -> List[Document]:
        """Split a single Document into chunks, preserving metadata."""
        text = doc.page_content
        raw_chunks = self._recursive_split(text, self.separators)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Clone metadata and add chunk-level keys
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = i
            metadata["chunk_total"] = len(raw_chunks)

            chunks.append(Document(page_content=chunk_text, metadata=metadata))

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the separator hierarchy.
        Merges short splits back together to respect chunk_size with overlap.
        """
        if not separators:
            return self._merge_with_overlap([text])

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)  # character-level fallback

        # Recursively split pieces that are still too large
        good_splits = []
        for split in splits:
            if len(split) <= self.chunk_size:
                good_splits.append(split)
            else:
                # This piece is still too big — recurse with next separator
                sub_splits = self._recursive_split(split, remaining_separators)
                good_splits.extend(sub_splits)

        return self._merge_with_overlap(good_splits, separator)

    def _merge_with_overlap(
        self, splits: List[str], separator: str = ""
    ) -> List[str]:
        """
        Merge small splits into chunks of up to chunk_size,
        adding chunk_overlap characters of context from the previous chunk.
        """
        chunks = []
        current_parts: List[str] = []
        current_len = 0
        sep_len = len(separator)

        for part in splits:
            part_len = len(part)
            # +sep_len for the rejoining separator
            addition = sep_len + part_len if current_parts else part_len

            if current_len + addition > self.chunk_size and current_parts:
                # Flush current chunk
                chunk_text = separator.join(current_parts)
                chunks.append(chunk_text)

                # Build overlap: keep trailing parts that fit within chunk_overlap
                overlap_parts = []
                overlap_len = 0
                for p in reversed(current_parts):
                    if overlap_len + len(p) + sep_len <= self.chunk_overlap:
                        overlap_parts.insert(0, p)
                        overlap_len += len(p) + sep_len
                    else:
                        break

                current_parts = overlap_parts
                current_len = overlap_len

            current_parts.append(part)
            current_len += part_len + (sep_len if len(current_parts) > 1 else 0)

        if current_parts:
            chunks.append(separator.join(current_parts))

        return [c for c in chunks if c.strip()]

    # ──────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────

    def stats(self, chunks: List[Document]) -> dict:
        """Return basic statistics about chunk sizes."""
        if not chunks:
            return {}
        lengths = [len(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_length": round(sum(lengths) / len(lengths)),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }
