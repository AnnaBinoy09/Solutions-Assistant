
import logging
from typing import List, Union
from .document_loader import Document

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Generates semantic embeddings using a local SentenceTransformer model.

    Usage:
        engine = EmbeddingEngine("all-MiniLM-L6-v2")
        query_vec = engine.embed_query("What is the refund policy?")
        doc_vecs  = engine.embed_documents(chunks)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: HuggingFace / SentenceTransformers model identifier.
                        First run downloads the model (~80 MB). Cached after that.
        """
        self.model_name = model_name
        self._model = None  # Lazy loading
        logger.info(f"EmbeddingEngine configured — model='{model_name}'")

    # ──────────────────────────────────────────
    # Lazy model loader
    # ──────────────────────────────────────────

    @property
    def model(self):
        """Load the SentenceTransformer model on first access (singleton)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "Install sentence-transformers: pip install sentence-transformers"
                )
            logger.info(f"Loading embedding model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully.")
        return self._model

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            query: User's natural language question.

        Returns:
            List[float] — dense embedding vector.
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Embed a list of Document chunks.

        Args:
            documents: Chunked Document objects with page_content.

        Returns:
            List[List[float]] — one embedding vector per document.
        """
        if not documents:
            logger.warning("embed_documents called with empty list.")
            return []

        texts = [doc.page_content for doc in documents]
        logger.info(f"Embedding {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 20,
            batch_size=32,
        )

        logger.info("Embedding complete.")
        return [e.tolist() for e in embeddings]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed raw strings directly (utility method).

        Args:
            texts: List of text strings.

        Returns:
            List[List[float]] — embedding vectors.
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        return [e.tolist() for e in embeddings]

    # ──────────────────────────────────────────
    # Info
    # ──────────────────────────────────────────

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()

    def __repr__(self):
        return f"EmbeddingEngine(model='{self.model_name}', dim={self.embedding_dimension})"
