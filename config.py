"""
config.py — Central configuration for the RAG Document Assistant.
All tunable parameters live here. Edit this file to adjust system behavior.
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
UPLOAD_TEMP_DIR = os.path.join(BASE_DIR, "uploads_temp")

# ─────────────────────────────────────────────
# Document Chunking
# ─────────────────────────────────────────────
CHUNK_SIZE = 500          # Target tokens/chars per chunk
CHUNK_OVERLAP = 50     # Overlap between adjacent chunks
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Recursive split order

# ─────────────────────────────────────────────
# Embedding Model (SentenceTransformers — local, no API key)
# ─────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────
# Vector Store (ChromaDB)
# ─────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "rag_documents"

# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────
TOP_K_RESULTS = 4                # Number of chunks to retrieve per query
SIMILARITY_THRESHOLD = 0.0       # Minimum similarity score (0 = no filter)

# ─────────────────────────────────────────────
# LLM (Ollama — locally hosted)
# ─────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3"         # Change to "phi3" or any pulled model
LLM_TEMPERATURE = 0.1            # Low temp for factual, grounded answers
LLM_MAX_TOKENS = 1024

# ─────────────────────────────────────────────
# Supported file types
# ─────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [
    # Documents
    ".pdf", ".docx", ".txt", ".md",
    # Spreadsheets
    ".csv", ".xlsx",
    # Presentations
    ".pptx",
    # Web
    ".html", ".htm",
    # Images (OCR)
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
]

# File type groups for UI display
FILE_TYPE_GROUPS = {
    "Documents": [".pdf", ".docx", ".txt", ".md"],
    "Spreadsheets": [".csv", ".xlsx"],
    "Presentations": [".pptx"],
    "Web": [".html", ".htm"],
    "Images (OCR)": [".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
}

# ─────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────
REPORT_MAX_CONTEXT_CHARS = 8000   # Max document context sent per report
REPORT_DEFAULT_AUDIENCE = "presales team"
REPORT_LLM_MAX_TOKENS = 2048      # Reports need more tokens than chat