# 📚 RAG-Based Solution Document Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** powered document assistant built using **Python, Streamlit, ChromaDB, Sentence Transformers, and Ollama**.

The system allows users to upload documents, semantically search their contents, generate grounded AI responses, create multi-document summaries, and generate structured presales reports — all using locally hosted LLMs.

---

# 🚀 Features

## 📂 Multi-Format Document Ingestion

Supports dynamic upload and processing of:

- PDF
- DOCX
- TXT
- Markdown
- CSV
- XLSX
- PPTX
- HTML
- Images (OCR enabled)

---

## ✂️ Intelligent Recursive Chunking

Documents are recursively split into context-preserving chunks with configurable:

- Chunk size
- Chunk overlap
- Separator hierarchy

---

## 🧠 Semantic Embeddings

Uses Sentence Transformers (`all-MiniLM-L6-v2`) for local embedding generation.

---

## 🗄️ Vector Database with ChromaDB

Stores chunk embeddings and metadata persistently using ChromaDB.

---

## 🔍 Semantic Retrieval Pipeline

Retrieves top-k relevant chunks using:

- Dense vector similarity
- Similarity thresholding
- Deduplication logic

---

## 🤖 Local LLM Integration with Ollama

Integrates locally hosted LLMs using Ollama REST APIs.

Supports:

- Streaming responses
- Timeout handling
- Health checks
- Model listing

---

## 🧾 Grounded Prompt Engineering

Custom prompt templates enforce:

- Strict grounding
- No hallucinations
- Citation-aware responses
- Structured answer formatting

---

## 🧠 Multi-Document Summarization

Generate:

- Concise summaries
- Detailed structured summaries
- Bullet-point summaries
- Cross-document synthesis

---

## 📊 Presales Report Generator

AI-generated structured reports including:

- Executive Summaries
- Competitive Analysis
- Requirements Extraction
- Risk Assessment
- Proposal Outlines
- Custom Business Analysis

---

## 💬 Multi-Session Chat System

Supports:

- Multiple named chats
- Chat switching
- Session persistence
- Document pinning per chat

---

## 🎨 Streamlit User Interface

Interactive UI includes:

- File uploads
- Chat interface
- Summarization tab
- Report generation tab
- Downloadable reports
- Chunk configuration controls
- LLM health monitoring

---

# 🏗️ System Architecture

```text
                ┌────────────────────┐
                │   User Uploads     │
                └─────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │  Document Loader    │
               └─────────┬───────────┘
                         │
                         ▼
               ┌─────────────────────┐
               │  Document Chunker   │
               └─────────┬───────────┘
                         │
                         ▼
               ┌─────────────────────┐
               │ Embedding Engine    │
               └─────────┬───────────┘
                         │
                         ▼
               ┌─────────────────────┐
               │   ChromaDB Store    │
               └─────────┬───────────┘
                         │
         ┌───────────────┴──────────────┐
         ▼                              ▼
 ┌─────────────────┐          ┌─────────────────┐
 │   Retriever     │          │ Document Summary│
 └────────┬────────┘          └────────┬────────┘
          │                             │
          ▼                             ▼
 ┌─────────────────┐          ┌─────────────────┐
 │ Prompt Builder  │          │ Report Builder  │
 └────────┬────────┘          └────────┬────────┘
          │                             │
          └──────────────┬──────────────┘
                         ▼
               ┌─────────────────────┐
               │   Ollama LLM        │
               └─────────┬───────────┘
                         ▼
                 AI Generated Output
```

---

# 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Frontend | Streamlit |
| Vector DB | ChromaDB |
| Embeddings | Sentence Transformers |
| LLM Hosting | Ollama |
| Models | Phi3 / Mistral |
| OCR | Tesseract OCR |
| Document Parsing | PyPDF, python-docx, pandas |
| Language | Python |

---

# 📦 Installation

## 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd rag-document-assistant
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Install Ollama

Download and install Ollama:

https://ollama.com

Run Ollama server:

```bash
ollama serve
```

Pull the model:

```bash
ollama pull phi3
```

---

# ▶️ Running the Application

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```text
project/
│
├── app.py
├── config.py
├── requirements.txt
│
├── modules/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm_handler.py
│   ├── prompt_builder.py
│   ├── rag_pipeline.py
│   ├── summarizer.py
│   ├── report_pipeline.py
│   ├── report_prompt_builder.py
│   └── chat_manager.py
│
├── chroma_db/
└── uploads_temp/
```

---

# ⚙️ Configuration

All tunable settings are centralized in `config.py`:

- Chunk size
- Chunk overlap
- Embedding model
- LLM model
- Retrieval parameters
- Context limits
- Supported file types

---

# 📌 Key Functionalities

## 1️⃣ Document Question Answering

Ask natural language questions and receive:

- Context-aware answers
- Source citations
- Relevant chunk references

---

## 2️⃣ Multi-Document Summarization

Summarize:

- Single documents
- Multiple documents
- Entire knowledge bases

---

## 3️⃣ Presales Intelligence Reports

Generate structured AI reports directly from uploaded business documents.

---

## 4️⃣ OCR-Based Image Understanding

Extracts text from:

- PNG
- JPG
- TIFF
- BMP

using Tesseract OCR.

---

# 🔒 Grounded AI Design

The system is intentionally designed to minimize hallucinations by:

- Restricting prompts to retrieved context
- Using citation-aware prompts
- Returning fallback responses when context is missing

---

# 📈 Future Enhancements

- Hybrid search (BM25 + vector search)
- GPU acceleration
- User authentication
- Persistent chat storage
- Agentic workflows
- LangGraph integration
- Streaming UI responses
- Cloud deployment
- Fine-tuned domain models

---

# 👩‍💻 Author

## Anna Binoy

AI & Software Developer

---

# 📄 License

This project is for educational, research, and portfolio purposes.

