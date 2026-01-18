# RLG Engine - Retrieval-Locked Generation

> **Better than RAG. Totally Offline. Near-Zero Hallucination.**

RLG is a grounded question-answering system that goes beyond traditional RAG (Retrieval-Augmented Generation) by enforcing citation verification at every step.

## 🎯 Key Differences from Standard RAG

| Standard RAG | RLG Engine |
|--------------|------------|
| Vector similarity only | **Multi-stage retrieval** (BM25 + Dense + Structural) |
| Generate then hope | **Generate then validate** |
| Trust LLM output | **Verify every sentence against sources** |
| Cloud API dependency | **100% offline** with Ollama |
| Black-box answers | **Citation links for every claim** |

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** (for local LLM)
   ```bash
   # Windows (via installer)
   # Download from https://ollama.ai
   
   # Start Ollama
   ollama serve
   
   # Pull a model (choose one)
   ollama pull mistral    # 7B, balanced
   ollama pull llama3     # 8B, high quality
   ollama pull phi3       # 3.8B, fast
   ```

### Installation

```bash
cd rlg-backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download embedding model (runs once)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Run the Server

```bash
uvicorn app.main:app --reload --port 8000
```

Visit: http://localhost:8000/docs

## 📁 Project Structure

```
rlg-backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── core/
│   │   ├── config.py        # All settings
│   │   └── database.py      # SQLite + FTS5
│   ├── models/
│   │   ├── document.py      # Document metadata
│   │   ├── chunk.py         # Text chunks with structure
│   │   ├── query.py         # Query tracking
│   │   └── answer.py        # Grounded answers
│   ├── schemas/             # Pydantic models
│   ├── services/
│   │   ├── embedding_service.py    # Local embeddings
│   │   ├── ingestion_service.py    # Document processing
│   │   ├── vector_index_service.py # FAISS vector search
│   │   ├── retrieval_service.py    # Multi-stage retrieval
│   │   ├── context_service.py      # Context building
│   │   ├── llm_service.py          # Ollama integration
│   │   └── validation_service.py   # Grounding verification
│   ├── api/
│   │   ├── documents.py     # Document CRUD
│   │   └── query.py         # Q&A endpoints
│   └── utils/
│       ├── tokenizer.py     # Text processing
│       └── scoring.py       # Ranking metrics
├── data/                    # Local data storage
│   ├── uploads/             # Uploaded documents
│   ├── indices/             # FAISS indices
│   └── cache/               # Model cache
└── requirements.txt
```

## 🔧 API Endpoints

### Documents

```bash
# Upload a document
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@my_document.pdf" \
  -F "category=technical" \
  -F "reliability_score=0.9"

# List documents
curl "http://localhost:8000/documents/"

# Get document chunks
curl "http://localhost:8000/documents/{id}/chunks"
```

### Query

```bash
# Ask a question (with grounding validation)
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "require_grounding": true}'

# Extractive mode (direct quotes only)
curl -X POST "http://localhost:8000/query/extractive" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key features?"}'

# Streaming response
curl -N "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain the architecture"}'
```

## ⚙️ Configuration

Edit `.env` or `app/core/config.py`:

```env
# LLM Settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral

# Retrieval Weights (must sum to 1.0)
BM25_WEIGHT=0.3
DENSE_WEIGHT=0.5
STRUCTURAL_WEIGHT=0.2

# Grounding Threshold
MIN_GROUNDING_CONFIDENCE=0.7
REQUIRE_EXACT_CITATION=true
```

## 🧪 How It Works

### 1. Multi-Stage Retrieval

```
Query → BM25 (keyword) ─┐
      → Dense (semantic) ├→ Merge → Structural Rerank → Top-K
      → Structural ──────┘
```

### 2. Context Building

```
Top-K Chunks → Add Citation Markers [1], [2]...
             → Build Grounded Prompt
             → Include Citation Key
```

### 3. Grounding Validation

For each sentence in the LLM response:
1. Check citation markers exist
2. Verify cited content matches source
3. Compute semantic similarity to sources
4. Mark as grounded/ungrounded

### 4. Response Filtering

```python
if grounding_score < MIN_THRESHOLD:
    return "No grounded answer found"
else:
    return answer_with_citations
```

## 📊 Grounding Score

Every response includes a grounding score (0-1):

- **1.0**: Every sentence verified against sources
- **0.7+**: High confidence, some inferences
- **0.5-0.7**: Moderate grounding
- **<0.5**: Rejected by default

## 🔒 Offline Guarantee

Everything runs locally:
- **Embeddings**: sentence-transformers (downloads model once)
- **Vector Search**: FAISS (local index)
- **Database**: SQLite with FTS5
- **LLM**: Ollama (local)

No data leaves your machine.

## 🛠️ Extending RLG

### Add Custom Document Types

```python
# In ingestion_service.py
def _process_custom(self, document, filepath):
    # Your custom processing
    return chunks
```

### Add Vector Index

Switch from FAISS to other options:
- ChromaDB
- Qdrant
- Milvus

### Add UI Layer

The API is designed for any frontend:
- React/Next.js
- Streamlit
- Gradio

## 📝 License

MIT License - Use freely, attribute kindly.

---

Built for **grounded truth**, not creative fiction. 🎯
