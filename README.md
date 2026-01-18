# RLG Engine - Retrieval-Locked Generation

[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Jaimintrv21/Krutrim)

This repository contains the **Retrieval-Locked Generation (RLG) Engine**, a grounded question-answering system designed for near-zero hallucination and 100% offline operation. It goes beyond traditional Retrieval-Augmented Generation (RAG) by enforcing strict citation verification for every generated claim, ensuring that all answers are verifiably rooted in the provided source documents.

The system comprises a powerful FastAPI backend for the core logic and a lightweight vanilla JavaScript frontend for user interaction.

## What is Retrieval-Locked Generation?

RLG is an advanced Q&A architecture that prioritizes factual accuracy and traceability. Unlike standard RAG systems that retrieve context and then generate an answer, RLG adds a critical final step: **validation**. Every sentence in the generated answer is cross-referenced with the source material to confirm its validity. If a claim cannot be verified, it is either rephrased, marked as ungrounded, or rejected entirely.

### RLG vs. Standard RAG

| Feature | Standard RAG | RLG Engine |
| :--- | :--- | :--- |
| **Retrieval** | Typically single-stage vector search | **Multi-stage retrieval** (BM25 keyword + Dense semantic + Structural) |
| **Generation** | Generate answer based on context | **Generate then validate** against sources |
| **Trust Model** | Trusts the LLM to stay on topic | **Verifies every sentence** against the source material |
| **Dependencies** | Often relies on cloud APIs (OpenAI, etc.) | **100% offline** using local models via Ollama |
| **Traceability** | May provide general source documents | Provides **per-sentence citation links** to exact source excerpts |
| **Hallucination** | Prone to "creative" inaccuracies and hallucination | Designed for **near-zero hallucination** by rejecting unverified claims |

## Architecture

The project is split into two main components:

-   `rlg-backend`: A Python FastAPI application that handles all core logic:
    -   Document ingestion, parsing, and chunking (PDF, DOCX, TXT, etc.).
    -   Local embedding generation using `sentence-transformers`.
    -   Indexing and multi-stage retrieval using FAISS (for vectors) and SQLite FTS5 (for keywords).
    -   Interaction with a local LLM via Ollama for generation.
    -   The crucial grounding validation service to check for hallucinations.
-   `rlg-frontend`: A simple, dependency-free web interface built with HTML, CSS, and JavaScript. It provides a user-friendly way to:
    -   Upload and manage documents.
    -   Ask questions in a chat-like interface.
    -   View analytics on query performance and grounding rates.

## Key Features

-   **100% Offline Operation**: Runs entirely on your local machine. No data ever leaves your system.
-   **Near-Zero Hallucination**: A strict validation service rejects or flags any information that cannot be traced back to the source documents.
-   **Verifiable Answers**: Every answer is returned with detailed citations, including the source document and the exact text that supports the claim.
-   **Multi-Stage Hybrid Retrieval**: Combines keyword (BM25), semantic (vector search), and structural signals for more accurate and relevant context retrieval than vector-only approaches.
-   **Structure-Aware Chunking**: Preserves document structure (headings, lists, tables) during ingestion to provide better-contextualized chunks for retrieval.
-   **Extractive Q&A Mode**: A high-certainty mode that forces the LLM to answer using only direct quotes from the source documents.
-   **Simple Web Interface**: An intuitive UI for uploading documents and interacting with the Q&A system.
-   **Comprehensive REST API**: A well-documented FastAPI backend allows for easy integration into other applications.

## How It Works

The RLG engine follows a four-step process to deliver highly accurate and grounded answers.

#### 1. Ingestion & Indexing
When a document is uploaded, it's processed and broken down into structured chunks. The engine understands headings, paragraphs, and lists to create semantically meaningful segments. Each chunk is then indexed in two ways:
-   A **vector index** (FAISS) is created from text embeddings for semantic search.
-   A **full-text search index** (SQLite FTS5) is created for efficient keyword search (BM25).

#### 2. Multi-Stage Retrieval
When a query is received, the engine retrieves relevant information using a hybrid approach:
1.  **BM25 Search**: Fast keyword matching identifies chunks with high term overlap.
2.  **Dense Retrieval**: Semantic vector search finds conceptually related chunks, even if they don't share keywords.
3.  **Structural Reranking**: The results are merged, and a final reranking algorithm boosts the scores of chunks that are structurally important (e.g., headings) or have high reliability scores.

#### 3. Grounded Generation
The top-ranked chunks are formatted into a context for the LLM. Crucially, each chunk is given a citation marker `[1]`, `[2]`, etc. The LLM is then prompted with strict instructions to answer the user's question *only* using the provided information and to cite its sources using the markers.

#### 4. Validation
This is the "locking" step in Retrieval-Locked Generation. Before returning the answer to the user, the model's response is passed to a validation service. This service:
1.  Splits the generated answer into individual sentences.
2.  For each sentence, it verifies that the claim is supported by the cited source chunk.
3.  It calculates a `grounding_score` based on how many sentences are verifiably backed by evidence.
4.  If the score is below a configurable threshold, the answer is rejected, and a "no answer found" message is returned, preventing hallucinations from reaching the user.

## Getting Started

### Prerequisites

1.  **Python 3.10+**
2.  **Ollama**: For running local language models.
    -   Download from [ollama.ai](https://ollama.ai).
    -   After installing, start the Ollama server:
        ```bash
        ollama serve
        ```
    -   Pull a model to use with the engine (Mistral is a good starting point):
        ```bash
        ollama pull mistral
        ```

### Backend Setup

1.  **Navigate to the backend directory and set up the environment:**
    ```bash
    cd rlg-backend
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure environment variables:**
    Copy the example `.env.example` file to a new file named `.env`.
    ```bash
    cp .env.example .env
    ```
    Modify the `.env` file if your Ollama setup is different or if you wish to use another model.
4.  **Run the backend server:**
    ```bash
    uvicorn app.main:app --reload --port 8000
    ```
    The backend API is now running. You can view the automatically generated API documentation at `http://localhost:8000/docs`.

### Frontend Usage

The frontend is a static web application that communicates with the backend API.

1.  **Open the `index.html` file in your browser:**
    ```bash
    cd rlg-frontend
    # Open the index.html file directly in your web browser (e.g., right-click -> Open with -> Google Chrome)
    ```
2.  **Connect to the backend:**
    -   By default, the frontend will try to connect to the backend at `http://localhost:8000`.
    -   If your backend is running on a different URL, go to the **Settings** page in the UI and update the **API URL**.

You can now use the web interface to upload documents and ask questions.

## Usage Examples (API)

You can also interact with the engine directly via its API.

#### Upload a Document

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@/path/to/your/document.pdf" \
  -F "category=research" \
  -F "reliability_score=0.9"
```

#### Ask a Question

```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main conclusion of the research?",
    "require_grounding": true
  }'
```

## Configuration

Key parameters for the RLG engine can be configured in the `rlg-backend/.env` file:

-   `OLLAMA_HOST`: The URL for your running Ollama instance.
-   `OLLAMA_MODEL`: The default model to use for generation (e.g., `mistral`, `llama3`).
-   `BM25_WEIGHT`, `DENSE_WEIGHT`, `STRUCTURAL_WEIGHT`: The weights for the multi-stage retrieval scoring (should sum to 1.0).
-   `MIN_GROUNDING_CONFIDENCE`: The minimum score (0.0 to 1.0) required for an answer to be considered valid. Default is `0.7`.
-   `CHUNK_SIZE`/`CHUNK_OVERLAP`: Parameters for document chunking.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
