"""
RLG Engine - Retrieval-Locked Generation
A grounded Q&A system that's better than RAG
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import init_db
from app.api import documents, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    print("🚀 Starting RLG Engine...")
    init_db()
    print("✓ Database initialized")
    
    # Check Ollama
    from app.services.llm_service import llm_service
    if llm_service.is_available():
        models = llm_service.get_available_models()
        print(f"✓ Ollama connected. Available models: {models}")
    else:
        print("⚠ Ollama not available. Start with: ollama serve")
    
    yield
    
    # Shutdown
    print("👋 Shutting down RLG Engine...")


app = FastAPI(
    title="RLG Engine",
    description="""
    # Retrieval-Locked Generation Engine
    
    A grounded Q&A system that achieves near-zero hallucination through:
    
    - **Multi-stage retrieval**: BM25 + Dense + Structural signals
    - **Citation-first answers**: Every claim links to source
    - **Grounding validation**: Rejects unverified responses
    - **100% offline**: Runs on local LLM (Ollama) and local embeddings
    
    ## Better than RAG
    
    Standard RAG retrieves and generates. RLG retrieves, generates, AND validates.
    Every sentence in the response is checked against sources.
    
    ## Quick Start
    
    1. Start Ollama: `ollama serve`
    2. Pull a model: `ollama pull mistral`
    3. Upload documents: POST /documents/upload
    4. Ask questions: POST /query/
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router, prefix="/documents")
app.include_router(query.router, prefix="/query")


@app.get("/")
async def root():
    """Health check and system info"""
    from app.services.llm_service import llm_service
    from app.services.vector_index_service import vector_index
    
    return {
        "name": "RLG Engine",
        "version": settings.APP_VERSION,
        "status": "running",
        "llm_available": llm_service.is_available(),
        "llm_model": settings.OLLAMA_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "vector_index": vector_index.get_stats()
    }


@app.get("/health")
async def health():
    """Simple health check"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
