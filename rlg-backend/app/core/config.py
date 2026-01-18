"""
RLG Configuration - All settings for offline-first operation
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Central configuration for RLG Engine.
    All paths are relative to the data directory for portability.
    """
    
    # Application
    APP_NAME: str = "RLG Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Paths (offline-first, all local)
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    INDEX_DIR: Path = DATA_DIR / "indices"
    CACHE_DIR: Path = DATA_DIR / "cache"
    
    # Database (SQLite for offline)
    DATABASE_URL: str = f"sqlite:///{DATA_DIR}/rlg.db"
    
    # Local Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Local LLM via Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"  # or llama3, phi3, etc.
    OLLAMA_TIMEOUT: int = 120
    
    # Retrieval Settings
    BM25_WEIGHT: float = 0.3
    DENSE_WEIGHT: float = 0.5
    STRUCTURAL_WEIGHT: float = 0.2
    
    TOP_K_RETRIEVAL: int = 20  # Initial retrieval
    TOP_K_RERANK: int = 5      # After reranking
    
    # Chunking Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Grounding Settings
    MIN_GROUNDING_CONFIDENCE: float = 0.7
    REQUIRE_EXACT_CITATION: bool = True
    MAX_GENERATION_TOKENS: int = 1024
    
    # OCR Settings
    TESSERACT_CMD: Optional[str] = None  # Auto-detect
    OCR_LANGUAGE: str = "eng"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def setup_directories(self):
        """Create necessary directories on startup"""
        for dir_path in [self.DATA_DIR, self.UPLOAD_DIR, self.INDEX_DIR, self.CACHE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.setup_directories()
