"""
Embedding Service - Local embeddings via sentence-transformers
100% offline - no API calls required
"""
import numpy as np
from typing import List, Optional
from pathlib import Path
import hashlib
import json

from app.core.config import settings


class EmbeddingService:
    """
    Local embedding service using sentence-transformers.
    Runs completely offline with cached models.
    """
    
    _instance: Optional["EmbeddingService"] = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the embedding model (lazy initialization)"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load model - will download once, then cached locally
            self._model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                cache_folder=str(settings.CACHE_DIR / "models")
            )
            print(f"✓ Loaded embedding model: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            print(f"✗ Failed to load embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string"""
        if not text.strip():
            return np.zeros(settings.EMBEDDING_DIMENSION)
        
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently"""
        if not texts:
            return np.array([])
        
        # Filter empty texts but keep track of indices
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            return np.zeros((len(texts), settings.EMBEDDING_DIMENSION))
        
        # Batch encode
        embeddings = self._model.encode(
            valid_texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(valid_texts) > 100
        )
        
        # Reconstruct full array with zeros for empty texts
        result = np.zeros((len(texts), settings.EMBEDDING_DIMENSION))
        for i, valid_idx in enumerate(valid_indices):
            result[valid_idx] = embeddings[i]
        
        return result
    
    def compute_similarity(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and documents"""
        # Embeddings are already normalized, so dot product = cosine similarity
        return np.dot(doc_embs, query_emb)
    
    def get_embedding_hash(self, text: str) -> str:
        """Get a hash of the embedding for caching"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]


# Singleton instance
embedding_service = EmbeddingService()
