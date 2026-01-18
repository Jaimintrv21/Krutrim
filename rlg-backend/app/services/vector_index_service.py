"""
Vector Index Service - FAISS-based local vector search
Maintains persistent indices for fast retrieval
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import pickle
import threading

from app.core.config import settings
from app.services.embedding_service import embedding_service


class VectorIndexService:
    """
    FAISS-based vector index for semantic search.
    Fully offline, persistent to disk.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.index_path = settings.INDEX_DIR / "faiss.index"
        self.mapping_path = settings.INDEX_DIR / "chunk_mapping.pkl"
        
        self.index = None
        self.chunk_ids: List[str] = []  # Maps index position to chunk_id
        
        self._load_or_create_index()
        self._initialized = True
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
        if self.index_path.exists() and self.mapping_path.exists():
            # Load existing
            self.index = faiss.read_index(str(self.index_path))
            with open(self.mapping_path, "rb") as f:
                self.chunk_ids = pickle.load(f)
            print(f"✓ Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            # Create new index
            self.index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)  # Inner product (cosine sim for normalized vectors)
            self.chunk_ids = []
            print("✓ Created new FAISS index")
    
    def add_chunks(self, chunk_ids: List[str], contents: List[str]):
        """Add chunks to the index"""
        if not chunk_ids or not contents:
            return
        
        # Generate embeddings
        embeddings = embedding_service.embed_batch(contents)
        
        # Add to FAISS
        self.index.add(embeddings.astype(np.float32))
        self.chunk_ids.extend(chunk_ids)
        
        # Persist
        self._save_index()
    
    def remove_chunks(self, chunk_ids_to_remove: List[str]):
        """Remove chunks from index (requires rebuild)"""
        import faiss
        
        indices_to_keep = [
            i for i, cid in enumerate(self.chunk_ids) 
            if cid not in chunk_ids_to_remove
        ]
        
        if len(indices_to_keep) == len(self.chunk_ids):
            return  # Nothing to remove
        
        # Rebuild index with remaining vectors
        if indices_to_keep:
            # Get vectors to keep
            vectors = np.array([
                self.index.reconstruct(i) for i in indices_to_keep
            ])
            
            # Create new index
            self.index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)
            self.index.add(vectors)
            self.chunk_ids = [self.chunk_ids[i] for i in indices_to_keep]
        else:
            # Empty index
            self.index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)
            self.chunk_ids = []
        
        self._save_index()
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for similar chunks.
        Returns list of (chunk_id, similarity_score) tuples.
        """
        if self.index.ntotal == 0:
            return []
        
        # Embed query
        query_embedding = embedding_service.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        
        # Map to chunk IDs
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(score)))
        
        return results
    
    def _save_index(self):
        """Persist index to disk"""
        import faiss
        
        faiss.write_index(self.index, str(self.index_path))
        with open(self.mapping_path, "wb") as f:
            pickle.dump(self.chunk_ids, f)
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": settings.EMBEDDING_DIMENSION,
            "index_size_bytes": self.index_path.stat().st_size if self.index_path.exists() else 0
        }


# Singleton instance
vector_index = VectorIndexService()
