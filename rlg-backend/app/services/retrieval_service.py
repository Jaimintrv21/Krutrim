"""
Retrieval Service - Multi-stage retrieval (BM25 + Dense + Structural)
This is the KEY improvement over standard RAG
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import text as sql_text

from app.core.config import settings
from app.core.database import get_db
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.vector_index_service import vector_index
from app.services.embedding_service import embedding_service


@dataclass
class RetrievedChunk:
    """A retrieved chunk with scoring metadata"""
    chunk_id: str
    content: str
    document_id: str
    document_name: str
    page_number: Optional[int]
    section_title: Optional[str]
    chunk_type: str
    
    # Scoring breakdown
    bm25_score: float = 0.0
    dense_score: float = 0.0
    structural_score: float = 0.0
    final_score: float = 0.0
    
    # For grounding
    confidence_weight: float = 1.0


class RetrievalService:
    """
    Multi-stage retrieval that combines:
    1. BM25 (keyword matching via SQLite FTS5)
    2. Dense retrieval (semantic via FAISS)
    3. Structural signals (headings, sections)
    
    This outperforms single-vector RAG significantly.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def retrieve(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        min_reliability: float = 0.5,
        top_k: int = 10
    ) -> List[RetrievedChunk]:
        """
        Main retrieval entry point.
        Returns top-K chunks with combined scoring.
        """
        # Step 1: Query expansion (extract key terms)
        query_terms = self._extract_query_terms(query)
        
        # Step 2: BM25 keyword search (fast filtering)
        bm25_results = self._bm25_search(query_terms, limit=settings.TOP_K_RETRIEVAL * 2)
        
        # Step 3: Dense semantic search
        dense_results = vector_index.search(query, top_k=settings.TOP_K_RETRIEVAL * 2)
        
        # Step 4: Merge and score candidates
        candidates = self._merge_results(bm25_results, dense_results)
        
        # Step 5: Filter by document/category constraints
        if document_ids or categories or min_reliability > 0:
            candidates = self._filter_candidates(
                candidates, document_ids, categories, min_reliability
            )
        
        # Step 6: Structural re-ranking (boost headings, exact matches)
        results = self._structural_rerank(candidates, query, query_terms)
        
        # Step 7: Return top-K
        return sorted(results, key=lambda x: x.final_score, reverse=True)[:top_k]
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract important terms from query for BM25"""
        # Remove stopwords and extract key terms
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "what", "how", 
                     "when", "where", "why", "who", "which", "can", "could", "would",
                     "should", "do", "does", "did", "have", "has", "had", "be", "been",
                     "being", "for", "to", "of", "in", "on", "at", "by", "with"}
        
        words = re.findall(r'\b\w+\b', query.lower())
        terms = [w for w in words if w not in stopwords and len(w) > 2]
        return terms
    
    def _bm25_search(
        self, 
        query_terms: List[str], 
        limit: int = 50
    ) -> Dict[str, float]:
        """
        BM25 search using SQLite FTS5.
        Returns dict of chunk_id -> BM25 score.
        """
        if not query_terms:
            return {}
        
        # Build FTS5 query
        fts_query = " OR ".join(query_terms)
        
        try:
            result = self.db.execute(sql_text("""
                SELECT chunk_id, bm25(chunks_fts) as score
                FROM chunks_fts
                WHERE chunks_fts MATCH :query
                ORDER BY score
                LIMIT :limit
            """), {"query": fts_query, "limit": limit})
            
            return {row.chunk_id: abs(row.score) for row in result}
        except Exception as e:
            # FTS table might not exist or be populated
            print(f"BM25 search error: {e}")
            return {}
    
    def _merge_results(
        self,
        bm25_results: Dict[str, float],
        dense_results: List[Tuple[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Merge BM25 and dense results with normalized scores.
        Returns dict of chunk_id -> {bm25_score, dense_score}
        """
        candidates = defaultdict(lambda: {"bm25_score": 0.0, "dense_score": 0.0})
        
        # Normalize BM25 scores
        if bm25_results:
            max_bm25 = max(bm25_results.values())
            for chunk_id, score in bm25_results.items():
                candidates[chunk_id]["bm25_score"] = score / max_bm25 if max_bm25 > 0 else 0
        
        # Dense scores are already 0-1 (cosine similarity)
        for chunk_id, score in dense_results:
            candidates[chunk_id]["dense_score"] = score
        
        return dict(candidates)
    
    def _filter_candidates(
        self,
        candidates: Dict[str, Dict[str, float]],
        document_ids: Optional[List[str]],
        categories: Optional[List[str]],
        min_reliability: float
    ) -> Dict[str, Dict[str, float]]:
        """Filter candidates by document/category constraints"""
        if not candidates:
            return {}
        
        # Get chunk metadata
        chunk_ids = list(candidates.keys())
        chunks = self.db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
        
        doc_ids = {c.document_id for c in chunks}
        docs = {d.id: d for d in self.db.query(Document).filter(Document.id.in_(doc_ids)).all()}
        
        filtered = {}
        for chunk in chunks:
            doc = docs.get(chunk.document_id)
            if not doc:
                continue
            
            # Check constraints
            if document_ids and doc.id not in document_ids:
                continue
            if categories and doc.category not in categories:
                continue
            if doc.reliability_score < min_reliability:
                continue
            
            filtered[chunk.id] = candidates[chunk.id]
        
        return filtered
    
    def _structural_rerank(
        self,
        candidates: Dict[str, Dict[str, float]],
        query: str,
        query_terms: List[str]
    ) -> List[RetrievedChunk]:
        """
        Apply structural signals and compute final scores.
        Boosts:
        - Headings that match query terms
        - Exact phrase matches
        - Higher confidence weights
        """
        if not candidates:
            return []
        
        chunk_ids = list(candidates.keys())
        chunks = self.db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
        
        # Get document info
        doc_ids = {c.document_id for c in chunks}
        docs = {d.id: d for d in self.db.query(Document).filter(Document.id.in_(doc_ids)).all()}
        
        results = []
        query_lower = query.lower()
        
        for chunk in chunks:
            scores = candidates.get(chunk.id, {})
            doc = docs.get(chunk.document_id)
            
            # Compute structural score
            structural_score = 0.0
            content_lower = chunk.content.lower()
            
            # Boost for exact query match
            if query_lower in content_lower:
                structural_score += 0.5
            
            # Boost for term coverage
            term_coverage = sum(1 for t in query_terms if t in content_lower) / max(len(query_terms), 1)
            structural_score += term_coverage * 0.3
            
            # Boost headings
            if chunk.chunk_type == "heading":
                structural_score += 0.2
            
            # Apply confidence weight from document
            doc_boost = (doc.reliability_score if doc else 1.0) * chunk.confidence_weight
            
            # Compute final weighted score
            final_score = (
                settings.BM25_WEIGHT * scores.get("bm25_score", 0) +
                settings.DENSE_WEIGHT * scores.get("dense_score", 0) +
                settings.STRUCTURAL_WEIGHT * structural_score
            ) * doc_boost
            
            results.append(RetrievedChunk(
                chunk_id=chunk.id,
                content=chunk.content,
                document_id=chunk.document_id,
                document_name=doc.filename if doc else "Unknown",
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                chunk_type=chunk.chunk_type,
                bm25_score=scores.get("bm25_score", 0),
                dense_score=scores.get("dense_score", 0),
                structural_score=structural_score,
                final_score=final_score,
                confidence_weight=chunk.confidence_weight
            ))
        
        return results
    
    def get_context_window(
        self,
        chunk_id: str,
        window_size: int = 2
    ) -> List[Chunk]:
        """
        Get surrounding chunks for context expansion.
        Useful for answers that span multiple chunks.
        """
        target = self.db.query(Chunk).filter(Chunk.id == chunk_id).first()
        if not target:
            return []
        
        # Get adjacent chunks by sequence index
        chunks = self.db.query(Chunk).filter(
            Chunk.document_id == target.document_id,
            Chunk.sequence_index.between(
                target.sequence_index - window_size,
                target.sequence_index + window_size
            )
        ).order_by(Chunk.sequence_index).all()
        
        return chunks
