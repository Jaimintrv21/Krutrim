"""
Context Service - Builds optimized context for the LLM
Includes citation markers and structure preservation
"""
from typing import List, Optional
from dataclasses import dataclass
import re

from app.services.retrieval_service import RetrievedChunk
from app.core.config import settings


@dataclass
class ContextChunk:
    """A chunk formatted for LLM context with citation marker"""
    marker: str  # e.g., [1]
    content: str
    citation: str  # Full citation string
    chunk_id: str


class ContextService:
    """
    Builds context window for LLM with:
    - Citation markers for grounding
    - Deduplication
    - Relevance ordering
    - Token budget management
    """
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
    
    def build_context(
        self,
        chunks: List[RetrievedChunk],
        query: str
    ) -> tuple[str, List[ContextChunk]]:
        """
        Build context string with numbered citations.
        Returns (context_string, list of ContextChunks for reference)
        """
        if not chunks:
            return "", []
        
        # Deduplicate by content hash
        seen_content = set()
        unique_chunks = []
        for chunk in chunks:
            content_key = chunk.content[:100]  # First 100 chars as key
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_chunks.append(chunk)
        
        # Build context with citation markers
        context_chunks = []
        context_parts = []
        
        for i, chunk in enumerate(unique_chunks, 1):
            marker = f"[{i}]"
            
            # Build citation string
            citation_parts = [chunk.document_name]
            if chunk.page_number:
                citation_parts.append(f"p.{chunk.page_number}")
            if chunk.section_title:
                citation_parts.append(f"§{chunk.section_title}")
            citation = " | ".join(citation_parts)
            
            # Format for context
            formatted = f"{marker} {chunk.content}"
            
            # Check token budget (rough estimate: 4 chars per token)
            estimated_tokens = sum(len(p) for p in context_parts) // 4
            if estimated_tokens + len(formatted) // 4 > self.max_tokens:
                break
            
            context_parts.append(formatted)
            context_chunks.append(ContextChunk(
                marker=marker,
                content=chunk.content,
                citation=citation,
                chunk_id=chunk.chunk_id
            ))
        
        # Build final context string
        context_header = "REFERENCE SOURCES (use citation markers in your answer):\n\n"
        context_string = context_header + "\n\n".join(context_parts)
        
        return context_string, context_chunks
    
    def build_grounded_prompt(
        self,
        context: str,
        question: str,
        context_chunks: List[ContextChunk]
    ) -> str:
        """
        Build a prompt that enforces grounded responses.
        Key to preventing hallucination.
        """
        citation_guide = "\n".join([
            f"{c.marker} = {c.citation}" for c in context_chunks
        ])
        
        prompt = f"""You are a precise question-answering assistant. Your answers MUST be grounded in the provided sources.

STRICT RULES:
1. ONLY use information from the REFERENCE SOURCES below
2. ALWAYS cite sources using the citation markers [1], [2], etc.
3. If information is not in the sources, say "I cannot find this information in the provided sources"
4. NEVER make up facts or use external knowledge
5. Quote exact phrases when possible to maintain accuracy

{context}

CITATION KEY:
{citation_guide}

QUESTION: {question}

ANSWER (with citations):"""
        
        return prompt
    
    def expand_context_window(
        self,
        chunks: List[RetrievedChunk],
        db,
        window_size: int = 1
    ) -> List[RetrievedChunk]:
        """
        Expand context by including adjacent chunks.
        Useful when answers span multiple chunks.
        """
        from app.models.chunk import Chunk
        
        expanded = []
        seen_ids = set()
        
        for chunk in chunks:
            # Add the main chunk
            if chunk.chunk_id not in seen_ids:
                expanded.append(chunk)
                seen_ids.add(chunk.chunk_id)
            
            # Get adjacent chunks
            adj_chunks = db.query(Chunk).filter(
                Chunk.document_id == chunk.document_id,
                Chunk.sequence_index.between(
                    chunk.sequence_index - window_size if hasattr(chunk, 'sequence_index') else 0,
                    chunk.sequence_index + window_size if hasattr(chunk, 'sequence_index') else 0
                )
            ).all()
            
            for adj in adj_chunks:
                if adj.id not in seen_ids:
                    # Create RetrievedChunk from adjacent
                    expanded.append(RetrievedChunk(
                        chunk_id=adj.id,
                        content=adj.content,
                        document_id=adj.document_id,
                        document_name=chunk.document_name,
                        page_number=adj.page_number,
                        section_title=adj.section_title,
                        chunk_type=adj.chunk_type,
                        final_score=chunk.final_score * 0.5,  # Lower score for adjacent
                        confidence_weight=adj.confidence_weight
                    ))
                    seen_ids.add(adj.id)
        
        # Re-sort by score
        return sorted(expanded, key=lambda x: x.final_score, reverse=True)
