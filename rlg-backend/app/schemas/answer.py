"""
Answer schemas - structured response with full provenance
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SourceCitation(BaseModel):
    """A citation to a source chunk"""
    chunk_id: str
    document_name: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    excerpt: str  # The exact text that supports the claim
    relevance_score: float
    match_type: str  # exact, paraphrase, inferred


class GroundedSentence(BaseModel):
    """A sentence with its grounding evidence"""
    text: str
    citations: List[SourceCitation]
    confidence: float
    is_grounded: bool


class AnswerResponse(BaseModel):
    """
    The main response to a query.
    Every sentence links back to source evidence.
    """
    # The answer
    answer: str
    
    # Sentence-level grounding (key differentiator from RAG)
    grounded_sentences: List[GroundedSentence]
    
    # Overall confidence metrics
    overall_confidence: float = Field(ge=0.0, le=1.0)
    grounding_score: float = Field(ge=0.0, le=1.0)
    
    # Source summary
    sources_used: List[SourceCitation]
    total_sources_retrieved: int
    
    # Status
    is_grounded: bool
    warning: Optional[str] = None  # e.g., "Low confidence answer"
    
    # Metadata
    query_id: str
    processing_time_ms: int
    model_used: str


class NoAnswerResponse(BaseModel):
    """Response when no grounded answer can be generated"""
    status: str = "no_grounded_answer"
    reason: str
    suggestions: List[str]  # How to reformulate the query
    partial_info: Optional[str] = None  # Any relevant but incomplete info
    sources_checked: int


class AnswerWithEvidence(BaseModel):
    """Extended answer format with full evidence chain"""
    answer: AnswerResponse
    evidence_chain: List[dict]  # Full reasoning trace
    retrieval_debug: dict  # Which chunks were considered and why
