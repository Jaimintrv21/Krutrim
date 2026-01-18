"""
Query schemas for the Q&A API
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class QueryRequest(BaseModel):
    """Request to ask a question"""
    question: str = Field(..., min_length=3, max_length=2000)
    
    # Optional filters
    document_ids: Optional[List[str]] = None  # Restrict to specific documents
    categories: Optional[List[str]] = None    # Filter by category
    min_reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Control parameters
    top_k: int = Field(default=5, ge=1, le=20)
    require_grounding: bool = True  # Reject ungrounded answers
    include_sources: bool = True    # Include source chunks in response


class QueryAnalysis(BaseModel):
    """Analysis of the user's query for retrieval optimization"""
    query_type: str  # factual, procedural, comparative, definition
    key_entities: List[str]
    required_context: List[str]  # What info is needed to answer
    complexity_score: float  # 0-1, affects retrieval depth
