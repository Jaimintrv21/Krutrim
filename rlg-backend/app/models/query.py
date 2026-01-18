"""
Query model - tracks questions for analytics and response improvement
"""
from sqlalchemy import Column, String, Text, Integer, Float, DateTime, Boolean
from datetime import datetime
import uuid

from app.core.database import Base


class Query(Base):
    """
    Tracks user queries for analytics and grounding validation.
    Stores retrieval metadata for continuous improvement.
    """
    __tablename__ = "queries"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Query content
    question = Column(Text, nullable=False)
    question_embedding_hash = Column(String(64))  # For caching similar queries
    
    # Query analysis
    query_type = Column(String(32))  # factual, procedural, comparative, etc.
    detected_entities = Column(Text)  # JSON: entities mentioned in query
    
    # Retrieval metrics
    retrieval_time_ms = Column(Integer)
    chunks_retrieved = Column(Integer)
    chunks_used = Column(Integer)
    
    # Generation metrics
    generation_time_ms = Column(Integer)
    tokens_used = Column(Integer)
    
    # Grounding validation
    grounding_score = Column(Float)  # 0-1, how well answer matches sources
    citations_count = Column(Integer, default=0)
    is_grounded = Column(Boolean, default=False)
    
    # User feedback (optional)
    user_rating = Column(Integer, nullable=True)  # 1-5
    user_feedback = Column(Text, nullable=True)
    
    # Response caching
    cached_response = Column(Text, nullable=True)
    cache_hit = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Query(id={self.id[:8]}, grounded={self.is_grounded})>"
