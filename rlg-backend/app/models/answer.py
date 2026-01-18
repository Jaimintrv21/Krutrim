"""
Answer model - stores grounded responses with full provenance
"""
from sqlalchemy import Column, String, Text, Float, DateTime, ForeignKey, Boolean, Integer
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.core.database import Base


class Answer(Base):
    """
    Represents a generated answer with full citation provenance.
    Every answer must link back to source chunks.
    """
    __tablename__ = "answers"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query_id = Column(String(36), ForeignKey("queries.id", ondelete="CASCADE"), nullable=False)
    
    # Response content
    answer_text = Column(Text, nullable=False)
    
    # Grounding evidence (JSON array of chunk references)
    source_chunks = Column(Text, nullable=False)  # JSON: [{chunk_id, relevance, excerpt}]
    
    # Confidence metrics
    overall_confidence = Column(Float, default=0.0)  # 0-1
    grounding_confidence = Column(Float, default=0.0)  # How well it matches sources
    coherence_score = Column(Float, default=0.0)  # Internal consistency
    
    # Validation status
    is_valid = Column(Boolean, default=False)
    validation_errors = Column(Text, nullable=True)  # JSON array
    
    # Generation metadata
    model_used = Column(String(64))
    prompt_template = Column(String(64))
    temperature = Column(Float, default=0.1)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Answer(id={self.id[:8]}, confidence={self.overall_confidence:.2f})>"


class AnswerChunkLink(Base):
    """
    Explicit many-to-many linking answer sentences to source chunks.
    Enables sentence-level grounding verification.
    """
    __tablename__ = "answer_chunk_links"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    answer_id = Column(String(36), ForeignKey("answers.id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(String(36), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    
    # Which sentence in the answer
    sentence_index = Column(Integer, default=0)
    sentence_text = Column(Text)
    
    # Grounding evidence
    matched_excerpt = Column(Text)  # Exact text from chunk that supports sentence
    similarity_score = Column(Float)  # How similar the sentence is to source
    match_type = Column(String(32))  # exact, paraphrase, inferred
    
    created_at = Column(DateTime, default=datetime.utcnow)
