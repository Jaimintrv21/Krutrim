"""
Document model - represents uploaded source files
"""
from sqlalchemy import Column, String, Text, Integer, DateTime, Enum, Float
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.core.database import Base


class DocumentStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentType(enum.Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    XLSX = "xlsx"
    MARKDOWN = "md"
    IMAGE = "image"  # For OCR


class Document(Base):
    """
    Represents a source document in the knowledge base.
    Tracks processing status and metadata for retrieval.
    """
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # File info
    filename = Column(String(512), nullable=False)
    filepath = Column(String(1024), nullable=False)
    file_type = Column(Enum(DocumentType), nullable=False)
    file_size = Column(Integer)  # bytes
    file_hash = Column(String(64))  # SHA-256 for deduplication
    
    # Processing status
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING)
    error_message = Column(Text, nullable=True)
    
    # Metadata for retrieval boost
    title = Column(String(512), nullable=True)
    author = Column(String(256), nullable=True)
    source_url = Column(String(1024), nullable=True)
    category = Column(String(128), nullable=True)
    tags = Column(Text, nullable=True)  # JSON array
    
    # Quality signals
    reliability_score = Column(Float, default=1.0)  # 0-1, for ranking boost
    
    # Processing metadata
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    indexed_at = Column(DateTime, nullable=True)
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status={self.status})>"
