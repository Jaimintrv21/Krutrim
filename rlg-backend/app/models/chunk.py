"""
Chunk model - represents text segments for retrieval
Enhanced with structural metadata for better grounding
"""
from sqlalchemy import Column, String, Text, Integer, Float, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.core.database import Base


class ChunkType(enum.Enum):
    """Structural type of chunk for context-aware retrieval"""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TABLE_CELL = "table_cell"
    CODE_BLOCK = "code"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    QUOTE = "quote"


class Chunk(Base):
    """
    A text segment from a document, optimized for grounded retrieval.
    Includes structural metadata for citation accuracy.
    """
    __tablename__ = "chunks"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    # Content
    content = Column(Text, nullable=False)
    content_hash = Column(String(64))  # For deduplication
    
    # Position for citation
    page_number = Column(Integer, nullable=True)
    section_title = Column(String(512), nullable=True)  # Parent heading
    paragraph_index = Column(Integer, default=0)
    char_start = Column(Integer)  # Position in original document
    char_end = Column(Integer)
    
    # Structural metadata
    chunk_type = Column(String(32), default=ChunkType.PARAGRAPH.value)
    heading_level = Column(Integer, nullable=True)  # 1-6 for headings
    is_table_header = Column(Boolean, default=False)
    list_depth = Column(Integer, nullable=True)
    
    # Quality signals for ranking
    confidence_weight = Column(Float, default=1.0)
    information_density = Column(Float, default=1.0)  # Computed from term frequency
    
    # Entity extraction cache (for graph-based retrieval)
    entities = Column(Text, nullable=True)  # JSON: [{type, value, positions}]
    
    # Sequence for context window
    sequence_index = Column(Integer, default=0)  # Order in document
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, page={self.page_number}, type={self.chunk_type})>"
    
    def get_citation(self) -> str:
        """Generate a formatted citation for this chunk"""
        parts = []
        if self.document:
            parts.append(self.document.filename)
        if self.page_number:
            parts.append(f"p.{self.page_number}")
        if self.section_title:
            parts.append(f"§{self.section_title}")
        return " | ".join(parts) if parts else f"Chunk {self.id[:8]}"
