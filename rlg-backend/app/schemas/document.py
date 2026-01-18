"""
Document schemas for API requests/responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class DocumentStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentTypeEnum(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    XLSX = "xlsx"
    MARKDOWN = "md"
    IMAGE = "image"


class DocumentUploadRequest(BaseModel):
    """Request for uploading a document"""
    title: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    reliability_score: float = Field(default=1.0, ge=0.0, le=1.0)


class DocumentResponse(BaseModel):
    """Response for a single document"""
    id: str
    filename: str
    file_type: DocumentTypeEnum
    status: DocumentStatusEnum
    title: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    reliability_score: float
    page_count: int
    chunk_count: int
    created_at: datetime
    indexed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Paginated list of documents"""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int


class ChunkResponse(BaseModel):
    """Response for a document chunk"""
    id: str
    content: str
    page_number: Optional[int]
    section_title: Optional[str]
    chunk_type: str
    confidence_weight: float
    citation: str
    
    class Config:
        from_attributes = True
