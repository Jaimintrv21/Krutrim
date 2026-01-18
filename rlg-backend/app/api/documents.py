"""
Documents API - Upload, manage, and search documents
"""
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List
from pathlib import Path
import shutil

from app.core.database import get_db_dependency
from app.core.config import settings
from app.models.document import Document, DocumentStatus
from app.models.chunk import Chunk
from app.services.ingestion_service import IngestionService
from app.services.vector_index_service import vector_index
from app.schemas.document import (
    DocumentResponse, 
    DocumentListResponse, 
    DocumentUploadRequest,
    ChunkResponse
)


router = APIRouter(tags=["Documents"])


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # Comma-separated
    reliability_score: float = Form(1.0),
    db: Session = Depends(get_db_dependency)
):
    """
    Upload and ingest a document.
    Supports PDF, DOCX, TXT, HTML, Markdown, and images (OCR).
    """
    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".html", ".htm", 
                          ".md", ".xlsx", ".xls", ".png", ".jpg", ".jpeg"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {ext}. Allowed: {allowed_extensions}"
        )
    
    # Save file
    upload_path = settings.UPLOAD_DIR / file.filename
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Prepare metadata
    metadata = {
        "title": title or file.filename,
        "author": author,
        "category": category,
        "tags": tags,
        "reliability_score": max(0.0, min(1.0, reliability_score))
    }
    
    # Ingest document
    try:
        ingestion_service = IngestionService(db)
        document = ingestion_service.ingest_file(upload_path, metadata)
        
        # Add chunks to vector index
        if document.status == DocumentStatus.INDEXED:
            chunks = db.query(Chunk).filter(Chunk.document_id == document.id).all()
            chunk_ids = [c.id for c in chunks]
            contents = [c.content for c in chunks]
            vector_index.add_chunks(chunk_ids, contents)
            
            # Update FTS index
            for chunk in chunks:
                db.execute(
                    text("INSERT INTO chunks_fts (content, chunk_id) VALUES (:content, :chunk_id)"),
                    {"content": chunk.content, "chunk_id": chunk.id}
                )
            db.commit()
        
        return DocumentResponse(
            id=document.id,
            filename=document.filename,
            file_type=document.file_type.value,
            status=document.status.value,
            title=document.title,
            author=document.author,
            category=document.category,
            tags=document.tags.split(",") if document.tags else None,
            reliability_score=document.reliability_score,
            page_count=document.page_count,
            chunk_count=document.chunk_count,
            created_at=document.created_at,
            indexed_at=document.indexed_at,
            error_message=document.error_message
        )
    
    except Exception as e:
        # Clean up file on failure
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db_dependency)
):
    """List all documents with pagination and filtering"""
    query = db.query(Document)
    
    if status:
        query = query.filter(Document.status == status)
    if category:
        query = query.filter(Document.category == category)
    
    total = query.count()
    documents = query.offset((page - 1) * page_size).limit(page_size).all()
    
    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=d.id,
                filename=d.filename,
                file_type=d.file_type.value,
                status=d.status.value,
                title=d.title,
                author=d.author,
                category=d.category,
                tags=d.tags.split(",") if d.tags else None,
                reliability_score=d.reliability_score,
                page_count=d.page_count,
                chunk_count=d.chunk_count,
                created_at=d.created_at,
                indexed_at=d.indexed_at
            ) for d in documents
        ],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    db: Session = Depends(get_db_dependency)
):
    """Get a single document by ID"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        file_type=document.file_type.value,
        status=document.status.value,
        title=document.title,
        author=document.author,
        category=document.category,
        tags=document.tags.split(",") if document.tags else None,
        reliability_score=document.reliability_score,
        page_count=document.page_count,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        indexed_at=document.indexed_at,
        error_message=document.error_message
    )


@router.get("/{document_id}/chunks", response_model=List[ChunkResponse])
async def get_document_chunks(
    document_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db_dependency)
):
    """Get chunks from a document"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunks = db.query(Chunk).filter(
        Chunk.document_id == document_id
    ).order_by(
        Chunk.sequence_index
    ).offset(
        (page - 1) * page_size
    ).limit(page_size).all()
    
    return [
        ChunkResponse(
            id=c.id,
            content=c.content,
            page_number=c.page_number,
            section_title=c.section_title,
            chunk_type=c.chunk_type,
            confidence_weight=c.confidence_weight,
            citation=c.get_citation()
        ) for c in chunks
    ]


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db_dependency)
):
    """Delete a document and its chunks"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get chunk IDs for vector index cleanup
    chunk_ids = [c.id for c in db.query(Chunk.id).filter(Chunk.document_id == document_id).all()]
    
    # Remove from vector index
    if chunk_ids:
        vector_index.remove_chunks(chunk_ids)
        
        # Remove from FTS
        for cid in chunk_ids:
            db.execute(text("DELETE FROM chunks_fts WHERE chunk_id = :chunk_id"), {"chunk_id": cid})
    
    # Delete document (cascades to chunks)
    db.delete(document)
    db.commit()
    
    # Clean up file
    filepath = Path(document.filepath)
    filepath.unlink(missing_ok=True)
    
    return {"status": "deleted", "document_id": document_id}


@router.post("/{document_id}/reindex")
async def reindex_document(
    document_id: str,
    db: Session = Depends(get_db_dependency)
):
    """Re-ingest a document (useful after updating ingestion logic)"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete existing chunks
    chunk_ids = [c.id for c in db.query(Chunk.id).filter(Chunk.document_id == document_id).all()]
    if chunk_ids:
        vector_index.remove_chunks(chunk_ids)
        db.query(Chunk).filter(Chunk.document_id == document_id).delete()
        db.commit()
    
    # Re-ingest
    ingestion_service = IngestionService(db)
    filepath = Path(document.filepath)
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Source file no longer exists")
    
    # Process document again
    metadata = {
        "title": document.title,
        "author": document.author,
        "category": document.category,
        "tags": document.tags,
        "reliability_score": document.reliability_score
    }
    
    try:
        chunks = ingestion_service._process_document(document, filepath)
        for chunk in chunks:
            chunk.document_id = document.id
            db.add(chunk)
        
        document.chunk_count = len(chunks)
        document.status = DocumentStatus.INDEXED
        db.commit()
        
        # Update indices
        chunk_ids = [c.id for c in chunks]
        contents = [c.content for c in chunks]
        vector_index.add_chunks(chunk_ids, contents)
        
        return {"status": "reindexed", "chunk_count": len(chunks)}
    
    except Exception as e:
        document.status = DocumentStatus.FAILED
        document.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")
