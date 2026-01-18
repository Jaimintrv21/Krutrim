"""
Ingestion Service - Document processing and chunking
Handles PDF, DOCX, TXT, HTML with structure preservation
"""
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Generator
from datetime import datetime
import json
import re

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document, DocumentStatus, DocumentType
from app.models.chunk import Chunk, ChunkType


class IngestionService:
    """
    Processes documents into chunks while preserving structure.
    Key improvement over basic RAG: structure-aware chunking.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def ingest_file(
        self,
        filepath: Path,
        metadata: Optional[dict] = None
    ) -> Document:
        """
        Main entry point for document ingestion.
        Returns the created Document with its chunks.
        """
        metadata = metadata or {}
        
        # Detect file type
        file_type = self._detect_file_type(filepath)
        
        # Compute file hash for deduplication
        file_hash = self._compute_file_hash(filepath)
        
        # Check for duplicate
        existing = self.db.query(Document).filter(
            Document.file_hash == file_hash
        ).first()
        if existing:
            return existing
        
        # Create document record
        document = Document(
            filename=filepath.name,
            filepath=str(filepath),
            file_type=file_type,
            file_size=filepath.stat().st_size,
            file_hash=file_hash,
            status=DocumentStatus.PROCESSING,
            **metadata
        )
        self.db.add(document)
        self.db.flush()
        
        try:
            # Extract and chunk content
            chunks = self._process_document(document, filepath)
            
            # Store chunks
            for chunk in chunks:
                self.db.add(chunk)
            
            # Update document status
            document.status = DocumentStatus.INDEXED
            document.chunk_count = len(chunks)
            document.indexed_at = datetime.utcnow()
            
            self.db.commit()
            
        except Exception as e:
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)
            self.db.commit()
            raise
        
        return document
    
    def _detect_file_type(self, filepath: Path) -> DocumentType:
        """Detect document type from extension"""
        ext = filepath.suffix.lower()
        mapping = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.DOCX,
            ".doc": DocumentType.DOCX,
            ".txt": DocumentType.TXT,
            ".html": DocumentType.HTML,
            ".htm": DocumentType.HTML,
            ".xlsx": DocumentType.XLSX,
            ".xls": DocumentType.XLSX,
            ".md": DocumentType.MARKDOWN,
            ".png": DocumentType.IMAGE,
            ".jpg": DocumentType.IMAGE,
            ".jpeg": DocumentType.IMAGE,
        }
        return mapping.get(ext, DocumentType.TXT)
    
    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA-256 hash of file content"""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _process_document(self, document: Document, filepath: Path) -> List[Chunk]:
        """Route to appropriate processor based on file type"""
        processors = {
            DocumentType.PDF: self._process_pdf,
            DocumentType.DOCX: self._process_docx,
            DocumentType.TXT: self._process_text,
            DocumentType.HTML: self._process_html,
            DocumentType.MARKDOWN: self._process_markdown,
            DocumentType.IMAGE: self._process_image,
        }
        processor = processors.get(document.file_type, self._process_text)
        return processor(document, filepath)
    
    def _process_pdf(self, document: Document, filepath: Path) -> List[Chunk]:
        """Process PDF with page tracking"""
        from pypdf import PdfReader
        
        chunks = []
        reader = PdfReader(str(filepath))
        document.page_count = len(reader.pages)
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            
            # Structure-aware chunking
            page_chunks = self._chunk_text_with_structure(
                text=text,
                document=document,
                page_number=page_num
            )
            chunks.extend(page_chunks)
        
        return chunks
    
    def _process_docx(self, document: Document, filepath: Path) -> List[Chunk]:
        """Process DOCX with heading structure"""
        from docx import Document as DocxDoc
        
        doc = DocxDoc(str(filepath))
        chunks = []
        current_section = None
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            # Detect headings
            style = para.style.name.lower() if para.style else ""
            if "heading" in style:
                current_section = text
                heading_level = int(re.search(r"\d", style).group()) if re.search(r"\d", style) else 1
                chunk = Chunk(
                    document_id=document.id,
                    content=text,
                    chunk_type=ChunkType.HEADING.value,
                    heading_level=heading_level,
                    section_title=current_section,
                    confidence_weight=1.2  # Boost headings
                )
            else:
                chunk = Chunk(
                    document_id=document.id,
                    content=text,
                    chunk_type=ChunkType.PARAGRAPH.value,
                    section_title=current_section,
                    confidence_weight=1.0
                )
            
            chunks.append(chunk)
        
        return self._merge_small_chunks(chunks)
    
    def _process_text(self, document: Document, filepath: Path) -> List[Chunk]:
        """Process plain text file"""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        return self._chunk_text_with_structure(text, document)
    
    def _process_html(self, document: Document, filepath: Path) -> List[Chunk]:
        """Process HTML with semantic structure"""
        from bs4 import BeautifulSoup
        
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        
        chunks = []
        
        # Extract headings and paragraphs
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
            text = tag.get_text(strip=True)
            if not text:
                continue
            
            if tag.name.startswith("h"):
                level = int(tag.name[1])
                chunk_type = ChunkType.HEADING.value
                weight = 1.2
            elif tag.name == "li":
                chunk_type = ChunkType.LIST_ITEM.value
                level = None
                weight = 1.0
            else:
                chunk_type = ChunkType.PARAGRAPH.value
                level = None
                weight = 1.0
            
            chunk = Chunk(
                document_id=document.id,
                content=text,
                chunk_type=chunk_type,
                heading_level=level,
                confidence_weight=weight
            )
            chunks.append(chunk)
        
        return self._merge_small_chunks(chunks)
    
    def _process_markdown(self, document: Document, filepath: Path) -> List[Chunk]:
        """Process Markdown with header structure"""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        chunks = []
        current_section = None
        
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Detect headers
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                level = len(header_match.group(1))
                text = header_match.group(2)
                current_section = text
                chunk = Chunk(
                    document_id=document.id,
                    content=text,
                    chunk_type=ChunkType.HEADING.value,
                    heading_level=level,
                    section_title=current_section,
                    confidence_weight=1.2
                )
            else:
                chunk = Chunk(
                    document_id=document.id,
                    content=line,
                    chunk_type=ChunkType.PARAGRAPH.value,
                    section_title=current_section,
                    confidence_weight=1.0
                )
            
            chunks.append(chunk)
        
        return self._merge_small_chunks(chunks)
    
    def _process_image(self, document: Document, filepath: Path) -> List[Chunk]:
        """Process image with OCR"""
        try:
            import pytesseract
            from PIL import Image
            
            image = Image.open(filepath)
            text = pytesseract.image_to_string(
                image,
                lang=settings.OCR_LANGUAGE
            )
            
            if not text.strip():
                return []
            
            return self._chunk_text_with_structure(text, document)
        except ImportError:
            raise ImportError("pytesseract and Pillow required for OCR")
    
    def _chunk_text_with_structure(
        self,
        text: str,
        document: Document,
        page_number: Optional[int] = None
    ) -> List[Chunk]:
        """
        Smart chunking that preserves sentence boundaries and semantic units.
        Better than naive fixed-size chunking.
        """
        chunks = []
        
        # Split into paragraphs first
        paragraphs = re.split(r"\n\s*\n", text)
        
        current_chunk = []
        current_length = 0
        sequence_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            # If paragraph itself exceeds chunk size, split by sentences
            if para_length > settings.CHUNK_SIZE:
                # Flush current chunk
                if current_chunk:
                    content = " ".join(current_chunk)
                    chunk = Chunk(
                        document_id=document.id,
                        content=content,
                        page_number=page_number,
                        chunk_type=ChunkType.PARAGRAPH.value,
                        sequence_index=sequence_index,
                        confidence_weight=1.0,
                        content_hash=hashlib.sha256(content.encode()).hexdigest()[:32]
                    )
                    chunks.append(chunk)
                    sequence_index += 1
                    current_chunk = []
                    current_length = 0
                
                # Split large paragraph by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    
                    if current_length + len(sent) > settings.CHUNK_SIZE and current_chunk:
                        content = " ".join(current_chunk)
                        chunk = Chunk(
                            document_id=document.id,
                            content=content,
                            page_number=page_number,
                            chunk_type=ChunkType.PARAGRAPH.value,
                            sequence_index=sequence_index,
                            confidence_weight=1.0,
                            content_hash=hashlib.sha256(content.encode()).hexdigest()[:32]
                        )
                        chunks.append(chunk)
                        sequence_index += 1
                        current_chunk = []
                        current_length = 0
                    
                    current_chunk.append(sent)
                    current_length += len(sent) + 1
            else:
                # Add whole paragraph
                if current_length + para_length > settings.CHUNK_SIZE and current_chunk:
                    content = " ".join(current_chunk)
                    chunk = Chunk(
                        document_id=document.id,
                        content=content,
                        page_number=page_number,
                        chunk_type=ChunkType.PARAGRAPH.value,
                        sequence_index=sequence_index,
                        confidence_weight=1.0,
                        content_hash=hashlib.sha256(content.encode()).hexdigest()[:32]
                    )
                    chunks.append(chunk)
                    sequence_index += 1
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(para)
                current_length += para_length + 1
        
        # Flush remaining
        if current_chunk:
            content = " ".join(current_chunk)
            chunk = Chunk(
                document_id=document.id,
                content=content,
                page_number=page_number,
                chunk_type=ChunkType.PARAGRAPH.value,
                sequence_index=sequence_index,
                confidence_weight=1.0,
                content_hash=hashlib.sha256(content.encode()).hexdigest()[:32]
            )
            chunks.append(chunk)
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[Chunk], min_size: int = 100) -> List[Chunk]:
        """Merge very small chunks with adjacent ones"""
        if len(chunks) < 2:
            return chunks
        
        merged = []
        buffer = None
        
        for chunk in chunks:
            if len(chunk.content) < min_size and chunk.chunk_type != ChunkType.HEADING.value:
                if buffer:
                    buffer.content += " " + chunk.content
                else:
                    buffer = chunk
            else:
                if buffer:
                    if len(buffer.content) < min_size:
                        chunk.content = buffer.content + " " + chunk.content
                    else:
                        merged.append(buffer)
                    buffer = None
                merged.append(chunk)
        
        if buffer:
            if merged:
                merged[-1].content += " " + buffer.content
            else:
                merged.append(buffer)
        
        return merged
