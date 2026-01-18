"""
Query API - The main Q&A endpoint with grounded responses
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, List, AsyncGenerator
from datetime import datetime
import time
import json

from app.core.database import get_db_dependency
from app.core.config import settings
from app.models.query import Query
from app.models.answer import Answer
from app.schemas.query import QueryRequest
from app.schemas.answer import (
    AnswerResponse, 
    NoAnswerResponse, 
    SourceCitation, 
    GroundedSentence,
    AnswerWithEvidence
)
from app.services.retrieval_service import RetrievalService
from app.services.context_service import ContextService
from app.services.llm_service import llm_service, extractive_llm
from app.services.validation_service import validation_service


router = APIRouter(tags=["Query"])


@router.post("/", response_model=AnswerResponse | NoAnswerResponse)
async def ask_question(
    request: QueryRequest,
    db: Session = Depends(get_db_dependency)
):
    """
    Ask a question and get a grounded answer.
    
    The RLG pipeline:
    1. Multi-stage retrieval (BM25 + Dense + Structural)
    2. Context building with citations
    3. LLM generation with grounding constraints
    4. Validation to reject hallucinations
    """
    start_time = time.time()
    
    # Create query record for analytics
    query_record = Query(
        question=request.question,
        query_type="question"
    )
    db.add(query_record)
    db.flush()
    
    try:
        # Step 1: Retrieval
        retrieval_start = time.time()
        retrieval_service = RetrievalService(db)
        chunks = retrieval_service.retrieve(
            query=request.question,
            document_ids=request.document_ids,
            categories=request.categories,
            min_reliability=request.min_reliability,
            top_k=request.top_k * 2  # Retrieve more, filter later
        )
        retrieval_time = int((time.time() - retrieval_start) * 1000)
        
        if not chunks:
            return NoAnswerResponse(
                reason="No relevant documents found",
                suggestions=[
                    "Try rephrasing your question",
                    "Upload relevant documents first",
                    "Broaden your search categories"
                ],
                sources_checked=0
            )
        
        query_record.chunks_retrieved = len(chunks)
        query_record.retrieval_time_ms = retrieval_time
        
        # Step 2: Build context with citations
        context_service = ContextService(max_tokens=4000)
        context, context_chunks = context_service.build_context(chunks, request.question)
        
        if not context_chunks:
            return NoAnswerResponse(
                reason="Retrieved content too short or irrelevant",
                suggestions=["Provide more detailed documents"],
                sources_checked=len(chunks)
            )
        
        # Step 3: Generate answer with grounding prompt
        generation_start = time.time()
        prompt = context_service.build_grounded_prompt(
            context=context,
            question=request.question,
            context_chunks=context_chunks
        )
        
        llm_response = llm_service.generate(
            prompt=prompt,
            temperature=0.1,  # Low temperature for factual responses
            max_tokens=settings.MAX_GENERATION_TOKENS
        )
        generation_time = int((time.time() - generation_start) * 1000)
        
        query_record.generation_time_ms = generation_time
        query_record.tokens_used = llm_response.tokens_used
        
        # Step 4: Validate grounding
        validation_result = validation_service.validate_answer(
            answer=llm_response.text,
            context_chunks=context_chunks
        )
        
        query_record.grounding_score = validation_result.grounding_score
        query_record.is_grounded = validation_result.is_valid
        
        # Reject if ungrounded (when required)
        if request.require_grounding:
            should_reject, rejection_reason = validation_service.reject_if_ungrounded(
                validation_result
            )
            if should_reject:
                return NoAnswerResponse(
                    reason=rejection_reason or "Answer failed grounding validation",
                    suggestions=[
                        "The sources may not contain this information",
                        "Try asking a more specific question"
                    ],
                    partial_info=llm_response.text[:200] + "..." if len(llm_response.text) > 200 else llm_response.text,
                    sources_checked=len(chunks)
                )
        
        # Build response with grounded sentences
        grounded_sentences = []
        for result in validation_result.sentence_results:
            citations = []
            for chunk_id, excerpt in zip(result.matched_chunks, result.matched_excerpts):
                # Find the context chunk
                matching_context = next(
                    (c for c in context_chunks if c.chunk_id == chunk_id), 
                    None
                )
                if matching_context:
                    citations.append(SourceCitation(
                        chunk_id=chunk_id,
                        document_name=matching_context.citation.split("|")[0].strip(),
                        excerpt=excerpt,
                        relevance_score=result.confidence,
                        match_type=result.match_type
                    ))
            
            grounded_sentences.append(GroundedSentence(
                text=result.sentence,
                citations=citations,
                confidence=result.confidence,
                is_grounded=result.is_grounded
            ))
        
        # Build source citations
        sources_used = []
        for chunk in context_chunks[:request.top_k]:
            retrieved = next((c for c in chunks if c.chunk_id == chunk.chunk_id), None)
            sources_used.append(SourceCitation(
                chunk_id=chunk.chunk_id,
                document_name=chunk.citation.split("|")[0].strip() if "|" in chunk.citation else chunk.citation,
                page_number=retrieved.page_number if retrieved else None,
                section=retrieved.section_title if retrieved else None,
                excerpt=chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content,
                relevance_score=retrieved.final_score if retrieved else 0.0,
                match_type="direct"
            ))
        
        query_record.chunks_used = len(sources_used)
        
        # Store answer
        answer_record = Answer(
            query_id=query_record.id,
            answer_text=llm_response.text,
            source_chunks=json.dumps([{
                "chunk_id": c.chunk_id,
                "citation": c.citation
            } for c in context_chunks]),
            overall_confidence=validation_result.grounding_score,
            grounding_confidence=validation_result.grounding_score,
            is_valid=validation_result.is_valid,
            model_used=llm_response.model
        )
        db.add(answer_record)
        db.commit()
        
        total_time = int((time.time() - start_time) * 1000)
        
        return AnswerResponse(
            answer=llm_response.text,
            grounded_sentences=grounded_sentences,
            overall_confidence=validation_result.grounding_score,
            grounding_score=validation_result.grounding_score,
            sources_used=sources_used,
            total_sources_retrieved=len(chunks),
            is_grounded=validation_result.is_valid,
            warning="; ".join(validation_result.warnings) if validation_result.warnings else None,
            query_id=query_record.id,
            processing_time_ms=total_time,
            model_used=llm_response.model
        )
    
    except ConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e) + " Make sure Ollama is running."
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")


@router.post("/extractive", response_model=AnswerResponse | NoAnswerResponse)
async def ask_extractive(
    request: QueryRequest,
    db: Session = Depends(get_db_dependency)
):
    """
    Extractive Q&A mode - forces direct quotes from sources.
    Even stronger grounding than generative mode.
    """
    start_time = time.time()
    
    # Retrieval
    retrieval_service = RetrievalService(db)
    chunks = retrieval_service.retrieve(
        query=request.question,
        document_ids=request.document_ids,
        categories=request.categories,
        min_reliability=request.min_reliability,
        top_k=request.top_k * 2
    )
    
    if not chunks:
        return NoAnswerResponse(
            reason="No relevant documents found",
            suggestions=["Upload relevant documents"],
            sources_checked=0
        )
    
    # Build context
    context_service = ContextService()
    context, context_chunks = context_service.build_context(chunks, request.question)
    
    # Use extractive LLM
    result = extractive_llm.extract_answer(
        context=context,
        question=request.question,
        context_chunks=context_chunks
    )
    
    if not result["found"]:
        return NoAnswerResponse(
            reason="No extractable answer found in sources",
            suggestions=["The information may not be in the documents"],
            sources_checked=len(chunks)
        )
    
    # Build grounded sentences from verified quotes
    grounded_sentences = []
    for quote in result["quotes"]:
        grounded_sentences.append(GroundedSentence(
            text=f'"{quote["quote"]}"',
            citations=[SourceCitation(
                chunk_id="",
                document_name=quote["source"],
                excerpt=quote["quote"],
                relevance_score=1.0 if quote["verified"] else 0.5,
                match_type="exact" if quote["verified"] else "unverified"
            )],
            confidence=1.0 if quote["verified"] else 0.5,
            is_grounded=quote["verified"]
        ))
    
    all_verified = result["all_verified"]
    total_time = int((time.time() - start_time) * 1000)
    
    return AnswerResponse(
        answer=result["answer"],
        grounded_sentences=grounded_sentences,
        overall_confidence=1.0 if all_verified else 0.7,
        grounding_score=1.0 if all_verified else 0.7,
        sources_used=[],
        total_sources_retrieved=len(chunks),
        is_grounded=all_verified,
        warning=None if all_verified else "Some quotes could not be verified",
        query_id="",
        processing_time_ms=total_time,
        model_used=extractive_llm.model
    )


@router.post("/stream")
async def ask_streaming(
    request: QueryRequest,
    db: Session = Depends(get_db_dependency)
):
    """
    Streaming Q&A - returns tokens as they're generated.
    Note: Validation happens after full response.
    """
    # Retrieval
    retrieval_service = RetrievalService(db)
    chunks = retrieval_service.retrieve(
        query=request.question,
        document_ids=request.document_ids,
        categories=request.categories,
        top_k=request.top_k
    )
    
    if not chunks:
        return {"error": "No relevant documents found"}
    
    # Build context
    context_service = ContextService()
    context, context_chunks = context_service.build_context(chunks, request.question)
    prompt = context_service.build_grounded_prompt(context, request.question, context_chunks)
    
    async def generate():
        full_response = ""
        async for token in llm_service.generate_stream(prompt):
            full_response += token
            yield f"data: {json.dumps({'token': token})}\n\n"
        
        # Validate at end
        validation = validation_service.validate_answer(full_response, context_chunks)
        yield f"data: {json.dumps({'done': True, 'grounding_score': validation.grounding_score})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/stats")
async def query_stats(db: Session = Depends(get_db_dependency)):
    """Get query analytics"""
    total_queries = db.query(Query).count()
    grounded_queries = db.query(Query).filter(Query.is_grounded == True).count()
    avg_grounding = db.query(func.avg(Query.grounding_score)).scalar() or 0
    
    return {
        "total_queries": total_queries,
        "grounded_queries": grounded_queries,
        "grounding_rate": grounded_queries / total_queries if total_queries > 0 else 0,
        "average_grounding_score": float(avg_grounding)
    }
