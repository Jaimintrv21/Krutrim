"""
Validation Service - Grounding verification and hallucination detection
THE KEY to near-zero hallucination
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
from difflib import SequenceMatcher

from app.services.context_service import ContextChunk
from app.services.embedding_service import embedding_service
from app.core.config import settings


@dataclass
class GroundingResult:
    """Result of grounding validation for a sentence"""
    sentence: str
    is_grounded: bool
    confidence: float
    matched_chunks: List[str]  # chunk_ids that support this sentence
    matched_excerpts: List[str]  # Exact text that supports
    match_type: str  # exact, paraphrase, inferred, ungrounded


@dataclass
class ValidationResult:
    """Overall validation result for an answer"""
    is_valid: bool
    grounding_score: float  # 0-1, proportion of sentences grounded
    sentence_results: List[GroundingResult]
    warnings: List[str]
    errors: List[str]


class ValidationService:
    """
    Validates that LLM responses are grounded in source material.
    Uses multiple strategies:
    1. Citation checking
    2. Semantic similarity
    3. Exact/fuzzy matching
    4. Claim extraction and verification
    """
    
    def __init__(self):
        self.min_confidence = settings.MIN_GROUNDING_CONFIDENCE
    
    def validate_answer(
        self,
        answer: str,
        context_chunks: List[ContextChunk]
    ) -> ValidationResult:
        """
        Main validation entry point.
        Checks each sentence in the answer for grounding.
        """
        if not answer or not context_chunks:
            return ValidationResult(
                is_valid=False,
                grounding_score=0.0,
                sentence_results=[],
                warnings=["Empty answer or context"],
                errors=[]
            )
        
        # Split answer into sentences
        sentences = self._split_sentences(answer)
        
        # Build content index for fast matching
        content_index = {
            chunk.chunk_id: chunk.content.lower()
            for chunk in context_chunks
        }
        
        # Validate each sentence
        sentence_results = []
        for sentence in sentences:
            result = self._validate_sentence(sentence, context_chunks, content_index)
            sentence_results.append(result)
        
        # Compute overall score
        grounded_count = sum(1 for r in sentence_results if r.is_grounded)
        grounding_score = grounded_count / len(sentence_results) if sentence_results else 0.0
        
        # Determine validity
        is_valid = grounding_score >= self.min_confidence
        
        # Generate warnings
        warnings = []
        errors = []
        
        ungrounded = [r.sentence for r in sentence_results if not r.is_grounded]
        if ungrounded:
            warnings.append(f"{len(ungrounded)} sentence(s) could not be verified")
        
        if grounding_score < 0.5:
            errors.append("Less than 50% of the answer is grounded in sources")
        
        # Check for citation markers
        if not re.search(r'\[\d+\]', answer):
            warnings.append("Answer contains no citation markers")
        
        return ValidationResult(
            is_valid=is_valid,
            grounding_score=grounding_score,
            sentence_results=sentence_results,
            warnings=warnings,
            errors=errors
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Handle common abbreviations
        text = re.sub(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', '\n', text)
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        return sentences
    
    def _validate_sentence(
        self,
        sentence: str,
        context_chunks: List[ContextChunk],
        content_index: Dict[str, str]
    ) -> GroundingResult:
        """
        Validate a single sentence against sources.
        Uses multiple matching strategies.
        """
        sentence_lower = sentence.lower()
        
        # Strategy 1: Check for citation markers and verify
        citations = re.findall(r'\[(\d+)\]', sentence)
        if citations:
            for citation_num in citations:
                idx = int(citation_num) - 1
                if 0 <= idx < len(context_chunks):
                    chunk = context_chunks[idx]
                    # Check if sentence content relates to cited chunk
                    similarity = self._semantic_similarity(
                        sentence, chunk.content
                    )
                    if similarity > 0.5:
                        return GroundingResult(
                            sentence=sentence,
                            is_grounded=True,
                            confidence=similarity,
                            matched_chunks=[chunk.chunk_id],
                            matched_excerpts=[self._find_matching_excerpt(sentence, chunk.content)],
                            match_type="cited"
                        )
        
        # Strategy 2: Exact substring match
        for chunk_id, content in content_index.items():
            # Remove citation markers for matching
            clean_sentence = re.sub(r'\[\d+\]', '', sentence_lower).strip()
            
            if len(clean_sentence) > 20 and clean_sentence in content:
                return GroundingResult(
                    sentence=sentence,
                    is_grounded=True,
                    confidence=1.0,
                    matched_chunks=[chunk_id],
                    matched_excerpts=[clean_sentence],
                    match_type="exact"
                )
        
        # Strategy 3: Fuzzy matching for paraphrases
        best_match_chunk = None
        best_match_score = 0.0
        best_excerpt = ""
        
        for chunk in context_chunks:
            score = self._fuzzy_match_score(sentence, chunk.content)
            if score > best_match_score:
                best_match_score = score
                best_match_chunk = chunk.chunk_id
                best_excerpt = self._find_matching_excerpt(sentence, chunk.content)
        
        if best_match_score > 0.6:
            return GroundingResult(
                sentence=sentence,
                is_grounded=True,
                confidence=best_match_score,
                matched_chunks=[best_match_chunk] if best_match_chunk else [],
                matched_excerpts=[best_excerpt] if best_excerpt else [],
                match_type="paraphrase"
            )
        
        # Strategy 4: Semantic similarity via embeddings
        best_semantic_chunk = None
        best_semantic_score = 0.0
        best_semantic_excerpt = ""
        
        for chunk in context_chunks:
            score = self._semantic_similarity(sentence, chunk.content)
            if score > best_semantic_score:
                best_semantic_score = score
                best_semantic_chunk = chunk.chunk_id
                best_semantic_excerpt = self._find_matching_excerpt(sentence, chunk.content)
        
        if best_semantic_score > 0.7:
            return GroundingResult(
                sentence=sentence,
                is_grounded=True,
                confidence=best_semantic_score,
                matched_chunks=[best_semantic_chunk] if best_semantic_chunk else [],
                matched_excerpts=[best_semantic_excerpt] if best_semantic_excerpt else [],
                match_type="inferred"
            )
        
        # No grounding found
        return GroundingResult(
            sentence=sentence,
            is_grounded=False,
            confidence=max(best_match_score, best_semantic_score),
            matched_chunks=[],
            matched_excerpts=[],
            match_type="ungrounded"
        )
    
    def _fuzzy_match_score(self, sentence: str, content: str) -> float:
        """Compute fuzzy string matching score"""
        sentence_lower = re.sub(r'\[\d+\]', '', sentence.lower()).strip()
        content_lower = content.lower()
        
        # Check for significant word overlap
        sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
        content_words = set(re.findall(r'\b\w+\b', content_lower))
        
        # Remove stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "to", "of", "in", "for", "on", "with", "at", "by", "from",
                     "this", "that", "these", "those", "it", "its"}
        sentence_words -= stopwords
        content_words -= stopwords
        
        if not sentence_words:
            return 0.0
        
        overlap = len(sentence_words & content_words)
        score = overlap / len(sentence_words)
        
        return score
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using embeddings"""
        import numpy as np
        
        emb1 = embedding_service.embed_text(text1)
        emb2 = embedding_service.embed_text(text2)
        
        # Cosine similarity (already normalized)
        similarity = float(np.dot(emb1, emb2))
        return max(0.0, min(1.0, similarity))
    
    def _find_matching_excerpt(
        self,
        sentence: str,
        content: str,
        max_length: int = 200
    ) -> str:
        """Find the most relevant excerpt from content that matches sentence"""
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        
        # Split content into sentences
        content_sentences = self._split_sentences(content)
        
        best_excerpt = ""
        best_score = 0
        
        for excerpt in content_sentences:
            excerpt_words = set(re.findall(r'\b\w+\b', excerpt.lower()))
            overlap = len(sentence_words & excerpt_words)
            if overlap > best_score:
                best_score = overlap
                best_excerpt = excerpt
        
        if len(best_excerpt) > max_length:
            best_excerpt = best_excerpt[:max_length] + "..."
        
        return best_excerpt
    
    def reject_if_ungrounded(
        self,
        validation_result: ValidationResult
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide whether to reject the answer.
        Returns (should_reject, rejection_reason)
        """
        if validation_result.errors:
            return True, validation_result.errors[0]
        
        if validation_result.grounding_score < self.min_confidence:
            return True, f"Answer grounding ({validation_result.grounding_score:.0%}) below threshold ({self.min_confidence:.0%})"
        
        return False, None


# Singleton instance
validation_service = ValidationService()
