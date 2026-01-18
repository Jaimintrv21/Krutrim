"""
Scoring utilities for retrieval and ranking
"""
from typing import List, Dict
import math


def bm25_score(
    query_terms: List[str],
    document: str,
    avg_doc_length: float,
    doc_count: int,
    term_doc_freq: Dict[str, int],
    k1: float = 1.5,
    b: float = 0.75
) -> float:
    """
    Compute BM25 score for a document given query terms.
    
    Args:
        query_terms: List of query terms
        document: Document text
        avg_doc_length: Average document length in corpus
        doc_count: Total number of documents
        term_doc_freq: Dict of term -> number of docs containing term
        k1: Term saturation parameter
        b: Length normalization parameter
    
    Returns:
        BM25 score
    """
    doc_terms = document.lower().split()
    doc_length = len(doc_terms)
    
    # Term frequency in document
    term_freq = {}
    for term in doc_terms:
        term_freq[term] = term_freq.get(term, 0) + 1
    
    score = 0.0
    for term in query_terms:
        if term not in term_freq:
            continue
        
        tf = term_freq[term]
        df = term_doc_freq.get(term, 0)
        
        # IDF component
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
        
        # TF component with length normalization
        tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
        
        score += idf * tf_component
    
    return score


def compute_recall(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """Compute recall score"""
    if not relevant:
        return 0.0
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def compute_precision(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """Compute precision score"""
    if not retrieved:
        return 0.0
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    return len(retrieved_set & relevant_set) / len(retrieved_set)


def compute_f1(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """Compute F1 score"""
    precision = compute_precision(retrieved, relevant)
    recall = compute_recall(retrieved, relevant)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_mrr(
    rankings: List[List[str]],
    relevant_items: List[str]
) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        rankings: List of ranked result lists
        relevant_items: List of relevant item IDs
    
    Returns:
        MRR score
    """
    relevant_set = set(relevant_items)
    reciprocal_ranks = []
    
    for ranking in rankings:
        for i, item in enumerate(ranking, 1):
            if item in relevant_set:
                reciprocal_ranks.append(1 / i)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to 0-1 range"""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    return [(s - min_score) / (max_score - min_score) for s in scores]


def combine_scores(
    scores_list: List[List[float]],
    weights: List[float]
) -> List[float]:
    """
    Combine multiple score lists with weights.
    Scores are normalized before combining.
    """
    if not scores_list or not weights:
        return []
    
    if len(scores_list) != len(weights):
        raise ValueError("Number of score lists must match number of weights")
    
    # Normalize each score list
    normalized = [normalize_scores(scores) for scores in scores_list]
    
    # Combine with weights
    combined = []
    num_items = len(normalized[0])
    
    for i in range(num_items):
        weighted_sum = sum(
            normalized[j][i] * weights[j]
            for j in range(len(weights))
        )
        combined.append(weighted_sum / sum(weights))
    
    return combined
