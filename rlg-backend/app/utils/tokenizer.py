"""
Tokenizer utilities for text processing
"""
import re
from typing import List, Set


# Common English stopwords
STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "done", "for", "from",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "might", "more", "most", "must", "my",
    "myself", "no", "nor", "not", "now", "of", "on", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "yourselves"
}


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.
    Lowercases and removes punctuation.
    """
    text = text.lower()
    words = re.findall(r'\b[a-z0-9]+\b', text)
    return words


def tokenize_without_stopwords(text: str) -> List[str]:
    """
    Tokenize text and remove stopwords.
    Useful for keyword extraction.
    """
    tokens = tokenize(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def extract_ngrams(text: str, n: int = 2) -> List[str]:
    """Extract n-grams from text"""
    tokens = tokenize(text)
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """
    Extract key phrases using simple frequency analysis.
    Returns most important phrases.
    """
    # Get bigrams and trigrams
    bigrams = extract_ngrams(text, 2)
    trigrams = extract_ngrams(text, 3)
    
    # Filter out phrases with stopwords at boundaries
    phrases = []
    for phrase in bigrams + trigrams:
        words = phrase.split()
        if words[0] not in STOPWORDS and words[-1] not in STOPWORDS:
            phrases.append(phrase)
    
    # Count frequency
    freq = {}
    for phrase in phrases:
        freq[phrase] = freq.get(phrase, 0) + 1
    
    # Sort by frequency and return top phrases
    sorted_phrases = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [phrase for phrase, count in sorted_phrases[:max_phrases]]


def estimate_tokens(text: str) -> int:
    """
    Estimate token count (roughly 4 chars per token).
    For more accurate counts, use the actual tokenizer.
    """
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximate token limit"""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    
    # Try to truncate at sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1]
    
    return truncated + "..."
