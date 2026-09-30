"""
ROUGE Metric Computation Module
Part of Phase 6: Quantitative Evaluation and Algorithm Comparison

Implements standard ROUGE metrics (Lin, 2004) from first principles:
- ROUGE-1: Unigram recall, precision, and F1
- ROUGE-2: Bigram recall, precision, and F1
- ROUGE-L: Longest Common Subsequence (LCS) recall, precision, and F1

All metrics are calculated explicitly without black-box third-party libraries.
"""

from collections import Counter
import re
from typing import Dict, List, Set, Tuple


def tokenize_for_rouge(text: str) -> List[str]:
    """
    Tokenize text consistently for ROUGE evaluation.
    
    Processing steps:
    1. Lowercase text.
    2. Extract alphanumeric word tokens (stripping standalone punctuation).
    3. Retain contractions and hyphens within words.
    
    Args:
        text: Raw or extractive text string.
        
    Returns:
        List of cleaned, lowercased word tokens.
    """
    if not text:
        return []
    # Lowercase and extract alphanumeric sequences (including intra-word hyphens/apostrophes)
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:['\-_][a-zA-Z0-9]+)*\b", text.lower())
    return tokens


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """
    Extract n-gram frequency distribution from a sequence of tokens.
    
    Args:
        tokens: List of word tokens.
        n: Size of n-grams (e.g. 1 for unigrams, 2 for bigrams).
        
    Returns:
        Counter mapping n-gram tuples to their integer frequency.
    """
    if not tokens or n <= 0 or len(tokens) < n:
        return Counter()
    
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)


def calculate_rouge_n(candidate: str, reference: str, n: int = 1) -> Dict[str, float]:
    """
    Calculate ROUGE-N (Precision, Recall, F1) between candidate and reference texts.
    
    Formula:
        Overlap = sum_gram min(count_cand(gram), count_ref(gram))
        Recall    = Overlap / total_reference_ngrams
        Precision = Overlap / total_candidate_ngrams
        F1        = (2 * Precision * Recall) / (Precision + Recall)
        
    Args:
        candidate: Generated summary text.
        reference: Ground-truth reference summary text.
        n: N-gram size (1 for ROUGE-1, 2 for ROUGE-2).
        
    Returns:
        Dictionary with 'precision', 'recall', and 'f1' rounded to 6 decimal places.
    """
    cand_tokens = tokenize_for_rouge(candidate)
    ref_tokens = tokenize_for_rouge(reference)
    
    if not cand_tokens or not ref_tokens or len(cand_tokens) < n or len(ref_tokens) < n:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    cand_ngrams = get_ngrams(cand_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)
    
    total_cand = sum(cand_ngrams.values())
    total_ref = sum(ref_ngrams.values())
    
    if total_cand == 0 or total_ref == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Calculate clipping overlap
    overlap = 0
    for gram, cand_count in cand_ngrams.items():
        if gram in ref_ngrams:
            overlap += min(cand_count, ref_ngrams[gram])
            
    recall = overlap / total_ref if total_ref > 0 else 0.0
    precision = overlap / total_cand if total_cand > 0 else 0.0
    
    if precision + recall > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def calculate_lcs_length(candidate_tokens: List[str], reference_tokens: List[str]) -> int:
    """
    Calculate the length of the Longest Common Subsequence (LCS) between two token sequences.
    
    Uses space-optimized dynamic programming: O(len(ref)) space and O(len(cand) * len(ref)) time.
    
    Args:
        candidate_tokens: List of tokens in candidate summary.
        reference_tokens: List of tokens in reference summary.
        
    Returns:
        Integer length of the longest common subsequence.
    """
    m, n = len(candidate_tokens), len(reference_tokens)
    if m == 0 or n == 0:
        return 0
    
    # Use two rows for space optimization
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        cand_token = candidate_tokens[i - 1]
        for j in range(1, n + 1):
            if cand_token == reference_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
        
    return prev[n]


def calculate_rouge_l(candidate: str, reference: str) -> Dict[str, float]:
    """
    Calculate ROUGE-L (Precision, Recall, F1) based on Longest Common Subsequence (LCS).
    
    Formula:
        Recall    = LCS(candidate, reference) / len(reference)
        Precision = LCS(candidate, reference) / len(candidate)
        F1        = (2 * Precision * Recall) / (Precision + Recall)
        
    Args:
        candidate: Generated summary text.
        reference: Ground-truth reference summary text.
        
    Returns:
        Dictionary with 'precision', 'recall', and 'f1' rounded to 6 decimal places.
    """
    cand_tokens = tokenize_for_rouge(candidate)
    ref_tokens = tokenize_for_rouge(reference)
    
    len_cand = len(cand_tokens)
    len_ref = len(ref_tokens)
    
    if len_cand == 0 or len_ref == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    lcs = calculate_lcs_length(cand_tokens, ref_tokens)
    
    recall = lcs / len_ref if len_ref > 0 else 0.0
    precision = lcs / len_cand if len_cand > 0 else 0.0
    
    if precision + recall > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def evaluate_summary_pair(candidate: str, reference: str) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all three standard ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L) for a summary pair.
    
    Args:
        candidate: Candidate generated summary.
        reference: Reference target highlight.
        
    Returns:
        Dictionary containing 'rouge1', 'rouge2', and 'rougeL' score dictionaries.
    """
    return {
        "rouge1": calculate_rouge_n(candidate, reference, n=1),
        "rouge2": calculate_rouge_n(candidate, reference, n=2),
        "rougeL": calculate_rouge_l(candidate, reference),
    }
