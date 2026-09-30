"""
Evaluation Module
NLP-Based News Article Summarization and Vocabulary Learning System

Provides first-principles quantitative evaluation tools for extractive summarization:
- ROUGE-1, ROUGE-2, and ROUGE-L metric calculation
- Longest Common Subsequence (LCS) calculation
- Evaluator pipeline for cross-algorithm benchmarking
- Statistical aggregations, win analysis, and visualization plotting
"""

__all__ = [
    "tokenize_for_rouge",
    "get_ngrams",
    "calculate_lcs_length",
    "calculate_rouge_n",
    "calculate_rouge_l",
    "evaluate_summary_pair",
    "Evaluator",
    "calculate_descriptive_stats",
    "compute_per_article_winners",
    "generate_plots",
]


def __getattr__(name: str):
    if name in (
        "tokenize_for_rouge",
        "get_ngrams",
        "calculate_lcs_length",
        "calculate_rouge_n",
        "calculate_rouge_l",
        "evaluate_summary_pair",
    ):
        import evaluation.rouge as r
        return getattr(r, name)
    elif name in (
        "calculate_descriptive_stats",
        "compute_per_article_winners",
        "generate_plots",
    ):
        import evaluation.comparison as c
        return getattr(c, name)
    elif name == "Evaluator":
        import evaluation.evaluator as e
        return getattr(e, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
