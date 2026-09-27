"""
Summarization Module
NLP-Based News Article Summarization and Vocabulary Learning System

Provides classical extractive text summarization algorithms:
- Frequency-based baseline summarization (Phase 3)
- TF-IDF-based extractive summarization (Phase 4)
- TextRank graph-based extractive summarization (Phase 5)
"""

__all__ = [
    "FrequencySummarizer",
    "TFIDFSummarizer",
    "TextRankSummarizer",
]


def __getattr__(name: str):
    if name == "FrequencySummarizer":
        from summarization.frequency import FrequencySummarizer
        return FrequencySummarizer
    elif name == "TFIDFSummarizer":
        from summarization.tfidf import TFIDFSummarizer
        return TFIDFSummarizer
    elif name == "TextRankSummarizer":
        from summarization.textrank import TextRankSummarizer
        return TextRankSummarizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
