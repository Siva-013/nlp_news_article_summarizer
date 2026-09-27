"""
Summarization Module
NLP-Based News Article Summarization and Vocabulary Learning System

Provides classical extractive text summarization algorithms:
- Frequency-based baseline summarization (Phase 3)
- TF-IDF-based extractive summarization (Phase 4)
- (Future phases will introduce TextRank summarization)
"""

__all__ = [
    "FrequencySummarizer",
    "TFIDFSummarizer",
]


def __getattr__(name: str):
    if name == "FrequencySummarizer":
        from summarization.frequency import FrequencySummarizer
        return FrequencySummarizer
    elif name == "TFIDFSummarizer":
        from summarization.tfidf import TFIDFSummarizer
        return TFIDFSummarizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
