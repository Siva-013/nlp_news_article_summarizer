"""
Summarization Module
NLP-Based News Article Summarization and Vocabulary Learning System

Provides classical extractive text summarization algorithms:
- Frequency-based baseline summarization (Phase 3)
- (Future phases will introduce TF-IDF and TextRank summarization)
"""

__all__ = [
    "FrequencySummarizer",
]


def __getattr__(name: str):
    if name == "FrequencySummarizer":
        from summarization.frequency import FrequencySummarizer
        return FrequencySummarizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
