"""
Vocabulary Package
Part of Phase 7 & 8: Vocabulary Extraction & New-Word Detection

This package provides classical NLP vocabulary extraction and new-word
detection for news articles.
"""

__all__ = [
    "VocabularyExtractor",
    "NewWordDetector",
    "DEFAULT_WEIGHTS",
    "DEFAULT_ELIGIBLE_POS",
    "DEFAULT_POS_WEIGHTS",
]


def __getattr__(name: str):
    if name in (
        "VocabularyExtractor",
        "DEFAULT_WEIGHTS",
        "DEFAULT_ELIGIBLE_POS",
        "DEFAULT_POS_WEIGHTS",
    ):
        import vocabulary.extractor as ext
        return getattr(ext, name)
    elif name == "NewWordDetector":
        import vocabulary.new_word_detector as nwd
        return getattr(nwd, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
