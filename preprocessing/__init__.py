"""
Preprocessing Module
NLP-Based News Article Summarization and Vocabulary Learning System

Provides classical linguistic preprocessing pipelines using spaCy:
- HTML/markup cleaning
- Unicode and whitespace normalization
- Sentence segmentation
- Tokenization
- Text normalization
- Stop-word filtering
- Lemmatization
- Part-of-Speech (POS) tagging
- Named Entity Recognition (NER)
- Dual representation preservation
"""

from preprocessing.cleaner import clean_html
from preprocessing.normalizer import normalize_unicode, normalize_whitespace, normalize_text
from preprocessing.sentence_splitter import SentenceSplitter
from preprocessing.tokenizer import SpacyTokenizer
from preprocessing.stopwords import StopwordHandler
from preprocessing.lemmatizer import SpacyLemmatizer
from preprocessing.linguistic_features import LinguisticFeatureExtractor
from preprocessing.pipeline import PreprocessingPipeline

__all__ = [
    "clean_html",
    "normalize_unicode",
    "normalize_whitespace",
    "normalize_text",
    "SentenceSplitter",
    "SpacyTokenizer",
    "StopwordHandler",
    "SpacyLemmatizer",
    "LinguisticFeatureExtractor",
    "PreprocessingPipeline",
]
