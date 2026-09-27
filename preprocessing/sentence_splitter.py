"""
Sentence Segmentation Module
Part of Phase 2: Data Preprocessing Pipeline

This module segments cleaned prose into distinct sentences using spaCy's
statistical dependency/rule-based sentence boundaries.
Crucially, it preserves the exact original wording, casing, and punctuation
of every sentence for downstream extractive summarization.
"""

from typing import List, Optional
import spacy
from spacy.language import Language


class SentenceSplitter:
    """
    Splits text into sentences using spaCy sentence segmentation while
    preserving exact original sentence representations.
    """

    def __init__(self, nlp: Optional[Language] = None, model_name: str = "en_core_web_sm"):
        """
        Initialize SentenceSplitter with a spaCy language model.

        Args:
            nlp: Optional pre-loaded spaCy Language instance to avoid reloading.
            model_name: spaCy model name to load if nlp instance is not provided.
        """
        if nlp is not None:
            self.nlp = nlp
        else:
            try:
                self.nlp = spacy.load(model_name)
            except OSError as err:
                raise OSError(
                    f"Failed to load spaCy model '{model_name}'. "
                    f"Please install it using: python -m spacy download {model_name}"
                ) from err

    def split_sentences(self, text: str) -> List[str]:
        """
        Segment input text into a list of cleanly segmented sentence strings.
        Preserves original casing and punctuation.

        Args:
            text: Cleaned prose text.

        Returns:
            List of non-empty, stripped original sentence strings.
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        sentences: List[str] = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text:
                sentences.append(sent_text)

        # Fallback if sentence segmentation returned empty on non-empty string
        if not sentences and text.strip():
            sentences.append(text.strip())

        return sentences
