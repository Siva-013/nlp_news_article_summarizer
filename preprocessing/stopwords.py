"""
Stop-Word Processing Module
Part of Phase 2: Data Preprocessing Pipeline

This module identifies stop words using spaCy's English stop word vocabulary.
Crucially, stop words are NOT deleted from the original text or sentence structure;
instead, stop-word boolean flags are recorded at the token level, and filtered
content token lists are constructed specifically for analytical scoring and
downstream feature extraction.
"""

from typing import List, Optional, Set
import spacy
from spacy.language import Language
from spacy.tokens import Token


class StopwordHandler:
    """
    Handles stop-word identification and content token filtering.
    """

    def __init__(self, nlp: Optional[Language] = None, model_name: str = "en_core_web_sm"):
        """
        Initialize StopwordHandler.

        Args:
            nlp: Optional pre-loaded spaCy Language instance.
            model_name: spaCy model name if nlp instance is not provided.
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

        # Cache default stop words set (lowercased)
        self.stop_words: Set[str] = set(self.nlp.Defaults.stop_words)

    def is_stopword(self, token_or_str: Token | str) -> bool:
        """
        Check if a given token or string is an English stop word.

        Args:
            token_or_str: spaCy Token instance or string.

        Returns:
            True if identified as a stop word, False otherwise.
        """
        if isinstance(token_or_str, Token):
            return bool(token_or_str.is_stop)
        return token_or_str.lower().strip() in self.stop_words

    def filter_content_tokens(
        self,
        tokens: List[Token],
        remove_punctuation: bool = True,
        remove_whitespace: bool = True,
    ) -> List[str]:
        """
        Filter tokens to extract meaningful content words (nouns, lexical verbs,
        adjectives, key adverbs) by filtering out stop words, punctuation, and whitespace.

        Args:
            tokens: Sequence of spaCy Token objects.
            remove_punctuation: Whether to exclude punctuation tokens.
            remove_whitespace: Whether to exclude whitespace tokens.

        Returns:
            List of content token strings in normalized lowercase.
        """
        content_tokens: List[str] = []
        for token in tokens:
            if token.is_stop:
                continue
            if remove_punctuation and token.is_punct:
                continue
            if remove_whitespace and (token.is_space or not token.text.strip()):
                continue
            # Additional safety: ensure token contains alphanumeric characters
            cleaned_text = token.text.strip().lower()
            if cleaned_text and any(c.isalnum() for c in cleaned_text):
                content_tokens.append(cleaned_text)

        return content_tokens
