"""
Linguistic Lemmatization Module
Part of Phase 2: Data Preprocessing Pipeline

This module produces morphological root forms (lemmas) for tokens using spaCy's
rule-based and vocabulary-driven lemmatizer.
Lemmas are stored alongside original words in parallel representations without
modifying original sentences.
"""

from typing import List, Optional
import spacy
from spacy.language import Language
from spacy.tokens import Token


class SpacyLemmatizer:
    """
    Extracts base canonical lemma representations from spaCy tokens.
    """

    def __init__(self, nlp: Optional[Language] = None, model_name: str = "en_core_web_sm"):
        """
        Initialize SpacyLemmatizer.

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

    def lemmatize_token(self, token: Token) -> str:
        """
        Extract the lowercased lemma string from a spaCy token.

        Args:
            token: spaCy Token instance.

        Returns:
            Lemma string (lowercased), or token text if lemma is unavailable.
        """
        lemma = token.lemma_.strip().lower()
        return lemma if lemma else token.text.strip().lower()

    def lemmatize_tokens(self, tokens: List[Token]) -> List[str]:
        """
        Lemmatize a sequence of spaCy Token objects.

        Args:
            tokens: List of spaCy Token objects.

        Returns:
            List of lemma strings.
        """
        return [self.lemmatize_token(tok) for tok in tokens]

    def lemmatize_text(self, text: str) -> List[str]:
        """
        Parse text with spaCy and return list of lemma strings.

        Args:
            text: Input string.

        Returns:
            List of lemma strings for all tokens.
        """
        if not text or not text.strip():
            return []
        doc = self.nlp(text)
        return [self.lemmatize_token(token) for token in doc]
