"""
Linguistic Tokenization Module
Part of Phase 2: Data Preprocessing Pipeline

This module tokenizes sentences using spaCy's rule-based and linguistic tokenizer.
Handles contractions (e.g. "don't" -> "do", "n't"), punctuation isolation,
hyphenated compounds, abbreviations, and numerical tokens correctly.
Crucially, it configures enhanced infix patterns so that parentheses and adjacent
punctuation (e.g., '(CNN)The' -> ['(', 'CNN', ')', 'The'] and 'word,"hello' -> ['word', ',', '"', 'hello'])
are cleanly separated rather than merged into malformed tokens.
"""

from typing import List, Optional, Tuple
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import compile_infix_regex


def get_custom_infixes(nlp: Language) -> Tuple[str, ...]:
    """
    Construct enhanced infix regex patterns for spaCy's tokenizer to handle
    punctuation boundaries without naive whitespace splitting.

    Handles:
    - Parentheses and brackets directly attached to word characters: e.g. '(CNN)The' -> '(', 'CNN', ')', 'The'
    - Quotation marks adjacent to words or punctuation without spaces: e.g. 'word,"hello' -> 'word', ',', '"', 'hello'
    - Punctuation directly before word characters without spacing.
    - Preserves standard hyphenated compounds, decimals, currency, and contractions.
    """
    return tuple(
        list(nlp.Defaults.infixes)
        + [
            # Closing/opening parentheses and brackets attached to alphanumeric tokens
            r"(?<=[a-zA-Z0-9])[\)\]\}](?=[a-zA-Z0-9])",
            r"(?<=[a-zA-Z0-9])[\(\[\{](?=[a-zA-Z0-9])",
            # Closing bracket/parenthesis followed by quotation mark
            r"(?<=[a-zA-Z0-9])[\)\]\}](?=[\"“”'’])",
            # Quotation marks directly adjacent to word characters or punctuation without whitespace
            r"(?<=[a-zA-Z0-9])[\"“”](?=[a-zA-Z0-9])",
            r"(?<=[,;:!?])[\"“”](?=[a-zA-Z0-9])",
            r"(?<=[a-zA-Z0-9])[,;:!?](?=[\"“”])",
            # Punctuation directly between alphanumeric words without space e.g. 'word,next'
            r"(?<=[a-zA-Z0-9])[,;](?=[a-zA-Z0-9])",
        ]
    )


def configure_spacy_tokenizer(nlp: Language) -> Language:
    """
    Configure a spaCy Language instance with hardened infix tokenization rules.

    Args:
        nlp: spaCy Language pipeline to configure.

    Returns:
        Configured spaCy Language pipeline.
    """
    custom_infixes = get_custom_infixes(nlp)
    infix_re = compile_infix_regex(custom_infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer
    return nlp


class SpacyTokenizer:
    """
    Performs linguistic tokenization on sentences or text spans using spaCy
    with hardened punctuation and bracket boundary rules.
    """

    def __init__(self, nlp: Optional[Language] = None, model_name: str = "en_core_web_sm"):
        """
        Initialize SpacyTokenizer.

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

        # Configure tokenizer with hardened infix rules
        configure_spacy_tokenizer(self.nlp)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text or sentence string into a list of token strings.

        Args:
            text: Sentence or text fragment.

        Returns:
            List of individual token strings.
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        return [token.text for token in doc]

    def tokenize_span(self, span_or_doc: Span | Doc) -> List[str]:
        """
        Tokenize directly from an existing spaCy Doc or Span to avoid re-parsing.

        Args:
            span_or_doc: spaCy Span or Doc instance.

        Returns:
            List of token string values.
        """
        return [token.text for token in span_or_doc]
