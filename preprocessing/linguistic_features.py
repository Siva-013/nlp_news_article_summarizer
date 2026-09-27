"""
Linguistic Features Module (POS Tagging & Named Entity Recognition)
Part of Phase 2: Data Preprocessing Pipeline

This module extracts:
1. Part-of-Speech (POS) tags:
   - Coarse POS tags (Universal Dependencies, e.g., NOUN, VERB, ADJ)
   - Fine-grained POS tags (Penn Treebank, e.g., NN, VBD, JJR)
2. Named Entities (NER):
   - Categorized entity spans (PERSON, ORG, GPE, LOC, DATE, MONEY, EVENT, etc.)
   with exact character offsets relative to the sentence text.
"""

from typing import Any, Dict, List, Optional
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token


class LinguisticFeatureExtractor:
    """
    Extracts POS tags, grammatical tags, and Named Entities using spaCy.
    """

    def __init__(self, nlp: Optional[Language] = None, model_name: str = "en_core_web_sm"):
        """
        Initialize LinguisticFeatureExtractor.

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

    def extract_pos_tags(self, tokens: List[Token]) -> Dict[str, List[str]]:
        """
        Extract both coarse Universal Dependencies POS tags and detailed Penn Treebank tags.

        Distinction:
        - Coarse POS (Universal Dependencies): Universal categories (NOUN, VERB, ADJ, PROPN).
          Ideal for feature filtering and keyword importance scoring.
        - Detailed Tag (Penn Treebank): Language-specific syntactic inflection (NN = singular noun,
          NNS = plural noun, VBD = past tense verb, VBG = gerund).

        Args:
            tokens: List of spaCy Token objects.

        Returns:
            Dictionary containing 'pos' (coarse) and 'tags' (fine-grained) lists.
        """
        coarse_pos = [token.pos_ for token in tokens]
        detailed_tags = [token.tag_ for token in tokens]
        return {
            "pos": coarse_pos,
            "tags": detailed_tags,
        }

    def extract_entities(self, doc_or_span: Doc | Span) -> List[Dict[str, Any]]:
        """
        Extract Named Entities from a spaCy Doc or sentence Span.
        Character offsets are calculated strictly relative to the span's text representation
        so that `span.text.strip()[start_char:end_char] == ent.text`.

        Args:
            doc_or_span: spaCy Doc or sentence Span.

        Returns:
            List of entity dictionaries with text, label, and relative character offsets.
        """
        entities: List[Dict[str, Any]] = []

        # If input is a Span within a Doc, calculate offsets relative to the trimmed span text
        if isinstance(doc_or_span, Span):
            base_offset = doc_or_span.start_char
            raw_text = doc_or_span.text
            leading_ws = len(raw_text) - len(raw_text.lstrip())
            adjusted_base = base_offset + leading_ws
        else:
            raw_text = doc_or_span.text
            leading_ws = len(raw_text) - len(raw_text.lstrip())
            adjusted_base = leading_ws

        for ent in doc_or_span.ents:
            rel_start = ent.start_char - adjusted_base
            rel_end = ent.end_char - adjusted_base
            entities.append(
                {
                    "text": ent.text.strip(),
                    "label": ent.label_,
                    "start_char": rel_start,
                    "end_char": rel_end,
                }
            )
        return entities
