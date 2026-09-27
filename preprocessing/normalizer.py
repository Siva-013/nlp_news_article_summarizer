"""
Text & Unicode Normalization Module
Part of Phase 2: Data Preprocessing Pipeline

This module implements:
1. Canonical Unicode normalization (NFC) using unicodedata.
2. Whitespace normalization (collapsing multi-spaces, tabs, irregular newlines).
3. Token normalization (lowercasing and analytical cleaning while preserving originals).
"""

import re
import unicodedata
from typing import List


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """
    Apply Unicode normalization to convert text into canonical composite representation.

    Why NFC (Normalization Form C) is used:
    In news articles and multi-source corpora, characters with diacritics or accents can
    be represented either as a single precomposed character (e.g., U+00E9 for 'é') or as
    decomposed combining characters (U+0065 'e' + U+0301 '´').
    NFC composes combining characters into standard precomposed codepoints, ensuring
    consistent vocabulary matching, exact string equality, and preventing vocabulary fragmentation.

    Args:
        text: Input string.
        form: Unicode normalization form ('NFC', 'NFKC', 'NFD', 'NFKD'). Default: 'NFC'.

    Returns:
        Canonically normalized Unicode string.
    """
    if not text or not isinstance(text, str):
        return ""
    return unicodedata.normalize(form, text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize irregular whitespace, tabs, and line breaks.

    Replaces:
    - Multiple consecutive spaces or tabs with a single space.
    - Consecutive newlines/carriage returns with a single space or newline.
    - Strips leading and trailing whitespace.

    Args:
        text: Input string.

    Returns:
        Cleanly spaced string without disrupted sentence boundaries.
    """
    if not text or not isinstance(text, str):
        return ""

    # Replace windows carriage returns
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace tabs and irregular whitespace with regular space
    normalized = re.sub(r"[ \t]+", " ", normalized)

    # Collapse multiple newlines into a single newline
    normalized = re.sub(r"\n\s*\n+", "\n", normalized)

    # Replace lone newlines with a single space to avoid broken sentence fragments in prose
    normalized = re.sub(r"\n", " ", normalized)

    # Final pass to collapse any newly created multiple spaces
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def normalize_text(text: str) -> str:
    """
    Convenience function applying both Unicode normalization and whitespace normalization.
    """
    return normalize_whitespace(normalize_unicode(text, form="NFC"))


def normalize_token(token_text: str) -> str:
    """
    Normalize an individual token for linguistic analysis and vocabulary matching.
    - Converts token to lowercase
    - Strips non-alphanumeric punctuation boundaries if present for word-level analysis

    Args:
        token_text: Original token string.

    Returns:
        Lowercased and normalized token string.
    """
    if not token_text:
        return ""
    # Canonical lowercase and stripped
    return token_text.lower().strip()
