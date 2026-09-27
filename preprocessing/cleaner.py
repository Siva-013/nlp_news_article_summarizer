"""
Text & HTML Cleaning Module
Part of Phase 2: Data Preprocessing Pipeline

This module removes unwanted HTML/markup and decodes HTML entities while
strictly preserving grammatical punctuation, numbers, quotation marks,
and sentence boundaries required for downstream extractive summarization.
"""

import html
import re


def clean_html(text: str) -> str:
    """
    Clean HTML markup and entities from raw input text.

    Steps:
    1. Replace line-breaking and block HTML tags (<br>, <p>, <div>, <tr>, </li>, </h1>..</h6>)
       with whitespace to preserve legitimate sentence boundaries and prevent text collision.
    2. Strip remaining generic HTML tags using regex without destroying mathematical comparisons (< or >).
    3. Decode HTML entities (e.g., &nbsp; -> space, &amp; -> &, &quot; -> ", &#39; -> ').
    4. Remove null bytes or anomalous non-printable control characters.

    Args:
        text: Raw text potentially containing markup or entities.

    Returns:
        Cleaned, human-readable text preserving sentence boundaries and punctuation.
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned = text

    # Step 1: Ensure block breaks have whitespace around them before stripping
    block_tags_pattern = r"(?i)</?(?:p|div|br|hr|li|tr|h[1-6]|blockquote|article|section)\b[^>]*>"
    cleaned = re.sub(block_tags_pattern, " ", cleaned)

    # Step 2: Strip any remaining HTML/XML tags
    # Target <tag attr="value"> or </tag>
    general_tag_pattern = r"<[a-zA-Z\/][^>]*>"
    cleaned = re.sub(general_tag_pattern, "", cleaned)

    # Step 3: Decode HTML entities (&nbsp;, &amp;, &quot;, &lt;, etc.)
    cleaned = html.unescape(cleaned)

    # Step 4: Replace non-breaking space characters with standard space
    cleaned = cleaned.replace("\xa0", " ")

    # Step 5: Remove null bytes or non-printable ASCII control codes (except standard \t, \n, \r)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    return cleaned
