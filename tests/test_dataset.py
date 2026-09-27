"""
Unit Tests for Phase 1: Dataset Exploration & Validation
Tests word counting, sentence counting, compression ratio, Type-Token Ratio,
validation checks, and JSONL file operations.
"""

import json
from pathlib import Path
import pytest

from dataset.explore_dataset import (
    count_words,
    count_sentences,
    calculate_vocabulary_metrics,
    calculate_statistics,
    validate_records,
    load_dataset,
)
from dataset.download_dataset import validate_records as validate_download_records


def test_count_words():
    """Verify word count logic on standard text, empty text, and whitespace."""
    assert count_words("") == 0
    assert count_words("   ") == 0
    assert count_words("Hello world") == 2
    assert count_words("   Natural   Language   Processing   ") == 3
    assert count_words("It's a state-of-the-art NLP model.") == 5


def test_count_sentences():
    """Verify sentence counting on single sentences, multi-sentences, and empty text."""
    assert count_sentences("") == 0
    assert count_sentences("   ") == 0
    assert count_sentences("Single sentence without period") == 1
    assert count_sentences("This is the first sentence. This is the second one!") == 2
    assert count_sentences("Headline 1. Second sentence? Third sentence...") == 3


def test_calculate_vocabulary_metrics():
    """Verify total words, unique words, and type-token ratio (TTR)."""
    text = "The cat sat on the mat. The cat was happy."
    total, unique, ttr = calculate_vocabulary_metrics(text)
    # Tokens: the, cat, sat, on, the, mat, the, cat, was, happy (10 tokens)
    # Unique: the, cat, sat, on, mat, was, happy (7 unique)
    assert total == 10
    assert unique == 7
    assert ttr == pytest.approx(0.7, rel=1e-2)

    # Empty text case
    total_empty, unique_empty, ttr_empty = calculate_vocabulary_metrics("")
    assert total_empty == 0
    assert unique_empty == 0
    assert ttr_empty == 0.0


def test_calculate_statistics():
    """Verify minimum, maximum, mean, median, and standard deviation calculations."""
    data = [10, 20, 30, 40, 50]
    stats = calculate_statistics(data)
    assert stats["min"] == 10
    assert stats["max"] == 50
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert stats["std"] > 0

    empty_stats = calculate_statistics([])
    assert empty_stats["mean"] == 0.0
    assert empty_stats["min"] == 0


def test_validation_detects_invalid_records():
    """Verify validation detects missing fields, empty texts, and duplicate IDs."""
    sample_records = [
        {"id": "id1", "article": "Valid article content here.", "highlights": "Valid highlight."},
        {"id": "", "article": "Article with missing ID", "highlights": "Highlight"},
        {"id": "id2", "article": "", "highlights": "Empty article text"},
        {"id": "id3", "article": "Article with empty highlights", "highlights": "   "},
        {"id": "id1", "article": "Duplicate ID record", "highlights": "Another highlight"},
    ]

    valid_recs, metrics = validate_records(sample_records)

    assert len(valid_recs) == 1
    assert valid_recs[0]["id"] == "id1"
    assert metrics["valid"] == 1
    assert metrics["missing_ids"] == 1
    assert metrics["empty_articles"] == 1
    assert metrics["empty_highlights"] == 1
    assert metrics["duplicate_ids"] == 1
    assert metrics["invalid"] == 4


def test_jsonl_roundtrip(tmp_path: Path):
    """Verify loading and validation of JSONL files."""
    test_file = tmp_path / "test_sample.jsonl"
    data = [
        {"id": "abc1", "article": "Article 1 text.", "highlights": "Highlight 1."},
        {"id": "abc2", "article": "Article 2 text.", "highlights": "Highlight 2."},
    ]
    with test_file.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    loaded = load_dataset(test_file)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "abc1"
    assert loaded[1]["article"] == "Article 2 text."
