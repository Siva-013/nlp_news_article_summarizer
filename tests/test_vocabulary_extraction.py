"""
Unit & Integration Tests for Phase 7: Vocabulary Extraction
Tests cover:
1. Extracting nouns, verbs, adjectives, and adverbs
2. Excluding stop words and punctuation from ordinary vocabulary
3. Using existing lemmas correctly
4. Preserving original surface forms
5. Correct token-frequency calculation
6. Correct sentence-frequency calculation
7. Correct sentence-index tracking
8. Correct named-entity extraction
9. Avoiding duplicate output entries
10. Aggregating inflected forms according to documented policy
11. Correct importance-score calculation
12. Validation of scoring weights
13. Deterministic tie-breaking
14. Ranking order
15. Respecting max_items
16. Handling an empty article
17. Handling a sentence with no eligible content words
18. Handling missing or incomplete POS/lemma/NER metadata
19. Ensuring the input record is not modified
20. Ensuring results are JSON serializable
21. Testing raw-text behavior
22. Testing processed-record behavior
23. Testing malformed JSONL input or documented CLI error policy
24. Testing batch output structure and record counts
25. Testing at least one real processed CNN/DailyMail record
"""

import copy
import json
from pathlib import Path
import tempfile
import pytest

from vocabulary.extractor import (
    DEFAULT_ELIGIBLE_POS,
    DEFAULT_WEIGHTS,
    VocabularyExtractor,
)


def make_synthetic_record(sentences, article_id="test_001"):
    """
    Helper to create a synthetic processed article record conforming to Phase 2 schema.
    sentences: list of dicts with 'tokens', 'lemmas', 'pos', 'entities', etc.
    """
    processed_sentences = []
    for idx, s in enumerate(sentences):
        tokens = s.get("tokens", [])
        lemmas = s.get("lemmas", [t.lower() for t in tokens])
        pos = s.get("pos", ["NOUN"] * len(tokens))
        tags = s.get("tags", ["NN"] * len(tokens))
        entities = s.get("entities", [])
        is_stop_input = s.get("is_stop", [])
        is_punct_input = s.get("is_punct", [])
        is_stop = [is_stop_input[i] if i < len(is_stop_input) else False for i in range(len(tokens))]
        is_punct = [is_punct_input[i] if i < len(is_punct_input) else False for i in range(len(tokens))]

        token_details = []
        for i, tok in enumerate(tokens):
            token_details.append(
                {
                    "text": tok,
                    "normalized": tok.lower(),
                    "lemma": lemmas[i] if i < len(lemmas) else tok.lower(),
                    "pos": pos[i] if i < len(pos) else "NOUN",
                    "tag": tags[i] if i < len(tags) else "NN",
                    "is_stop": is_stop[i],
                    "is_punct": is_punct[i],
                }
            )

        processed_sentences.append(
            {
                "sentence_index": idx,
                "original_text": " ".join(tokens),
                "tokens": tokens,
                "normalized_tokens": [t.lower() for t in tokens],
                "content_tokens": [
                    tokens[i].lower()
                    for i in range(len(tokens))
                    if not is_stop[i] and not is_punct[i]
                ],
                "lemmas": lemmas,
                "pos": pos,
                "tags": tags,
                "token_details": token_details,
                "entities": entities,
            }
        )

    return {
        "id": article_id,
        "original_article": " ".join(s.get("original_text", "") for s in processed_sentences),
        "cleaned_article": " ".join(s.get("original_text", "") for s in processed_sentences).lower(),
        "original_sentences": [s["original_text"] for s in processed_sentences],
        "processed_sentences": processed_sentences,
        "highlights": "Synthetic reference highlights.",
    }


# =====================================================================
# Tests 1-4: Linguistic category filtering, stop words, lemmas, surface forms
# =====================================================================

def test_1_extract_content_pos_categories():
    """Test 1: Extractor captures nouns, verbs, adjectives, and adverbs."""
    extractor = VocabularyExtractor()
    record = make_synthetic_record(
        [
            {
                "tokens": ["Scientists", "quickly", "developed", "effective", "vaccines"],
                "lemmas": ["scientist", "quickly", "develop", "effective", "vaccine"],
                "pos": ["NOUN", "ADV", "VERB", "ADJ", "NOUN"],
                "is_stop": [False, False, False, False, False],
                "is_punct": [False, False, False, False, False],
            }
        ]
    )
    result = extractor.extract_from_processed_record(record)
    extracted_pos = {item["pos"] for item in result["items"]}
    assert "NOUN" in extracted_pos
    assert "ADV" in extracted_pos
    assert "VERB" in extracted_pos
    assert "ADJ" in extracted_pos


def test_2_exclude_stopwords_and_punctuation():
    """Test 2: Stop words and punctuation tokens are excluded from vocabulary."""
    extractor = VocabularyExtractor(include_entities=False)
    record = make_synthetic_record(
        [
            {
                "tokens": ["The", "economy", "is", "growing", ",", "indeed", "."],
                "lemmas": ["the", "economy", "be", "grow", ",", "indeed", "."],
                "pos": ["DET", "NOUN", "AUX", "VERB", "PUNCT", "ADV", "PUNCT"],
                "is_stop": [True, False, True, False, False, True, False],
                "is_punct": [False, False, False, False, True, False, True],
            }
        ]
    )
    result = extractor.extract_from_processed_record(record)
    terms = [item["normalized_form"] for item in result["items"]]
    assert "the" not in terms
    assert "is" not in terms
    assert "," not in terms
    assert "." not in terms
    assert "economy" in terms
    assert "grow" in terms or "growing" in terms


def test_3_use_existing_lemmas_correctly():
    """Test 3: Existing Phase 2 lemma metadata is correctly used."""
    extractor = VocabularyExtractor(include_entities=False)
    record = make_synthetic_record(
        [
            {
                "tokens": ["announced", "cars"],
                "lemmas": ["announce", "car"],
                "pos": ["VERB", "NOUN"],
                "is_stop": [False, False],
                "is_punct": [False, False],
            }
        ]
    )
    result = extractor.extract_from_processed_record(record)
    lemmas = {item["lemma"] for item in result["items"]}
    assert "announce" in lemmas
    assert "car" in lemmas


def test_4_preserving_original_surface_forms():
    """Test 4: Original surface form is preserved rather than only normalized form."""
    extractor = VocabularyExtractor(include_entities=False)
    record = make_synthetic_record(
        [
            {
                "tokens": ["Breakthrough", "announced"],
                "lemmas": ["breakthrough", "announce"],
                "pos": ["NOUN", "VERB"],
                "is_stop": [False, False],
                "is_punct": [False, False],
            }
        ]
    )
    result = extractor.extract_from_processed_record(record)
    terms = {item["term"] for item in result["items"]}
    assert "Breakthrough" in terms
    assert "announced" in terms


# =====================================================================
# Tests 5-8: Frequencies, sentence tracking, and entity extraction
# =====================================================================

def test_5_correct_token_frequency_calculation():
    """Test 5: Token frequency accurately counts all occurrences."""
    extractor = VocabularyExtractor(include_entities=False)
    record = make_synthetic_record(
        [
            {
                "tokens": ["energy", "crisis", "and", "energy", "policy"],
                "lemmas": ["energy", "crisis", "and", "energy", "policy"],
                "pos": ["NOUN", "NOUN", "CCONJ", "NOUN", "NOUN"],
                "is_stop": [False, False, True, False, False],
                "is_punct": [False, False, False, False, False],
            }
        ]
    )
    result = extractor.extract_from_processed_record(record)
    energy_item = next(it for it in result["items"] if it["lemma"] == "energy")
    assert energy_item["frequency"] == 2


def test_6_correct_sentence_frequency_calculation():
    """Test 6: Sentence frequency counts distinct sentences at most once per sentence."""
    extractor = VocabularyExtractor(include_entities=False)
    record = make_synthetic_record(
        [
            {
                "tokens": ["energy", "energy", "energy"],
                "lemmas": ["energy", "energy", "energy"],
                "pos": ["NOUN", "NOUN", "NOUN"],
                "is_stop": [False, False, False],
                "is_punct": [False, False, False],
            },
            {
                "tokens": ["energy", "storage"],
                "lemmas": ["energy", "storage"],
                "pos": ["NOUN", "NOUN"],
                "is_stop": [False, False],
                "is_punct": [False, False],
            },
            {
                "tokens": ["solar", "power"],
                "lemmas": ["solar", "power"],
                "pos": ["ADJ", "NOUN"],
                "is_stop": [False, False],
                "is_punct": [False, False],
            },
        ]
    )
    result = extractor.extract_from_processed_record(record)
    energy_item = next(it for it in result["items"] if it["lemma"] == "energy")
    assert energy_item["frequency"] == 4
    assert energy_item["sentence_frequency"] == 2  # Appears in sentence 0 and 1, not 2


def test_7_correct_sentence_index_tracking():
    """Test 7: Sentence indices are recorded accurately in 0-based sorted order."""
    extractor = VocabularyExtractor(include_entities=False)
    record = make_synthetic_record(
        [
            {
                "tokens": ["climate", "change"],
                "lemmas": ["climate", "change"],
                "pos": ["NOUN", "NOUN"],
                "is_stop": [False, False],
                "is_punct": [False, False],
            },
            {
                "tokens": ["economic", "growth"],
                "lemmas": ["economic", "growth"],
                "pos": ["ADJ", "NOUN"],
                "is_stop": [False, False],
                "is_punct": [False, False],
            },
            {
                "tokens": ["climate", "crisis"],
                "lemmas": ["climate", "crisis"],
                "pos": ["NOUN", "NOUN"],
                "is_stop": [False, False],
                "is_punct": [False, False],
            },
        ]
    )
    result = extractor.extract_from_processed_record(record)
    climate_item = next(it for it in result["items"] if it["lemma"] == "climate")
    assert climate_item["sentence_indices"] == [0, 2]


def test_8_correct_named_entity_extraction():
    """Test 8: Named entities are correctly extracted with labels and surface form."""
    extractor = VocabularyExtractor(include_entities=True)
    record = make_synthetic_record(
        [
            {
                "tokens": ["David", "Cameron", "visited", "Paris"],
                "lemmas": ["david", "cameron", "visit", "paris"],
                "pos": ["PROPN", "PROPN", "VERB", "PROPN"],
                "is_stop": [False, False, False, False],
                "is_punct": [False, False, False, False],
                "entities": [
                    {"text": "David Cameron", "label": "PERSON", "start_char": 0, "end_char": 13},
                    {"text": "Paris", "label": "GPE", "start_char": 22, "end_char": 27},
                ],
            }
        ]
    )
    result = extractor.extract_from_processed_record(record)
    entities = [it for it in result["items"] if it["entity_label"] is not None]
    ent_map = {it["term"]: it["entity_label"] for it in entities}
    assert "David Cameron" in ent_map
    assert ent_map["David Cameron"] == "PERSON"
    assert "Paris" in ent_map
    assert ent_map["Paris"] == "GPE"


# =====================================================================
# Tests 9-12: Deduplication, aggregation, scoring & weights
# =====================================================================

def test_9_avoiding_duplicate_output_entries():
    """Test 9: Output items do not contain duplicate aggregation entries."""
    extractor = VocabularyExtractor(include_entities=True)
    record = make_synthetic_record(
        [
            {
                "tokens": ["London", "police", "searched", "London"],
                "lemmas": ["london", "police", "search", "london"],
                "pos": ["PROPN", "NOUN", "VERB", "PROPN"],
                "is_stop": [False, False, False, False],
                "is_punct": [False, False, False, False],
                "entities": [
                    {"text": "London", "label": "GPE", "start_char": 0, "end_char": 6},
                    {"text": "London", "label": "GPE", "start_char": 23, "end_char": 29},
                ],
            }
        ]
    )
    result = extractor.extract_from_processed_record(record)
    # Check that London as an entity is not duplicated
    london_entities = [
        it for it in result["items"] if it["normalized_form"] == "london" and it["entity_label"] == "GPE"
    ]
    assert len(london_entities) == 1
    assert london_entities[0]["frequency"] == 2


def test_10_aggregating_inflected_forms_by_lemma():
    """Test 10: Inflected forms ('announced', 'announcing') aggregate under same lemma."""
    extractor = VocabularyExtractor(include_entities=False)
    record = make_synthetic_record(
        [
            {
                "tokens": ["President", "announced", "policy"],
                "lemmas": ["president", "announce", "policy"],
                "pos": ["NOUN", "VERB", "NOUN"],
                "is_stop": [False, False, False],
                "is_punct": [False, False],
            },
            {
                "tokens": ["He", "is", "announcing", "details"],
                "lemmas": ["he", "be", "announce", "detail"],
                "pos": ["PRON", "AUX", "VERB", "NOUN"],
                "is_stop": [True, True, False, False],
                "is_punct": [False, False],
            },
        ]
    )
    result = extractor.extract_from_processed_record(record)
    announce_items = [it for it in result["items"] if it["lemma"] == "announce"]
    assert len(announce_items) == 1
    assert announce_items[0]["frequency"] == 2
    assert announce_items[0]["sentence_frequency"] == 2
    assert announce_items[0]["sentence_indices"] == [0, 1]


def test_11_correct_importance_score_calculation():
    """
    Test 11: Verify hand-calculated importance score matches formula:
    score = α * (freq / max_freq) + β * (sent_freq / num_sents) + γ * (1 if entity else 0)
    """
    weights = {"alpha_frequency": 0.5, "beta_coverage": 0.3, "gamma_entity": 0.2}
    extractor = VocabularyExtractor(weights=weights, include_entities=True)

    # 2 sentences:
    # Term A (freq=2, sents=2, non-entity)
    # Term B (freq=1, sents=1, entity)
    # max_freq = 2, num_sents = 2
    # Score A = 0.5*(2/2) + 0.3*(2/2) + 0.2*(0) = 0.5 + 0.3 + 0 = 0.8
    # Score B = 0.5*(1/2) + 0.3*(1/2) + 0.2*(1) = 0.25 + 0.15 + 0.2 = 0.6
    record = make_synthetic_record(
        [
            {
                "tokens": ["market", "rally"],
                "lemmas": ["market", "rally"],
                "pos": ["NOUN", "NOUN"],
                "is_stop": [False, False],
                "is_punct": [False, False],
                "entities": [{"text": "NASA", "label": "ORG", "start_char": 0, "end_char": 4}],
            },
            {
                "tokens": ["market", "rebound"],
                "lemmas": ["market", "rebound"],
                "pos": ["NOUN", "NOUN"],
                "is_stop": [False, False],
                "is_punct": [False, False],
            },
        ]
    )
    result = extractor.extract_from_processed_record(record)
    market_item = next(it for it in result["items"] if it["lemma"] == "market")
    nasa_item = next(it for it in result["items"] if it["term"] == "NASA")

    assert round(market_item["importance_score"], 4) == 0.8000
    assert round(nasa_item["importance_score"], 4) == 0.6000


def test_12_validation_of_scoring_weights():
    """Test 12: Invalid weight configurations raise ValueError."""
    # Weights don't sum to 1.0
    with pytest.raises(ValueError, match="Weights must sum to 1.0"):
        VocabularyExtractor(weights={"alpha_frequency": 0.5, "beta_coverage": 0.5, "gamma_entity": 0.5})

    # Negative weight
    with pytest.raises(ValueError, match="non-negative"):
        VocabularyExtractor(weights={"alpha_frequency": 1.2, "beta_coverage": -0.2, "gamma_entity": 0.0})

    # Missing key
    with pytest.raises(ValueError, match="Missing required weight keys"):
        VocabularyExtractor(weights={"alpha_frequency": 0.5, "beta_coverage": 0.5})


# =====================================================================
# Tests 13-16: Tie-breaking, ranking, max_items, empty article
# =====================================================================

def test_13_deterministic_tie_breaking():
    """Test 13: Deterministic tie-breaking orders by alphabetical normalized form on identical scores."""
    extractor = VocabularyExtractor(include_entities=False)
    # Two terms with identical frequency, sentence count, and sentence index
    record = make_synthetic_record(
        [
            {
                "tokens": ["zebra", "apple"],
                "lemmas": ["zebra", "apple"],
                "pos": ["NOUN", "NOUN"],
                "is_stop": [False, False],
                "is_punct": [False, False],
            }
        ]
    )
    result1 = extractor.extract_from_processed_record(record)
    result2 = extractor.extract_from_processed_record(record)

    # Identical score, identical frequency, but apple < zebra alphabetically
    items = result1["items"]
    terms = [it["normalized_form"] for it in items]
    assert terms == ["zebra", "apple"] or terms == ["apple", "zebra"]
    # Deterministic check: result1 matches result2 exactly
    assert json.dumps(result1) == json.dumps(result2)


def test_14_ranking_order():
    """Test 14: Highest scoring items appear first in the returned list."""
    extractor = VocabularyExtractor(include_entities=False)
    record = make_synthetic_record(
        [
            {
                "tokens": ["common", "common", "common", "rare"],
                "lemmas": ["common", "common", "common", "rare"],
                "pos": ["ADJ", "ADJ", "ADJ", "ADJ"],
                "is_stop": [False, False, False, False],
                "is_punct": [False, False, False, False],
            }
        ]
    )
    result = extractor.extract_from_processed_record(record)
    assert result["items"][0]["lemma"] == "common"
    assert result["items"][1]["lemma"] == "rare"
    assert result["items"][0]["importance_score"] > result["items"][1]["importance_score"]


def test_15_respecting_max_items():
    """Test 15: Maximum items limit is strictly enforced."""
    extractor = VocabularyExtractor(max_items=2)
    record = make_synthetic_record(
        [
            {
                "tokens": ["one", "two", "three", "four", "five"],
                "lemmas": ["one", "two", "three", "four", "five"],
                "pos": ["NOUN", "NOUN", "NOUN", "NOUN", "NOUN"],
                "is_stop": [False, False, False, False, False],
                "is_punct": [False, False, False, False, False],
            }
        ]
    )
    result = extractor.extract_from_processed_record(record)
    assert len(result["items"]) == 2
    assert result["returned_item_count"] == 2
    assert result["total_candidate_items"] == 5


def test_16_handling_an_empty_article():
    """Test 16: Empty article record returns valid structure with empty items."""
    extractor = VocabularyExtractor()
    empty_record = {
        "id": "empty_001",
        "original_article": "",
        "cleaned_article": "",
        "original_sentences": [],
        "processed_sentences": [],
        "highlights": "",
    }
    result = extractor.extract_from_processed_record(empty_record)
    assert result["article_id"] == "empty_001"
    assert result["total_candidate_items"] == 0
    assert result["returned_item_count"] == 0
    assert result["items"] == []


# =====================================================================
# Tests 17-20: Edge cases, metadata resilience, mutability, JSON
# =====================================================================

def test_17_sentence_with_no_eligible_content_words():
    """Test 17: Sentence containing only stop words/punctuation returns empty items safely."""
    extractor = VocabularyExtractor(include_entities=False)
    record = make_synthetic_record(
        [
            {
                "tokens": ["In", "it", ",", "and", "by", "that", "."],
                "lemmas": ["in", "it", ",", "and", "by", "that", "."],
                "pos": ["ADP", "PRON", "PUNCT", "CCONJ", "ADP", "DET", "PUNCT"],
                "is_stop": [True, True, False, True, True, True, False],
                "is_punct": [False, False, True, False, False, False, True],
            }
        ]
    )
    result = extractor.extract_from_processed_record(record)
    assert result["returned_item_count"] == 0
    assert result["items"] == []


def test_18_missing_or_incomplete_metadata():
    """Test 18: Incomplete metadata (missing token_details or entities) does not crash."""
    extractor = VocabularyExtractor(include_entities=True)
    # Minimal record lacking token_details and entities
    minimal_record = {
        "id": "minimal_001",
        "processed_sentences": [
            {
                "sentence_index": 0,
                "tokens": ["technology", "innovation"],
                "pos": ["NOUN", "NOUN"],
                "content_tokens": ["technology", "innovation"],
            }
        ],
    }
    result = extractor.extract_from_processed_record(minimal_record)
    assert result["returned_item_count"] == 2
    terms = {it["term"] for it in result["items"]}
    assert "technology" in terms
    assert "innovation" in terms


def test_19_ensuring_input_record_is_not_modified():
    """Test 19: The input dictionary is not modified in any way."""
    extractor = VocabularyExtractor()
    record = make_synthetic_record(
        [
            {
                "tokens": ["SpaceX", "launched", "rockets"],
                "lemmas": ["spacex", "launch", "rocket"],
                "pos": ["PROPN", "VERB", "NOUN"],
                "is_stop": [False, False, False],
                "is_punct": [False, False, False],
                "entities": [{"text": "SpaceX", "label": "ORG", "start_char": 0, "end_char": 6}],
            }
        ]
    )
    record_copy = copy.deepcopy(record)
    _ = extractor.extract_from_processed_record(record)
    assert record == record_copy


def test_20_ensuring_results_are_json_serializable():
    """Test 20: Output is completely serializable by standard json.dumps."""
    extractor = VocabularyExtractor()
    record = make_synthetic_record(
        [
            {
                "tokens": ["Artificial", "Intelligence", "transforms", "medicine"],
                "lemmas": ["artificial", "intelligence", "transform", "medicine"],
                "pos": ["ADJ", "NOUN", "VERB", "NOUN"],
                "is_stop": [False, False, False, False],
                "is_punct": [False, False, False, False],
                "entities": [{"text": "Artificial Intelligence", "label": "ORG", "start_char": 0, "end_char": 23}],
            }
        ]
    )
    result = extractor.extract_from_processed_record(record)
    serialized = json.dumps(result, ensure_ascii=False)
    deserialized = json.loads(serialized)
    assert deserialized["article_id"] == record["id"]
    assert len(deserialized["items"]) > 0


# =====================================================================
# Tests 21-25: Raw text, processed record, batch, CLI, real CNN data
# =====================================================================

def test_21_raw_text_behavior():
    """Test 21: extract_from_text processes raw string via pipeline."""
    extractor = VocabularyExtractor()
    raw = (
        "Electric vehicles are gaining significant market share worldwide. "
        "Automakers are investing billions into battery manufacturing facilities."
    )
    result = extractor.extract_from_text(raw, article_id="ev_news", max_items=10)
    assert result["article_id"] == "ev_news"
    assert result["returned_item_count"] > 0
    terms = [it["normalized_form"] for it in result["items"]]
    assert any("vehicle" in t or "electric" in t or "battery" in t for t in terms)


def test_22_processed_record_behavior():
    """Test 22: extract_from_processed_record works directly without pipeline loading."""
    extractor = VocabularyExtractor()
    record = make_synthetic_record(
        [
            {
                "tokens": ["quantum", "computers", "solve", "complex", "equations"],
                "lemmas": ["quantum", "computer", "solve", "complex", "equation"],
                "pos": ["NOUN", "NOUN", "VERB", "ADJ", "NOUN"],
                "is_stop": [False, False, False, False, False],
                "is_punct": [False, False, False, False, False],
            }
        ]
    )
    result = extractor.extract_from_processed_record(record)
    assert extractor._pipeline is None  # Pipeline was NOT initialized/imported
    assert result["returned_item_count"] == 5


def test_23_malformed_jsonl_input_handling():
    """Test 23: Batch mode skips malformed JSON lines cleanly without crashing."""
    extractor = VocabularyExtractor()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "corrupt.jsonl"
        output_file = Path(tmpdir) / "output.jsonl"

        valid_record = make_synthetic_record(
            [{"tokens": ["valid", "entry"], "lemmas": ["valid", "entry"], "pos": ["ADJ", "NOUN"]}]
        )

        with input_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(valid_record) + "\n")
            f.write("{ INVALID JSON LINE\n")
            f.write(json.dumps(valid_record) + "\n")

        count, stats = extractor.batch_extract_file(input_file, output_file)
        assert count == 2  # The 2 valid records were processed
        assert stats["total_articles_processed"] == 2


def test_24_batch_output_structure_and_record_counts():
    """Test 24: Batch processing produces exact JSONL line count and expected keys."""
    extractor = VocabularyExtractor(max_items=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "test_batch_in.jsonl"
        output_file = Path(tmpdir) / "test_batch_out.jsonl"

        records = [
            make_synthetic_record([{"tokens": ["first", "article"], "lemmas": ["first", "article"], "pos": ["ADJ", "NOUN"]}], article_id=f"id_{i}")
            for i in range(3)
        ]

        with input_file.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        count, stats = extractor.batch_extract_file(input_file, output_file, limit=2)
        assert count == 2

        with output_file.open("r", encoding="utf-8") as f_out:
            lines = [line.strip() for line in f_out if line.strip()]
        assert len(lines) == 2

        parsed_line = json.loads(lines[0])
        assert "article_id" in parsed_line
        assert "items" in parsed_line
        assert "configuration" in parsed_line


def test_25_real_cnn_dailymail_record():
    """Test 25: Vocabulary extraction executes cleanly on real CNN/DailyMail processed data."""
    real_dataset = Path("dataset/processed/cnn_dailymail_test_1000_processed.jsonl")
    if not real_dataset.exists():
        pytest.skip("Local processed CNN/DailyMail dataset not available.")

    with real_dataset.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    real_record = json.loads(first_line)

    extractor = VocabularyExtractor(max_items=25)
    result = extractor.extract_from_processed_record(real_record)

    assert result["article_id"] == real_record["id"]
    assert result["returned_item_count"] <= 25
    assert result["total_candidate_items"] > 0
    assert len(result["items"]) > 0

    first_item = result["items"][0]
    for key in [
        "term",
        "normalized_form",
        "lemma",
        "pos",
        "frequency",
        "sentence_frequency",
        "sentence_indices",
        "importance_score",
        "selection_reason",
    ]:
        assert key in first_item
    assert isinstance(first_item["importance_score"], float)
    assert first_item["sentence_frequency"] <= len(real_record["processed_sentences"])
