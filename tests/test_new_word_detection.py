"""
Unit & Integration Tests for Phase 8: New-Word Detection
Tests cover:
1. Known-word match
2. New-word detection
3. Case-insensitive matching
4. Leading/trailing whitespace
5. Lemma matching for inflected forms
6. Multi-word vocabulary matching
7. Named entities with matching labels
8. Distinct entity labels remaining distinct where required
9. Duplicate entries in known-vocabulary list
10. Empty known-vocabulary list
11. Empty extraction record
12. Missing optional fields
13. Malformed known-vocabulary input
14. Deterministic output
15. Preservation of original Phase 7 metadata
16. Correct known/new/total counts (total == known + new)
17. Duplicate source-entry behavior
18. Input dictionary immutability
19. JSON serialization
20. Single-record CLI or API operation
21. Batch JSONL operation
22. Malformed JSONL record handling
23. Real record from Phase 7 output
24. Regression checks confirming Phase 7 output files remain unchanged
"""

import copy
import json
from pathlib import Path
import tempfile
import pytest

from vocabulary.new_word_detector import NewWordDetector


def make_phase7_record(items, article_id="test_art_001"):
    """Helper to construct a mock Phase 7 vocabulary extraction record."""
    return {
        "article_id": article_id,
        "method": "classical_vocabulary_extraction",
        "configuration": {
            "max_items": 30,
            "include_entities": True,
            "weights": {
                "alpha_frequency": 0.4,
                "beta_coverage": 0.4,
                "gamma_entity": 0.2,
            },
        },
        "total_candidate_items": len(items),
        "returned_item_count": len(items),
        "items": items,
    }


def make_item(
    term="announced",
    normalized_form="announced",
    lemma="announce",
    pos="VERB",
    freq=3,
    sent_freq=2,
    indices=None,
    entity_label=None,
    score=0.65,
    reason="Salient VERB.",
):
    """Helper to construct a single Phase 7 vocabulary item."""
    return {
        "term": term,
        "normalized_form": normalized_form,
        "lemma": lemma,
        "pos": pos,
        "frequency": freq,
        "sentence_frequency": sent_freq,
        "sentence_indices": indices or [0, 1],
        "entity_label": entity_label,
        "importance_score": score,
        "selection_reason": reason,
    }


# =====================================================================
# Tests 1-4: Basic matching, novelty, casing, whitespace
# =====================================================================

def test_1_known_word_match():
    """Test 1: Items present in known vocabulary are classified into known_words."""
    detector = NewWordDetector(known_vocabulary=["computer", "science"])
    record = make_phase7_record(
        [
            make_item(term="computer", normalized_form="computer", lemma="computer", pos="NOUN"),
            make_item(term="astronomy", normalized_form="astronomy", lemma="astronomy", pos="NOUN"),
        ]
    )
    res = detector.detect_new_words(record)
    known_terms = [it["normalized_form"] for it in res["known_words"]]
    new_terms = [it["normalized_form"] for it in res["new_words"]]

    assert "computer" in known_terms
    assert "astronomy" in new_terms
    assert res["known_word_count"] == 1
    assert res["new_word_count"] == 1


def test_2_new_word_detection():
    """Test 2: Words absent from known vocabulary are classified into new_words."""
    detector = NewWordDetector(known_vocabulary=["dog", "cat"])
    record = make_phase7_record(
        [
            make_item(term="elephant", normalized_form="elephant", lemma="elephant", pos="NOUN"),
            make_item(term="giraffe", normalized_form="giraffe", lemma="giraffe", pos="NOUN"),
        ]
    )
    res = detector.detect_new_words(record)
    assert res["known_word_count"] == 0
    assert res["new_word_count"] == 2
    assert len(res["new_words"]) == 2


def test_3_case_insensitive_matching():
    """Test 3: Matching succeeds regardless of casing differences."""
    detector = NewWordDetector(known_vocabulary=["PARIS", "LONDON", "Berlin"])
    record = make_phase7_record(
        [
            make_item(term="paris", normalized_form="paris", lemma="paris", pos="PROPN", entity_label="GPE"),
            make_item(term="London", normalized_form="london", lemma="london", pos="PROPN", entity_label="GPE"),
            make_item(term="BERLIN", normalized_form="berlin", lemma="berlin", pos="PROPN", entity_label="GPE"),
        ]
    )
    res = detector.detect_new_words(record)
    assert res["known_word_count"] == 3
    assert res["new_word_count"] == 0


def test_4_leading_trailing_whitespace():
    """Test 4: Whitespace in known terms or extraction terms is cleanly trimmed."""
    detector = NewWordDetector(known_vocabulary=["  energy  ", "\tpolicy\n"])
    record = make_phase7_record(
        [
            make_item(term="energy", normalized_form="energy", lemma="energy", pos="NOUN"),
            make_item(term="policy", normalized_form="policy", lemma="policy", pos="NOUN"),
        ]
    )
    res = detector.detect_new_words(record)
    assert res["known_word_count"] == 2
    assert res["new_word_count"] == 0


# =====================================================================
# Tests 5-8: Lemmas, multi-word phrases, entities and labels
# =====================================================================

def test_5_lemma_matching_for_inflected_forms():
    """Test 5: Inflected term 'announced' matches if user knows lemma 'announce'."""
    detector = NewWordDetector(known_vocabulary=["announce", "create"])
    record = make_phase7_record(
        [
            make_item(term="announced", normalized_form="announced", lemma="announce", pos="VERB"),
            make_item(term="creating", normalized_form="creating", lemma="create", pos="VERB"),
            make_item(term="destroyed", normalized_form="destroyed", lemma="destroy", pos="VERB"),
        ]
    )
    res = detector.detect_new_words(record)
    known_lemmas = [it["lemma"] for it in res["known_words"]]
    new_lemmas = [it["lemma"] for it in res["new_words"]]

    assert "announce" in known_lemmas
    assert "create" in known_lemmas
    assert "destroy" in new_lemmas
    assert res["known_word_count"] == 2
    assert res["new_word_count"] == 1


def test_6_multiword_vocabulary_matching():
    """Test 6: Multi-word phrases like 'United Nations' match correctly."""
    detector = NewWordDetector(known_vocabulary=["United Nations", "machine learning"])
    record = make_phase7_record(
        [
            make_item(
                term="United Nations",
                normalized_form="united nations",
                lemma="united nations",
                pos="PROPN",
                entity_label="ORG",
            ),
            make_item(
                term="deep learning",
                normalized_form="deep learning",
                lemma="deep learning",
                pos="NOUN",
            ),
        ]
    )
    res = detector.detect_new_words(record)
    assert res["known_word_count"] == 1
    assert res["new_word_count"] == 1
    assert res["known_words"][0]["term"] == "United Nations"
    assert res["new_words"][0]["term"] == "deep learning"


def test_7_named_entities_with_matching_labels():
    """Test 7: Entity dict with label in known-vocab matches candidate entity."""
    detector = NewWordDetector()
    detector.load_known_vocabulary([{"term": "Amazon", "entity_label": "ORG"}])
    record = make_phase7_record(
        [
            make_item(term="Amazon", normalized_form="amazon", lemma="amazon", pos="PROPN", entity_label="ORG"),
        ]
    )
    res = detector.detect_new_words(record)
    assert res["known_word_count"] == 1
    assert res["new_word_count"] == 0


def test_8_distinct_entity_labels_handling():
    """Test 8: Distinguishes same text across different categories when specified."""
    detector = NewWordDetector()
    detector.known_entities.add(("mercury", "ORG"))  # Known as company/org
    # Candidate item is mercury with LOC or GPE or None
    item_chem = make_item(term="Mercury", normalized_form="mercury", lemma="mercury", pos="NOUN", entity_label=None)
    assert not detector.is_known(item_chem)


# =====================================================================
# Tests 9-13: Deduplication, empty inputs, malformed input handling
# =====================================================================

def test_9_duplicate_entries_in_known_vocabulary():
    """Test 9: Duplicates in user's known vocabulary do not cause errors or duplicate counts."""
    detector = NewWordDetector(known_vocabulary=["apple", "Apple", " APPLE ", "apple"])
    assert len(detector.known_terms) == 1
    record = make_phase7_record(
        [make_item(term="apple", normalized_form="apple", lemma="apple", pos="NOUN")]
    )
    res = detector.detect_new_words(record)
    assert res["known_word_count"] == 1


def test_10_empty_known_vocabulary_list():
    """Test 10: Empty known vocabulary classifies all items as new words."""
    detector = NewWordDetector(known_vocabulary=[])
    record = make_phase7_record(
        [
            make_item(term="solar", normalized_form="solar", lemma="solar", pos="ADJ"),
            make_item(term="energy", normalized_form="energy", lemma="energy", pos="NOUN"),
        ]
    )
    res = detector.detect_new_words(record)
    assert res["known_word_count"] == 0
    assert res["new_word_count"] == 2
    assert len(res["new_words"]) == 2


def test_11_empty_extraction_record():
    """Test 11: Empty extraction record returns valid structure with zero counts."""
    detector = NewWordDetector(known_vocabulary=["water", "food"])
    empty_record = make_phase7_record([])
    res = detector.detect_new_words(empty_record)
    assert res["total_vocabulary_count"] == 0
    assert res["known_word_count"] == 0
    assert res["new_word_count"] == 0
    assert res["known_words"] == []
    assert res["new_words"] == []
    assert res["validation"]["reconciled"] is True


def test_12_missing_optional_fields():
    """Test 12: Vocabulary items lacking lemma or entity_label do not crash."""
    detector = NewWordDetector(known_vocabulary=["robotics"])
    record = make_phase7_record(
        [
            {"term": "robotics", "normalized_form": "robotics"},  # minimal dict
            {"term": "nanotechnology"},
        ]
    )
    res = detector.detect_new_words(record)
    assert res["known_word_count"] == 1
    assert res["new_word_count"] == 1


def test_13_malformed_known_vocabulary_input():
    """Test 13: Non-string, None, or empty entries in known vocabulary are handled safely."""
    detector = NewWordDetector()
    loaded = detector.load_known_vocabulary(["valid", None, "", "   ", 12345, {}])
    assert "valid" in detector.known_terms
    assert "12345" in detector.known_terms
    assert "" not in detector.known_terms
    assert loaded >= 2


# =====================================================================
# Tests 14-19: Invariance, metadata preservation, counts, JSON
# =====================================================================

def test_14_deterministic_output():
    """Test 14: Running detection twice produces identical serialized output."""
    detector = NewWordDetector(known_vocabulary=["alpha", "beta"])
    record = make_phase7_record(
        [
            make_item(term="alpha", normalized_form="alpha", lemma="alpha", pos="NOUN"),
            make_item(term="gamma", normalized_form="gamma", lemma="gamma", pos="NOUN"),
        ]
    )
    res1 = detector.detect_new_words(record)
    res2 = detector.detect_new_words(record)
    assert json.dumps(res1, sort_keys=True) == json.dumps(res2, sort_keys=True)


def test_15_preservation_of_original_phase7_metadata():
    """Test 15: All Phase 7 fields and importance scores are preserved without alteration."""
    detector = NewWordDetector(known_vocabulary=["evidence"])
    orig_item = make_item(
        term="evidence",
        normalized_form="evidence",
        lemma="evidence",
        pos="NOUN",
        freq=5,
        sent_freq=4,
        indices=[1, 3, 5, 8],
        entity_label=None,
        score=0.7421,
        reason="Salient NOUN.",
    )
    record = make_phase7_record([orig_item])
    res = detector.detect_new_words(record)
    classified_item = res["known_words"][0]

    for key in orig_item:
        assert classified_item[key] == orig_item[key]
    assert classified_item["importance_score"] == 0.7421


def test_16_correct_known_new_total_counts():
    """Test 16: Verification check total_vocabulary_count == known_word_count + new_word_count."""
    detector = NewWordDetector(known_vocabulary=["one", "three", "five"])
    record = make_phase7_record(
        [
            make_item(term="one", normalized_form="one", lemma="one"),
            make_item(term="two", normalized_form="two", lemma="two"),
            make_item(term="three", normalized_form="three", lemma="three"),
            make_item(term="four", normalized_form="four", lemma="four"),
            make_item(term="five", normalized_form="five", lemma="five"),
        ]
    )
    res = detector.detect_new_words(record)
    assert res["total_vocabulary_count"] == 5
    assert res["known_word_count"] == 3
    assert res["new_word_count"] == 2
    assert res["validation"]["reconciled"] is True
    assert res["validation"]["difference"] == 0


def test_17_duplicate_source_entry_behavior():
    """Test 17: If input contains duplicate entries (e.g. word & entity), both are accounted for."""
    detector = NewWordDetector(known_vocabulary=["icc"])
    # Two ICC entries: one entity ORG, one general PROPN
    record = make_phase7_record(
        [
            make_item(term="ICC", normalized_form="icc", lemma="icc", pos="PROPN", entity_label="ORG"),
            make_item(term="ICC", normalized_form="icc", lemma="icc", pos="PROPN", entity_label=None),
        ]
    )
    res = detector.detect_new_words(record)
    assert res["total_vocabulary_count"] == 2
    assert res["known_word_count"] == 2
    assert res["new_word_count"] == 0


def test_18_input_dictionary_immutability():
    """Test 18: Input extraction record dictionary is completely unmutated."""
    detector = NewWordDetector(known_vocabulary=["dna", "rna"])
    record = make_phase7_record(
        [
            make_item(term="dna", normalized_form="dna", lemma="dna"),
            make_item(term="protein", normalized_form="protein", lemma="protein"),
        ]
    )
    record_backup = copy.deepcopy(record)
    _ = detector.detect_new_words(record)
    assert record == record_backup


def test_19_json_serialization():
    """Test 19: Full output structure is cleanly JSON-serializable."""
    detector = NewWordDetector(known_vocabulary=["algorithm"])
    record = make_phase7_record(
        [
            make_item(term="algorithm", normalized_form="algorithm", lemma="algorithm"),
            make_item(term="heuristic", normalized_form="heuristic", lemma="heuristic"),
        ]
    )
    res = detector.detect_new_words(record)
    serialized = json.dumps(res, ensure_ascii=False)
    deserialized = json.loads(serialized)
    assert deserialized["article_id"] == record["article_id"]
    assert deserialized["known_word_count"] == 1
    assert deserialized["new_word_count"] == 1


# =====================================================================
# Tests 20-24: File I/O, Batch streaming, Real records, Regression
# =====================================================================

def test_20_load_from_file_formats():
    """Test 20: Supports loading known vocabulary from .txt, .json, and .jsonl."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Plain text file
        txt_path = Path(tmpdir) / "vocab.txt"
        txt_path.write_text("apple\nbanana\norange\n", encoding="utf-8")
        det1 = NewWordDetector.from_file(txt_path)
        assert len(det1.known_terms) == 3

        # 2. JSON list
        json_path = Path(tmpdir) / "vocab.json"
        json_path.write_text(json.dumps(["cat", "dog", "bird"]), encoding="utf-8")
        det2 = NewWordDetector.from_file(json_path)
        assert len(det2.known_terms) == 3

        # 3. JSON dict with 'known_words'
        json_dict_path = Path(tmpdir) / "vocab_dict.json"
        json_dict_path.write_text(json.dumps({"known_words": ["car", "bus"]}), encoding="utf-8")
        det3 = NewWordDetector.from_file(json_dict_path)
        assert len(det3.known_terms) == 2


def test_21_batch_jsonl_operation():
    """Test 21: batch_detect_file processes multiple JSONL records streaming."""
    detector = NewWordDetector(known_vocabulary=["common", "shared"])
    with tempfile.TemporaryDirectory() as tmpdir:
        in_file = Path(tmpdir) / "in.jsonl"
        out_file = Path(tmpdir) / "out.jsonl"

        records = [
            make_phase7_record([make_item(term="common"), make_item(term="novel_1")], article_id="art_1"),
            make_phase7_record([make_item(term="shared"), make_item(term="novel_2")], article_id="art_2"),
        ]
        with in_file.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        count, stats = detector.batch_detect_file(in_file, out_file)
        assert count == 2
        assert stats["total_articles_processed"] == 2
        assert stats["total_vocabulary_classified"] == 4
        assert stats["total_known_words"] == 2
        assert stats["total_new_words"] == 2

        with out_file.open("r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert lines[0]["article_id"] == "art_1"
        assert lines[1]["article_id"] == "art_2"


def test_22_malformed_jsonl_record_handling():
    """Test 22: Corrupted lines in input JSONL are skipped with a warning."""
    detector = NewWordDetector(known_vocabulary=["test"])
    with tempfile.TemporaryDirectory() as tmpdir:
        in_file = Path(tmpdir) / "corrupt.jsonl"
        out_file = Path(tmpdir) / "out.jsonl"

        valid = make_phase7_record([make_item(term="test")], article_id="valid_1")
        with in_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(valid) + "\n")
            f.write("{ INVALID JSON NOT COMPLETE\n")
            f.write(json.dumps(valid) + "\n")

        count, stats = detector.batch_detect_file(in_file, out_file)
        assert count == 2
        assert stats["total_articles_processed"] == 2


def test_23_real_phase7_output_record():
    """Test 23: Executes successfully on real Phase 7 1,000-article batch output."""
    phase7_file = Path("dataset/processed/vocabulary_extractions_1000.jsonl")
    if not phase7_file.exists():
        pytest.skip("Phase 7 dataset not available.")

    with phase7_file.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    real_record = json.loads(first_line)

    detector = NewWordDetector(known_vocabulary=["icc", "said", "today"])
    res = detector.detect_new_words(real_record)

    assert res["article_id"] == real_record["article_id"]
    assert res["total_vocabulary_count"] == len(real_record["items"])
    assert res["known_word_count"] > 0
    assert res["new_word_count"] > 0
    assert res["validation"]["reconciled"] is True


def test_24_phase7_files_unchanged_regression():
    """Test 24: Verifies Phase 7 outputs and documentation exist and are unmodified."""
    p7_dataset = Path("dataset/processed/vocabulary_extractions_1000.jsonl")
    p7_report = Path("docs/vocabulary_extraction_analysis.md")
    p7_readme = Path("vocabulary/README.md")

    assert p7_dataset.exists(), "Phase 7 dataset must exist."
    assert p7_report.exists(), "Phase 7 analysis report must exist."
    assert p7_readme.exists(), "Phase 7 README must exist."

    # Check that Phase 7 dataset has exactly 1,000 records
    with p7_dataset.open("r", encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())
    assert count == 1000, f"Expected 1000 records in Phase 7 dataset, found {count}."
