"""
Unit Tests for Phase 3: Frequency-Based Extractive Summarization
Tests word frequency calculation, normalization, sentence scoring,
zero-content-word handling, ranking, top-K selection, tie-breaking,
original sentence restoration, sentence ordering, and real CNN/DailyMail summarization.
"""

import json
from pathlib import Path
import pytest

from summarization.frequency import FrequencySummarizer


@pytest.fixture(scope="module")
def summarizer():
    return FrequencySummarizer()


# 1. Word frequency calculation
def test_calculate_word_frequencies(summarizer):
    words = ["government", "policy", "government", "economic", "policy", "government"]
    freqs = summarizer.calculate_word_frequencies(words)
    assert freqs["government"] == 3
    assert freqs["policy"] == 2
    assert freqs["economic"] == 1
    assert len(freqs) == 3

    # Empty list handling
    assert summarizer.calculate_word_frequencies([]) == {}


# 2. Frequency normalization
def test_normalize_frequencies(summarizer):
    freqs = {"government": 4, "policy": 2, "economy": 1}
    norm_freqs = summarizer.normalize_frequencies(freqs)
    assert norm_freqs["government"] == 1.0  # 4 / 4
    assert norm_freqs["policy"] == 0.5      # 2 / 4
    assert norm_freqs["economy"] == 0.25    # 1 / 4

    # Empty dictionary
    assert summarizer.normalize_frequencies({}) == {}


# 3. Sentence scoring
def test_sentence_scoring(summarizer):
    norm_freqs = {"government": 1.0, "policy": 0.5, "announced": 0.25}
    content_tokens = ["government", "announced", "policy"]
    score = summarizer.score_sentence(content_tokens, norm_freqs)
    # Expected: (1.0 + 0.25 + 0.5) / 3 = 1.75 / 3 = 0.583333
    assert score == pytest.approx(0.583333, rel=1e-4)


# 4. Zero-content-word sentence handling
def test_zero_content_word_sentence_score(summarizer):
    norm_freqs = {"government": 1.0}
    assert summarizer.score_sentence([], norm_freqs) == 0.0
    assert summarizer.score_sentence(["", "   "], norm_freqs) == 0.0


# 5. Sentence ranking & top-K selection
def test_sentence_ranking_and_selection(summarizer):
    norm_freqs = {"high": 1.0, "medium": 0.5, "low": 0.1}
    processed_sents = [
        {"sentence_index": 0, "original_text": "Sentence zero with low word.", "content_tokens": ["low"]},
        {"sentence_index": 1, "original_text": "Sentence one with high word.", "content_tokens": ["high"]},
        {"sentence_index": 2, "original_text": "Sentence two with medium word.", "content_tokens": ["medium"]},
    ]

    ranked = summarizer.rank_sentences(processed_sents, norm_freqs)
    assert ranked[0]["sentence_index"] == 1  # high (score 1.0)
    assert ranked[1]["sentence_index"] == 2  # medium (score 0.5)
    assert ranked[2]["sentence_index"] == 0  # low (score 0.1)


# 6. Tie-breaking: earlier sentence wins
def test_deterministic_tie_breaking(summarizer):
    norm_freqs = {"target": 1.0}
    # Three sentences with identical content words and identical scores
    processed_sents = [
        {"sentence_index": 5, "original_text": "Sentence five.", "content_tokens": ["target"]},
        {"sentence_index": 2, "original_text": "Sentence two.", "content_tokens": ["target"]},
        {"sentence_index": 8, "original_text": "Sentence eight.", "content_tokens": ["target"]},
    ]

    ranked = summarizer.rank_sentences(processed_sents, norm_freqs)
    # Earlier sentence index must win the tie
    assert ranked[0]["sentence_index"] == 2
    assert ranked[1]["sentence_index"] == 5
    assert ranked[2]["sentence_index"] == 8


# 7. Original sentence restoration and original ordering
def test_original_sentence_restoration_and_ordering(summarizer):
    record = {
        "id": "synthetic_01",
        "processed_sentences": [
            {"sentence_index": 0, "original_text": "First sentence.", "content_tokens": ["common"]},
            {"sentence_index": 1, "original_text": "Second sentence.", "content_tokens": ["rare"]},
            {"sentence_index": 2, "original_text": "Third sentence.", "content_tokens": ["common", "important"]},
            {"sentence_index": 3, "original_text": "Fourth sentence.", "content_tokens": ["common", "important"]},
        ],
        "highlights": "Reference summary highlights.",
    }

    # Request top 2 sentences
    res = summarizer.summarize_processed_record(record, num_sentences=2)
    assert res["num_sentences_selected"] == 2

    # Selected sentences must be restored to original ascending chronological order
    # Sent 0 score = 1.0 (contains "common"), Sent 2 & 3 score = 0.8333, Sent 1 score = 0.3333
    # Top 2 sentences are Sent 0 and Sent 2 (Sentence 2 wins tie over 3 by earlier index)
    indices = [s["sentence_index"] for s in res["selected_sentences"]]
    assert indices == sorted(indices)
    assert indices == [0, 2]

    # Final summary text must contain exact original sentences joined with space
    assert res["summary"] == "First sentence. Third sentence."
    assert "First sentence." in res["summary"]
    assert "Third sentence." in res["summary"]



# 8. Determinism: multiple runs yield identical results
def test_deterministic_output(summarizer):
    record = {
        "id": "det_01",
        "processed_sentences": [
            {"sentence_index": 0, "original_text": "Alpha text.", "content_tokens": ["alpha", "beta"]},
            {"sentence_index": 1, "original_text": "Beta text.", "content_tokens": ["beta"]},
            {"sentence_index": 2, "original_text": "Gamma text.", "content_tokens": ["gamma", "beta"]},
        ],
    }
    run1 = summarizer.summarize_processed_record(record, num_sentences=2)
    run2 = summarizer.summarize_processed_record(record, num_sentences=2)
    assert run1["summary"] == run2["summary"]
    assert [s["score"] for s in run1["selected_sentences"]] == [s["score"] for s in run2["selected_sentences"]]


# 9. Single raw article summarization end-to-end
def test_single_article_summarization(summarizer):
    article_text = (
        "The United Nations convened in Geneva to discuss climate action. "
        "Global leaders emphasized that climate targets must be achieved by 2030. "
        "Economic representatives cautioned that renewable transition requires substantial funding. "
        "The climate conference concluded with a binding multilateral agreement."
    )
    result = summarizer.summarize_text(article_text, num_sentences=2)
    assert result["num_sentences_selected"] == 2
    assert len(result["summary"]) > 0
    # Summary should be composed of verbatim sentences from the input
    for sent in result["selected_sentences"]:
        assert sent["original_text"] in article_text


# 10. Real CNN/DailyMail record summarization
def test_real_cnn_dailymail_record_summarization(summarizer):
    proc_file = Path("dataset/processed/cnn_dailymail_test_1000_processed.jsonl")
    if not proc_file.exists():
        pytest.skip("Processed dataset not found")

    with proc_file.open("r", encoding="utf-8") as f:
        first_record = json.loads(f.readline())

    result = summarizer.summarize_processed_record(first_record, num_sentences=3)
    assert result["article_id"] == first_record["id"]
    assert result["num_sentences_selected"] == 3
    assert len(result["selected_sentences"]) == 3

    # Verify original sentence text is retained
    for s in result["selected_sentences"]:
        assert len(s["original_text"]) > 0
        assert s["original_text"] in first_record["cleaned_article"]

    # Verify chronological ordering
    selected_indices = [s["sentence_index"] for s in result["selected_sentences"]]
    assert selected_indices == sorted(selected_indices)
