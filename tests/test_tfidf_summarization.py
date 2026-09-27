"""
Tests for Phase 4: TF-IDF-Based Extractive Summarization

Validates:
1. TF calculation (formula, counts, normalization by sentence content terms)
2. DF calculation (document frequency across sentences, counted at most once per sentence)
3. IDF calculation (log(N/DF), zero values for corpus-wide terms, non-negative bounds)
4. TF-IDF calculation (product of TF and IDF)
5. Sentence scoring (length-normalized average TF-IDF of content terms)
6. Hand-calculated test case from Section 14 (cat dog, cat mouse, cat bird)
7. Zero-content-word sentence handling (score = 0.0, pushed below scored sentences)
8. Sentence ranking (descending score order)
9. Deterministic tie-breaking (earlier sentence index wins ties)
10. Top-K sentence selection
11. Original sentence restoration (exact verbatim text preserved)
12. Original sentence chronological ordering (summary preserves discourse flow)
13. Deterministic output reproducibility
14. Single article summarization via raw text pipeline
15. Real CNN/DailyMail record summarization from processed dataset
"""

import json
import math
from pathlib import Path
import pytest

from summarization.tfidf import TFIDFSummarizer


# ==============================================================================
# Hand-Calculated Fixtures and Ground Truths
# ==============================================================================

@pytest.fixture
def hand_calculated_sentences():
    """
    3-sentence hand-calculated example from Section 14:
    Sentence 0: cat dog
    Sentence 1: cat mouse
    Sentence 2: cat bird
    """
    return [
        {
            "sentence_index": 0,
            "original_text": "The cat and the dog played outside.",
            "content_tokens": ["cat", "dog"],
        },
        {
            "sentence_index": 1,
            "original_text": "A cat chased the mouse across the room.",
            "content_tokens": ["cat", "mouse"],
        },
        {
            "sentence_index": 2,
            "original_text": "The cat watched the bird on the tree.",
            "content_tokens": ["cat", "bird"],
        },
    ]


@pytest.fixture
def real_cnn_processed_record():
    """Load first real CNN/DailyMail processed record from dataset."""
    processed_path = Path("dataset/processed/cnn_dailymail_test_1000_processed.jsonl")
    if not processed_path.exists():
        pytest.skip("Processed dataset not available on disk.")
    with open(processed_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            pytest.skip("Empty dataset file.")
        return json.loads(first_line)


# ==============================================================================
# Unit Tests
# ==============================================================================

def test_tf_calculation():
    """
    Test 1: TF calculation
    TF(t, d) = count(t in d) / total number of content terms in d
    """
    summarizer = TFIDFSummarizer()
    tokens = ["cat", "dog", "cat", "bird"]
    tf = summarizer.calculate_tf(tokens)

    # 4 content tokens total: cat=2, dog=1, bird=1
    assert len(tf) == 3
    assert tf["cat"] == pytest.approx(2 / 4, rel=1e-5)
    assert tf["dog"] == pytest.approx(1 / 4, rel=1e-5)
    assert tf["bird"] == pytest.approx(1 / 4, rel=1e-5)
    assert sum(tf.values()) == pytest.approx(1.0, rel=1e-5)

    # Empty list edge case
    assert summarizer.calculate_tf([]) == {}


def test_df_calculation():
    """
    Test 2: DF calculation
    DF(t) = number of sentences containing term t (at most once per sentence)
    """
    summarizer = TFIDFSummarizer()
    sentences = [
        ["cat", "dog", "dog"],  # dog appears twice, but DF should only count once
        ["cat", "mouse"],
        ["cat", "bird"],
    ]
    df = summarizer.calculate_df(sentences)

    assert df["cat"] == 3
    assert df["dog"] == 1
    assert df["mouse"] == 1
    assert df["bird"] == 1


def test_idf_calculation():
    """
    Test 3: IDF calculation
    IDF(t) = log(N / DF(t))
    """
    summarizer = TFIDFSummarizer()
    df = {"cat": 3, "dog": 1, "mouse": 1}
    num_sentences = 3

    idf = summarizer.calculate_idf(df, num_sentences, smooth=False)

    # cat appears in all 3 sentences: log(3/3) = log(1) = 0.0
    assert idf["cat"] == pytest.approx(0.0, abs=1e-6)
    # dog and mouse appear in 1 sentence: log(3/1) = log(3) ≈ 1.098612
    expected_dog_idf = math.log(3.0)
    assert idf["dog"] == pytest.approx(expected_dog_idf, rel=1e-5)
    assert idf["mouse"] == pytest.approx(expected_dog_idf, rel=1e-5)

    # Verify smoothed IDF option
    idf_smooth = summarizer.calculate_idf(df, num_sentences, smooth=True)
    # log(1 + 3/3) = log(2)
    assert idf_smooth["cat"] == pytest.approx(math.log(2.0), rel=1e-5)
    # log(1 + 3/1) = log(4)
    assert idf_smooth["dog"] == pytest.approx(math.log(4.0), rel=1e-5)


def test_tfidf_calculation():
    """
    Test 4: TF-IDF calculation
    TFIDF(t, d) = TF(t, d) * IDF(t)
    """
    summarizer = TFIDFSummarizer()
    tf = {"cat": 0.5, "dog": 0.5}
    idf = {"cat": 0.0, "dog": math.log(3.0)}

    tfidf = summarizer.calculate_tfidf(tf, idf)

    assert tfidf["cat"] == pytest.approx(0.0, abs=1e-6)
    assert tfidf["dog"] == pytest.approx(0.5 * math.log(3.0), rel=1e-5)


def test_sentence_scoring():
    """
    Test 5: Sentence scoring
    SentenceScore(S) = sum(TFIDF values of content terms in S) / number of content terms in S
    """
    summarizer = TFIDFSummarizer()
    idf = {"cat": 0.0, "dog": math.log(3.0)}
    tokens = ["cat", "dog"]

    score = summarizer.score_sentence(tokens, idf)

    # tf = {cat: 0.5, dog: 0.5}
    # tfidf = {cat: 0.0, dog: 0.5 * log(3)}
    # sum = 0.5 * log(3)
    # score = (0.5 * log(3)) / 2 = 0.25 * log(3) ≈ 0.274653
    expected = 0.25 * math.log(3.0)
    assert score == pytest.approx(expected, rel=1e-4)


def test_hand_calculated_example(hand_calculated_sentences):
    """
    Test 6: Hand-calculated test case from Section 14
    Sentence 0: cat dog
    Sentence 1: cat mouse
    Sentence 2: cat bird
    """
    summarizer = TFIDFSummarizer()
    record = {
        "id": "hand_test_001",
        "processed_sentences": hand_calculated_sentences,
        "highlights": "Hand calculated ground truth summary.",
    }

    result = summarizer.summarize_processed_record(record, num_sentences=2)

    # Check DF
    df = summarizer.calculate_df([s["content_tokens"] for s in hand_calculated_sentences])
    assert df == {"cat": 3, "dog": 1, "mouse": 1, "bird": 1}

    # Check IDF
    idf = summarizer.calculate_idf(df, num_sentences=3)
    assert idf["cat"] == pytest.approx(0.0, abs=1e-6)
    assert idf["dog"] == pytest.approx(math.log(3.0), rel=1e-5)
    assert idf["mouse"] == pytest.approx(math.log(3.0), rel=1e-5)
    assert idf["bird"] == pytest.approx(math.log(3.0), rel=1e-5)

    # Check Sentence Scores:
    # All 3 sentences have 1 shared word (cat, IDF=0) and 1 distinctive word (IDF=log(3))
    # Score = (0.5 * log(3)) / 2 = 0.25 * log(3) ≈ 0.274653
    expected_score = round(0.25 * math.log(3.0), 4)
    for s in result["selected_sentences"]:
        assert s["score"] == pytest.approx(expected_score, abs=1e-3)

    # Due to deterministic tie-breaking, earlier indices (0, 1) win over index 2
    selected_indices = [s["sentence_index"] for s in result["selected_sentences"]]
    assert selected_indices == [0, 1]


def test_zero_content_word_sentence():
    """
    Test 7: Sentence with zero content terms receives score = 0.0
    and is pushed below sentences with content terms.
    """
    summarizer = TFIDFSummarizer()
    sentences = [
        {
            "sentence_index": 0,
            "original_text": "In fact, however.",
            "content_tokens": [],
        },
        {
            "sentence_index": 1,
            "original_text": "The economy rebounded strongly.",
            "content_tokens": ["economy", "rebounded", "strongly"],
        },
    ]

    candidates, df, idf = summarizer.rank_sentences(sentences)

    assert candidates[0]["sentence_index"] == 1
    assert candidates[0]["score"] > 0.0
    assert candidates[1]["sentence_index"] == 0
    assert candidates[1]["score"] == 0.0


def test_sentence_ranking():
    """
    Test 8: Sentence ranking orders by descending TF-IDF score.
    """
    summarizer = TFIDFSummarizer()
    sentences = [
        {
            "sentence_index": 0,
            "original_text": "Common common common.",
            "content_tokens": ["common", "common"],
        },
        {
            "sentence_index": 1,
            "original_text": "Distinctive unique terminology.",
            "content_tokens": ["distinctive", "unique"],
        },
        {
            "sentence_index": 2,
            "original_text": "Common term here.",
            "content_tokens": ["common", "term"],
        },
    ]

    candidates, df, idf = summarizer.rank_sentences(sentences)

    # Sentence 1 has purely rare words appearing only once across the 3 sentences
    # Sentence 1 must rank #1 (score: 0.5493 vs 0.3760 for Sent 2 vs 0.2027 for Sent 0)
    assert candidates[0]["sentence_index"] == 1
    assert candidates[0]["score"] > candidates[1]["score"]
    assert candidates[1]["score"] >= candidates[2]["score"]


def test_deterministic_tie_breaking():
    """
    Test 9: Deterministic tie-breaking ensures earlier sentence index wins when scores are equal.
    """
    summarizer = TFIDFSummarizer()
    # Sentences with identical content structure
    sentences = [
        {
            "sentence_index": 5,
            "original_text": "Alpha beta.",
            "content_tokens": ["shared", "alpha"],
        },
        {
            "sentence_index": 2,
            "original_text": "Gamma delta.",
            "content_tokens": ["shared", "gamma"],
        },
    ]

    candidates, _, _ = summarizer.rank_sentences(sentences)
    assert candidates[0]["score"] == candidates[1]["score"]
    # Index 2 must win over index 5
    assert candidates[0]["sentence_index"] == 2
    assert candidates[1]["sentence_index"] == 5


def test_top_k_selection():
    """
    Test 10: Top-K selection properly respects requested sentence count.
    """
    summarizer = TFIDFSummarizer()
    sents = [
        {"sentence_index": i, "original_text": f"Sentence {i}.", "content_tokens": [f"word_{i}"]}
        for i in range(10)
    ]
    record = {"id": "rec_10", "processed_sentences": sents}

    res_3 = summarizer.summarize_processed_record(record, num_sentences=3)
    assert res_3["num_sentences_selected"] == 3
    assert len(res_3["selected_sentences"]) == 3

    res_5 = summarizer.summarize_processed_record(record, num_sentences=5)
    assert res_5["num_sentences_selected"] == 5

    # Request more sentences than exist: should clamp to available sentences
    res_20 = summarizer.summarize_processed_record(record, num_sentences=20)
    assert res_20["num_sentences_selected"] == 10


def test_original_sentence_restoration():
    """
    Test 11: Summary must return verbatim original sentence text,
    not tokens, lemmas, or lowercase text.
    """
    summarizer = TFIDFSummarizer()
    verbatim_text = "The United States announced a $5.2 billion package on Wednesday!"
    sentences = [
        {
            "sentence_index": 0,
            "original_text": verbatim_text,
            "content_tokens": ["package", "wednesday"],
        },
        {
            "sentence_index": 1,
            "original_text": "Officials praised the swift decision.",
            "content_tokens": ["common", "common", "common"],
        },
        {
            "sentence_index": 2,
            "original_text": "Another sentence.",
            "content_tokens": ["common"],
        },
    ]
    record = {"id": "orig_test", "processed_sentences": sentences}

    result = summarizer.summarize_processed_record(record, num_sentences=1)

    assert len(result["selected_sentences"]) == 1
    selected_sent = result["selected_sentences"][0]
    # Verify verbatim original text preservation
    assert selected_sent["original_text"] == verbatim_text
    assert selected_sent["text"] == verbatim_text
    assert verbatim_text in result["summary"]


def test_original_sentence_ordering():
    """
    Test 12: Chronological restoration ensures selected sentences appear
    in their original narrative order regardless of their ranking score order.
    """
    summarizer = TFIDFSummarizer()
    sentences = [
        {
            "sentence_index": 0,
            "original_text": "Introduction sentence.",
            "content_tokens": ["common", "common"],
        },
        {
            "sentence_index": 1,
            "original_text": "Middle sentence with very rare vocabulary.",
            "content_tokens": ["common", "distinct_a"],
        },
        {
            "sentence_index": 2,
            "original_text": "Conclusion sentence with distinctive terms.",
            "content_tokens": ["distinct_b", "distinct_c"],
        },
    ]
    record = {"id": "order_test", "processed_sentences": sentences}

    result = summarizer.summarize_processed_record(record, num_sentences=2)

    indices = [s["sentence_index"] for s in result["selected_sentences"]]
    # Indices must be strictly ascending
    assert indices == sorted(indices)
    assert indices == [1, 2]
    # Summary string must match chronological joining
    expected_summary = "Middle sentence with very rare vocabulary. Conclusion sentence with distinctive terms."
    assert result["summary"] == expected_summary


def test_deterministic_output():
    """
    Test 13: Multiple invocations with the same input produce 100% identical output.
    """
    summarizer = TFIDFSummarizer()
    record = {
        "id": "det_test",
        "processed_sentences": [
            {"sentence_index": 0, "original_text": "First sentence.", "content_tokens": ["first", "sentence"]},
            {"sentence_index": 1, "original_text": "Second sentence.", "content_tokens": ["second", "sentence"]},
            {"sentence_index": 2, "original_text": "Third sentence.", "content_tokens": ["third", "sentence"]},
        ],
    }

    res1 = summarizer.summarize_processed_record(record, num_sentences=2)
    res2 = summarizer.summarize_processed_record(record, num_sentences=2)

    assert res1["summary"] == res2["summary"]
    assert res1["selected_sentences"] == res2["selected_sentences"]


def test_single_article_raw_text_summarization():
    """
    Test 14: summarize_text() handles raw string input via pipeline.
    """
    summarizer = TFIDFSummarizer()
    raw = (
        "Artificial intelligence is transforming business operations across the globe. "
        "Modern companies use machine learning models to automate routine workflows. "
        "However, executive leaders must ensure robust data privacy and security governance."
    )

    result = summarizer.summarize_text(raw, num_sentences=2)

    assert result["num_sentences_selected"] == 2
    assert len(result["summary"]) > 0
    assert len(result["selected_sentences"]) == 2
    for s in result["selected_sentences"]:
        assert s["score"] > 0.0
        assert len(s["text"]) > 0


def test_real_cnn_dailymail_record_summarization(real_cnn_processed_record):
    """
    Test 15: Summarize a real CNN/DailyMail record from Phase 2.
    """
    summarizer = TFIDFSummarizer()
    res = summarizer.summarize_processed_record(real_cnn_processed_record, num_sentences=3)

    assert res["article_id"] == real_cnn_processed_record["id"]
    assert res["method"] == "tfidf"
    assert res["num_sentences_selected"] == 3
    assert len(res["selected_sentences"]) == 3
    assert len(res["summary"]) > 0
    assert res["summary"].count(".") >= 1

    # Check top terms
    assert len(res["top_terms"]) > 0
    for term in res["top_terms"]:
        assert "term" in term
        assert "tfidf" in term
        assert "df" in term
        assert "idf" in term
