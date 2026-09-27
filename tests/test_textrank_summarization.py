"""
Tests for Phase 5: TextRank Extractive Summarization

Validates:
1. TF calculation
2. DF calculation (counting at most once per sentence)
3. IDF calculation (ln(N / DF(t)))
4. TF-IDF sentence vectors (sparse representation)
5. Cosine similarity (identical -> 1, orthogonal -> 0, zero vector -> 0)
6. Symmetry: sim(A, B) == sim(B, A)
7. Similarity matrix (diagonal=0, symmetric, non-negative)
8. PageRank initialization (1/N uniform mass)
9. PageRank convergence on a connected graph
10. Dangling node handling (isolated sentence with zero similarity edges)
11. Deterministic ranking (score descending, lower index on tie)
12. Top-K selection (clamped to available sentences)
13. Original order restoration ([4, 1, 3] -> [1, 3, 4])
14. Original sentence preservation (exact verbatim source strings)
15. Deterministic output reproducibility
16. Single article summarization via raw text pipeline
17. Real CNN/DailyMail record summarization
18. Tiny hand-calculated graph test (Section 21: cats chase mice vs hunt mice vs football)
19. Edge case handling (single sentence, zero content tokens, empty text)
"""

import json
import math
from pathlib import Path
import pytest

from summarization.textrank import TextRankSummarizer


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


def test_tf_calculation():
    """Test 1: TF calculation per sentence."""
    summarizer = TextRankSummarizer()
    tokens = ["cat", "dog", "cat", "bird"]
    tf = summarizer.calculate_tf(tokens)

    assert len(tf) == 3
    assert tf["cat"] == pytest.approx(2 / 4, rel=1e-5)
    assert tf["dog"] == pytest.approx(1 / 4, rel=1e-5)
    assert tf["bird"] == pytest.approx(1 / 4, rel=1e-5)
    assert sum(tf.values()) == pytest.approx(1.0, rel=1e-5)
    assert summarizer.calculate_tf([]) == {}


def test_df_calculation():
    """Test 2: DF calculation counts each term at most once per sentence."""
    summarizer = TextRankSummarizer()
    sentences = [
        ["cat", "cat", "dog"],
        ["dog", "mouse"],
        ["bird"],
    ]
    df = summarizer.calculate_df(sentences)

    assert df["cat"] == 1
    assert df["dog"] == 2
    assert df["mouse"] == 1
    assert df["bird"] == 1


def test_idf_calculation():
    """Test 3: IDF calculation IDF(t) = ln(N / DF(t))."""
    summarizer = TextRankSummarizer()
    df = {"common": 4, "rare": 1}
    num_sentences = 4

    idf = summarizer.calculate_idf(df, num_sentences)

    # common appears in all 4 sentences: ln(4/4) = 0.0
    assert idf["common"] == pytest.approx(0.0, abs=1e-6)
    # rare appears in 1 sentence: ln(4/1) = ln(4)
    assert idf["rare"] == pytest.approx(math.log(4.0), rel=1e-5)


def test_tfidf_sentence_vectors():
    """Test 4: Sparse TF-IDF sentence vector construction."""
    summarizer = TextRankSummarizer()
    sentences = [
        ["cat", "dog"],
        ["cat", "mouse"],
        ["cat", "bird"],
    ]
    vectors, df, idf = summarizer.build_sentence_vectors(sentences)

    assert len(vectors) == 3
    # "cat" appears in all 3 sentences -> IDF = ln(3/3) = 0.0 -> TFIDF = 0.0
    assert vectors[0]["cat"] == pytest.approx(0.0, abs=1e-6)
    # "dog" appears in 1 sentence -> TF = 0.5, IDF = ln(3) -> TFIDF = 0.5 * ln(3)
    assert vectors[0]["dog"] == pytest.approx(0.5 * math.log(3.0), rel=1e-5)


def test_cosine_similarity():
    """Test 5: Cosine similarity properties (identical, orthogonal, zero magnitude)."""
    summarizer = TextRankSummarizer()

    # Identical vectors -> similarity 1.0
    v1 = {"a": 1.0, "b": 2.0}
    assert summarizer.cosine_similarity(v1, v1) == pytest.approx(1.0, rel=1e-5)

    # Orthogonal vectors -> similarity 0.0
    v2 = {"c": 1.0, "d": 2.0}
    assert summarizer.cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-6)

    # Zero magnitude vector -> similarity 0.0
    assert summarizer.cosine_similarity({}, v1) == 0.0
    assert summarizer.cosine_similarity(v1, {}) == 0.0


def test_cosine_similarity_symmetry():
    """Test 6: Symmetry: sim(A, B) == sim(B, A)."""
    summarizer = TextRankSummarizer()
    va = {"alpha": 0.4, "beta": 0.8, "gamma": 0.1}
    vb = {"beta": 0.5, "delta": 0.9}

    sim_ab = summarizer.cosine_similarity(va, vb)
    sim_ba = summarizer.cosine_similarity(vb, va)

    assert sim_ab == pytest.approx(sim_ba, rel=1e-6)
    assert 0.0 < sim_ab < 1.0


def test_similarity_matrix():
    """Test 7: Similarity matrix structure (diagonal=0, symmetric, non-negative)."""
    summarizer = TextRankSummarizer()
    vectors = [
        {"x": 1.0, "y": 0.5},
        {"y": 0.5, "z": 1.0},
        {"w": 2.0},
    ]
    matrix = summarizer.build_similarity_matrix(vectors)

    n = len(vectors)
    assert len(matrix) == n
    for i in range(n):
        assert len(matrix[i]) == n
        assert matrix[i][i] == 0.0  # No self-loops
        for j in range(n):
            assert matrix[i][j] >= 0.0  # Non-negative
            assert matrix[i][j] == pytest.approx(matrix[j][i], rel=1e-6)  # Symmetric


def test_pagerank_initialization():
    """Test 8: PageRank initialization PR(S_i) = 1/N."""
    summarizer = TextRankSummarizer()
    init_4 = summarizer.initialize_pagerank(4)
    assert len(init_4) == 4
    for val in init_4:
        assert val == pytest.approx(0.25, rel=1e-6)
    assert sum(init_4) == pytest.approx(1.0, rel=1e-6)


def test_pagerank_convergence():
    """Test 9: PageRank iterations converge on a connected graph."""
    summarizer = TextRankSummarizer(damping_factor=0.85, max_iterations=100, tolerance=1e-6)
    # Simple 3-node connected graph
    matrix = [
        [0.0, 0.5, 0.2],
        [0.5, 0.0, 0.8],
        [0.2, 0.8, 0.0],
    ]
    scores, iters, converged, diff = summarizer.calculate_pagerank(matrix)

    assert converged is True
    assert iters > 1
    assert diff < 1e-6
    assert len(scores) == 3
    # PageRank scores must sum to 1.0
    assert sum(scores) == pytest.approx(1.0, rel=1e-5)
    # Node 1 is most central (connected strongly to both 0 and 2)
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_dangling_node_handling():
    """Test 10: Node with zero outgoing edges (dangling) does not cause division by zero."""
    summarizer = TextRankSummarizer()
    # Node 2 has 0 similarity to everything
    matrix = [
        [0.0, 0.6, 0.0],
        [0.6, 0.0, 0.0],
        [0.0, 0.0, 0.0],  # Dangling node
    ]
    scores, iters, converged, diff = summarizer.calculate_pagerank(matrix)

    assert converged is True
    assert len(scores) == 3
    assert sum(scores) == pytest.approx(1.0, rel=1e-5)
    assert all(s > 0.0 for s in scores)


def test_deterministic_ranking():
    """Test 11: Ranking uses PageRank score descending, and lower sentence index on ties."""
    summarizer = TextRankSummarizer()
    sentences = [
        {"sentence_index": 3, "original_text": "Sentence three.", "content_tokens": ["word_a"]},
        {"sentence_index": 1, "original_text": "Sentence one.", "content_tokens": ["word_b"]},
        {"sentence_index": 2, "original_text": "Sentence two.", "content_tokens": ["word_c"]},
    ]
    # In an all-disconnected graph, all nodes receive uniform score 1/3
    ranked, _, _ = summarizer.rank_sentences(sentences)

    assert len(ranked) == 3
    assert ranked[0]["score"] == pytest.approx(ranked[1]["score"], rel=1e-5)
    # Lower sentence_index wins ties: 1, then 2, then 3
    assert ranked[0]["sentence_index"] == 1
    assert ranked[1]["sentence_index"] == 2
    assert ranked[2]["sentence_index"] == 3


def test_top_k_selection():
    """Test 12: Top-K selection respects requested count."""
    summarizer = TextRankSummarizer()
    sents = [
        {"sentence_index": i, "original_text": f"Text {i}.", "content_tokens": [f"tok_{i}"]}
        for i in range(8)
    ]
    record = {"id": "top_k_test", "processed_sentences": sents}

    res_3 = summarizer.summarize_processed_record(record, num_sentences=3)
    assert res_3["num_sentences_selected"] == 3

    res_6 = summarizer.summarize_processed_record(record, num_sentences=6)
    assert res_6["num_sentences_selected"] == 6

    # Clamped to available sentences
    res_15 = summarizer.summarize_processed_record(record, num_sentences=15)
    assert res_15["num_sentences_selected"] == 8


def test_original_order_restoration():
    """Test 13: Final selected sentences are sorted back into chronological article order."""
    summarizer = TextRankSummarizer()
    ranked_candidates = [
        {"sentence_index": 4, "score": 0.45, "original_text": "Fourth."},
        {"sentence_index": 1, "score": 0.35, "original_text": "First."},
        {"sentence_index": 3, "score": 0.25, "original_text": "Third."},
    ]

    selected = summarizer.select_top_sentences(ranked_candidates, num_sentences=3)
    indices = [s["sentence_index"] for s in selected]

    assert indices == [1, 3, 4]


def test_original_sentence_preservation():
    """Test 14: Verbatim source text is preserved in summary without modifications."""
    summarizer = TextRankSummarizer()
    verbatim = "Scientists from NASA and ESA confirmed the exoplanet observation on Friday!"
    sentences = [
        {"sentence_index": 0, "original_text": verbatim, "content_tokens": ["scientists", "nasa", "esa", "exoplanet"]},
        {"sentence_index": 1, "original_text": "Other sentence.", "content_tokens": ["other", "sentence"]},
    ]
    record = {"id": "preservation_test", "processed_sentences": sentences}

    result = summarizer.summarize_processed_record(record, num_sentences=1)
    selected = result["selected_sentences"][0]

    assert selected["original_text"] == verbatim
    assert selected["text"] == verbatim
    assert verbatim in result["summary"]


def test_deterministic_output():
    """Test 15: Running TextRank twice on identical input produces identical outputs."""
    summarizer = TextRankSummarizer()
    record = {
        "id": "det_test",
        "processed_sentences": [
            {"sentence_index": 0, "original_text": "First part.", "content_tokens": ["solar", "energy"]},
            {"sentence_index": 1, "original_text": "Second part.", "content_tokens": ["solar", "power", "grid"]},
            {"sentence_index": 2, "original_text": "Third part.", "content_tokens": ["wind", "turbines"]},
        ],
    }

    r1 = summarizer.summarize_processed_record(record, num_sentences=2)
    r2 = summarizer.summarize_processed_record(record, num_sentences=2)

    assert r1["summary"] == r2["summary"]
    assert r1["selected_sentences"] == r2["selected_sentences"]
    assert r1["metadata"] == r2["metadata"]


def test_single_article_raw_text_summarization():
    """Test 16: summarize_text() handles raw string input via pipeline."""
    summarizer = TextRankSummarizer()
    raw = (
        "Electric vehicles are gaining significant market share worldwide. "
        "Automakers are investing billions into advanced battery gigafactories. "
        "Meanwhile, renewable energy infrastructure must scale to meet charging demands."
    )

    result = summarizer.summarize_text(raw, num_sentences=2)

    assert result["num_sentences_selected"] == 2
    assert len(result["summary"]) > 0
    assert len(result["selected_sentences"]) == 2
    assert result["metadata"]["converged"] is True


def test_real_cnn_dailymail_record_summarization(real_cnn_processed_record):
    """Test 17: Summarize a real CNN/DailyMail record from Phase 2."""
    summarizer = TextRankSummarizer()
    res = summarizer.summarize_processed_record(real_cnn_processed_record, num_sentences=3)

    assert res["article_id"] == real_cnn_processed_record["id"]
    assert res["method"] == "textrank"
    assert res["num_sentences_selected"] == 3
    assert len(res["selected_sentences"]) == 3
    assert len(res["summary"]) > 0
    assert res["metadata"]["converged"] is True
    assert res["metadata"]["iterations"] >= 1


def test_tiny_hand_calculated_graph(monkeypatch):
    """
    Test 18: Tiny hand-calculated test from Section 21:
    Sentence 0: "cats chase mice"
    Sentence 1: "cats hunt mice"
    Sentence 2: "football teams won matches"
    Sentences 0 & 1 share words and must have higher similarity than with Sentence 2.
    """
    summarizer = TextRankSummarizer()
    sentences = [
        ["cats", "chase", "mice"],
        ["cats", "hunt", "mice"],
        ["football", "teams", "won", "matches"],
    ]
    vectors, df, idf = summarizer.build_sentence_vectors(sentences)
    sim_matrix = summarizer.build_similarity_matrix(vectors)

    # Similarity between 0 and 1 must be strictly greater than between 0 and 2, and 1 and 2
    sim_01 = sim_matrix[0][1]
    sim_02 = sim_matrix[0][2]
    sim_12 = sim_matrix[1][2]

    assert sim_01 > 0.0
    assert sim_02 == 0.0  # Zero overlap with football sentence
    assert sim_12 == 0.0  # Zero overlap with football sentence
    assert sim_01 > sim_02
    assert sim_01 > sim_12


def test_edge_cases():
    """Test 19: Edge cases (empty article, single sentence, empty content words)."""
    summarizer = TextRankSummarizer()

    # Empty record
    empty_res = summarizer.summarize_processed_record({"id": "e", "processed_sentences": []})
    assert empty_res["num_sentences_selected"] == 0
    assert empty_res["summary"] == ""

    # Single sentence
    single_res = summarizer.summarize_processed_record({
        "id": "s",
        "processed_sentences": [
            {"sentence_index": 0, "original_text": "Only one sentence.", "content_tokens": ["one", "sentence"]}
        ]
    })
    assert single_res["num_sentences_selected"] == 1
    assert single_res["summary"] == "Only one sentence."
    assert single_res["selected_sentences"][0]["textrank_score"] == 1.0

    # Sentence with zero content tokens
    zero_tokens_res = summarizer.summarize_processed_record({
        "id": "z",
        "processed_sentences": [
            {"sentence_index": 0, "original_text": "However, therefore.", "content_tokens": []},
            {"sentence_index": 1, "original_text": "Valid statement.", "content_tokens": ["valid", "statement"]},
        ]
    })
    assert zero_tokens_res["num_sentences_selected"] == 2
