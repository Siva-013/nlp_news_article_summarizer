"""
Tests for Phase 6: Quantitative Evaluation and Algorithm Comparison

Validates:
1. ROUGE-1 perfect match
2. ROUGE-1 partial match
3. ROUGE-1 no overlap
4. ROUGE-2 perfect match
5. ROUGE-2 no overlap
6. ROUGE-L perfect match
7. ROUGE-L partial match
8. Empty candidate edge case
9. Empty reference edge case
10. Tokenization for ROUGE (punctuation removal, lowercase, contractions)
11. N-gram generation (unigrams, bigrams, out-of-bounds length)
12. Longest Common Subsequence (LCS) calculation
13. Article ID matching across summary files
14. Per-article evaluation output structure
15. Corpus-level macro aggregation (mean, median, std, min, max)
16. Deterministic output reproducibility
17. Real CNN/DailyMail record evaluation
18. Repeated token overlap clipping
"""

import json
from pathlib import Path
import pytest

from evaluation.comparison import (
    calculate_descriptive_stats,
    compute_per_article_winners,
)
from evaluation.evaluator import Evaluator
from evaluation.rouge import (
    calculate_lcs_length,
    calculate_rouge_l,
    calculate_rouge_n,
    evaluate_summary_pair,
    get_ngrams,
    tokenize_for_rouge,
)


# ==============================================================================
# Unit Tests: Tokenization and N-Grams
# ==============================================================================

def test_tokenization_for_rouge():
    """Test 10: Tokenization handles punctuation, casing, and contractions."""
    raw = "The U.S. government's new climate-policy was announced yesterday!"
    tokens = tokenize_for_rouge(raw)
    
    assert "the" in tokens
    assert "u.s" not in tokens or "u" in tokens or "government's" in tokens
    # Verify all lowercase
    assert all(t == t.lower() for t in tokens)
    # Standalone exclamation and period stripped
    assert "!" not in tokens
    assert "." not in tokens
    assert tokenize_for_rouge("") == []


def test_ngram_generation():
    """Test 11: N-gram frequency generation."""
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    
    # Unigrams (n=1)
    unigrams = get_ngrams(tokens, 1)
    assert unigrams[("the",)] == 2
    assert unigrams[("cat",)] == 1
    assert sum(unigrams.values()) == 6
    
    # Bigrams (n=2)
    bigrams = get_ngrams(tokens, 2)
    assert bigrams[("the", "cat")] == 1
    assert bigrams[("sat", "on")] == 1
    assert sum(bigrams.values()) == 5
    
    # Out of bounds n
    assert get_ngrams(tokens, 10) == {}
    assert get_ngrams([], 1) == {}


def test_lcs_calculation():
    """Test 12: Longest Common Subsequence (LCS) dynamic programming."""
    cand = ["the", "brown", "fox", "jumps", "high"]
    ref = ["the", "quick", "brown", "fox", "jumps"]
    
    # Common subsequence: "the", "brown", "fox", "jumps" (len 4)
    assert calculate_lcs_length(cand, ref) == 4
    
    # No common subsequence
    assert calculate_lcs_length(["a", "b"], ["c", "d"]) == 0
    
    # Empty lists
    assert calculate_lcs_length([], ref) == 0
    assert calculate_lcs_length(cand, []) == 0


# ==============================================================================
# Unit Tests: ROUGE Metrics
# ==============================================================================

def test_rouge_1_perfect_match():
    """Test 1: ROUGE-1 perfect identical match yields 1.0 P, R, and F1."""
    text = "The president signed the international trade agreement."
    res = calculate_rouge_n(text, text, n=1)
    
    assert res["precision"] == pytest.approx(1.0, rel=1e-5)
    assert res["recall"] == pytest.approx(1.0, rel=1e-5)
    assert res["f1"] == pytest.approx(1.0, rel=1e-5)


def test_rouge_1_partial_match():
    """Test 2: ROUGE-1 partial overlap matches exact mathematical expectation."""
    cand = "the cat sat on the mat"       # 6 unigrams: the:2, cat:1, sat:1, on:1, mat:1
    ref = "the dog sat on the rug"        # 6 unigrams: the:2, dog:1, sat:1, on:1, rug:1
    
    # Overlapping unigrams: the (min 2,2 = 2), sat (1), on (1) -> total 4 overlap
    # Precision = 4 / 6 = 2/3
    # Recall = 4 / 6 = 2/3
    # F1 = 2/3 ≈ 0.666667
    res = calculate_rouge_n(cand, ref, n=1)
    
    assert res["precision"] == pytest.approx(4 / 6, rel=1e-4)
    assert res["recall"] == pytest.approx(4 / 6, rel=1e-4)
    assert res["f1"] == pytest.approx(4 / 6, rel=1e-4)


def test_rouge_1_no_overlap():
    """Test 3: ROUGE-1 completely disjoint texts yield 0.0."""
    cand = "apples bananas cherries"
    ref = "dogs cats birds"
    res = calculate_rouge_n(cand, ref, n=1)
    
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0


def test_rouge_2_perfect_match():
    """Test 4: ROUGE-2 perfect identical match yields 1.0."""
    text = "Artificial intelligence will transform healthcare and education systems."
    res = calculate_rouge_n(text, text, n=2)
    
    assert res["precision"] == pytest.approx(1.0, rel=1e-5)
    assert res["recall"] == pytest.approx(1.0, rel=1e-5)
    assert res["f1"] == pytest.approx(1.0, rel=1e-5)


def test_rouge_2_no_overlap():
    """Test 5: ROUGE-2 with identical unigrams but zero bigram overlap yields 0.0."""
    cand = "apple banana cherry date"
    ref = "banana apple date cherry"
    res = calculate_rouge_2 = calculate_rouge_n(cand, ref, n=2)
    
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0


def test_rouge_l_perfect_match():
    """Test 6: ROUGE-L identical match yields 1.0."""
    text = "The solar energy company announced record quarterly revenues."
    res = calculate_rouge_l(text, text)
    
    assert res["precision"] == pytest.approx(1.0, rel=1e-5)
    assert res["recall"] == pytest.approx(1.0, rel=1e-5)
    assert res["f1"] == pytest.approx(1.0, rel=1e-5)


def test_rouge_l_partial_match():
    """Test 7: ROUGE-L partial match with subsequence alignment."""
    cand = "police arrested three suspects on Tuesday"    # 6 tokens
    ref = "police arrested suspects on Tuesday morning"    # 6 tokens
    # LCS: ["police", "arrested", "suspects", "on", "Tuesday"] (len 5)
    res = calculate_rouge_l(cand, ref)
    
    assert res["precision"] == pytest.approx(5 / 6, rel=1e-4)
    assert res["recall"] == pytest.approx(5 / 6, rel=1e-4)
    assert res["f1"] == pytest.approx(5 / 6, rel=1e-4)


def test_empty_candidate_and_reference():
    """Tests 8 & 9: Empty strings safely return 0.0 without division by zero."""
    valid_text = "Some valid summary sentence."
    
    # Test 8: Empty candidate
    res_c_empty = calculate_rouge_n("", valid_text, n=1)
    assert res_c_empty == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert calculate_rouge_l("", valid_text) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Test 9: Empty reference
    res_r_empty = calculate_rouge_n(valid_text, "", n=1)
    assert res_r_empty == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert calculate_rouge_l(valid_text, "") == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_repeated_token_clipping():
    """Test 18: Candidate with repeated words cannot inflate overlap beyond reference count."""
    cand = "cat cat cat cat cat"     # 5 occurrences
    ref = "the cat is black"          # 1 occurrence
    # Overlap must be clipped to 1
    res = calculate_rouge_n(cand, ref, n=1)
    
    # cand has 5 tokens, ref has 4 tokens
    # overlap = 1
    # P = 1/5 = 0.2, R = 1/4 = 0.25
    assert res["precision"] == pytest.approx(0.2, rel=1e-4)
    assert res["recall"] == pytest.approx(0.25, rel=1e-4)


# ==============================================================================
# Integration Tests: Evaluator Pipeline & Statistical Aggregation
# ==============================================================================

def test_article_id_matching(tmp_path):
    """Test 13: Evaluator correctly matches records across multiple JSONL files by ID."""
    ref_file = tmp_path / "ref.jsonl"
    freq_file = tmp_path / "freq.jsonl"
    tfidf_file = tmp_path / "tfidf.jsonl"
    textrank_file = tmp_path / "textrank.jsonl"
    
    ref_file.write_text('{"id": "doc1", "highlights": "Ref one."}\n{"id": "doc2", "highlights": "Ref two."}\n', encoding="utf-8")
    freq_file.write_text('{"id": "doc1", "summary": "Freq one."}\n{"id": "doc2", "summary": "Freq two."}\n', encoding="utf-8")
    tfidf_file.write_text('{"id": "doc1", "summary": "TFIDF one."}\n{"id": "doc2", "summary": "TFIDF two."}\n', encoding="utf-8")
    textrank_file.write_text('{"id": "doc1", "summary": "TR one."}\n{"id": "doc2", "summary": "TR two."}\n', encoding="utf-8")
    
    evaluator = Evaluator(
        dataset_path=str(ref_file),
        frequency_path=str(freq_file),
        tfidf_path=str(tfidf_file),
        textrank_path=str(textrank_file),
    )
    
    out_json = tmp_path / "res.json"
    out_jsonl = tmp_path / "articles.jsonl"
    plots_dir = tmp_path / "plots"
    
    results = evaluator.run_evaluation(
        output_results_path=str(out_json),
        article_results_path=str(out_jsonl),
        plot_dir=str(plots_dir),
    )
    
    assert results["metadata"]["num_articles_evaluated"] == 2
    assert out_json.exists()
    assert out_jsonl.exists()


def test_per_article_evaluation_structure():
    """Test 14: evaluate_summary_pair returns correct dictionary structure."""
    cand = "A quick summary of the report."
    ref = "The full report was summarized."
    scores = evaluate_summary_pair(cand, ref)
    
    for metric in ("rouge1", "rouge2", "rougeL"):
        assert metric in scores
        for sub_metric in ("precision", "recall", "f1"):
            assert sub_metric in scores[metric]
            assert 0.0 <= scores[metric][sub_metric] <= 1.0


def test_corpus_level_aggregation():
    """Test 15: Macro-averages and descriptive statistics computation."""
    data = [0.1, 0.2, 0.3, 0.4, 0.5]
    stats = calculate_descriptive_stats(data)
    
    assert stats["mean"] == pytest.approx(0.3, rel=1e-5)
    assert stats["median"] == pytest.approx(0.3, rel=1e-5)
    assert stats["min"] == 0.1
    assert stats["max"] == 0.5
    assert stats["std"] > 0.0


def test_deterministic_output():
    """Test 16: Multiple evaluation runs on identical input produce identical scores."""
    cand = "Global summit discusses climate change commitments and renewable targets."
    ref = "World leaders met at climate summit to set renewable targets."
    
    eval1 = evaluate_summary_pair(cand, ref)
    eval2 = evaluate_summary_pair(cand, ref)
    
    assert eval1 == eval2


def test_real_cnn_dailymail_record_evaluation():
    """Test 17: Evaluates on a real CNN/DailyMail processed record pair."""
    processed_file = Path("dataset/processed/cnn_dailymail_test_1000_processed.jsonl")
    freq_file = Path("dataset/processed/frequency_summaries_1000.jsonl")
    
    if not processed_file.exists() or not freq_file.exists():
        pytest.skip("Processed dataset files not found on disk.")
        
    with open(processed_file, "r", encoding="utf-8") as pf, open(freq_file, "r", encoding="utf-8") as ff:
        prec = json.loads(pf.readline())
        frec = json.loads(ff.readline())
        
    scores = evaluate_summary_pair(frec["summary"], prec["highlights"])
    
    assert scores["rouge1"]["f1"] > 0.0
    assert scores["rougeL"]["f1"] > 0.0
    assert 0.0 <= scores["rouge2"]["f1"] <= 1.0
