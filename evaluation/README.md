# Evaluation Module

Part of **Phase 6: Quantitative Evaluation and Algorithm Comparison** in the *NLP-Based News Article Summarization and Vocabulary Learning System*.

---

## 📌 Overview

The evaluation module provides a clean, rigorous, and explainable benchmarking suite to compare classical extractive summarization methods:
1. **Frequency-based Extractive Summarizer** (Phase 3)
2. **TF-IDF Extractive Summarizer** (Phase 4)
3. **TextRank Extractive Summarizer** (Phase 5)

against human ground-truth reference highlights from the **CNN/DailyMail (v3.0.0)** dataset.

---

## 🔬 Mathematical Formulations

All ROUGE metrics are implemented from first principles without external black-box evaluation dependencies.

### 1. ROUGE-N (Unigram / Bigram Overlap)
For $N \in \{1, 2\}$:

$$\text{Recall}_{\text{ROUGE-}N} = \frac{\sum_{\text{gram}_n \in \text{Ref}} \min(\text{count}_{\text{cand}}(\text{gram}_n), \text{count}_{\text{ref}}(\text{gram}_n))}{\sum_{\text{gram}_n \in \text{Ref}} \text{count}_{\text{ref}}(\text{gram}_n)}$$

$$\text{Precision}_{\text{ROUGE-}N} = \frac{\sum_{\text{gram}_n \in \text{Cand}} \min(\text{count}_{\text{cand}}(\text{gram}_n), \text{count}_{\text{ref}}(\text{gram}_n))}{\sum_{\text{gram}_n \in \text{Cand}} \text{count}_{\text{cand}}(\text{gram}_n)}$$

$$\text{F1}_{\text{ROUGE-}N} = \frac{2 \times \text{Precision}_{\text{ROUGE-}N} \times \text{Recall}_{\text{ROUGE-}N}}{\text{Precision}_{\text{ROUGE-}N} + \text{Recall}_{\text{ROUGE-}N}}$$

### 2. ROUGE-L (Longest Common Subsequence)
Let $\text{LCS}(\text{Cand}, \text{Ref})$ be the length of the longest common subsequence of word tokens between the candidate summary and reference highlights:

$$\text{Recall}_{\text{ROUGE-L}} = \frac{\text{LCS}(\text{Cand}, \text{Ref})}{|\text{Ref}|}$$

$$\text{Precision}_{\text{ROUGE-L}} = \frac{\text{LCS}(\text{Cand}, \text{Ref})}{|\text{Cand}|}$$

$$\text{F1}_{\text{ROUGE-L}} = \frac{2 \times \text{Precision}_{\text{ROUGE-L}} \times \text{Recall}_{\text{ROUGE-L}}}{\text{Precision}_{\text{ROUGE-L}} + \text{Recall}_{\text{ROUGE-L}}}$$

---

## 📂 Module Architecture

```text
evaluation/
├── __init__.py           # Package interface & public exports
├── rouge.py              # First-principles ROUGE-1, ROUGE-2, LCS, and ROUGE-L engine
├── evaluator.py          # Central orchestrator: loads JSONL files, computes macro metrics, CLI
├── comparison.py         # Statistical analysis (mean, std, min, max), win-ties, and plotting
└── README.md             # Module documentation
```

---

## 🚀 CLI Usage

Run full evaluation over the 1,000 articles using default paths:

```powershell
python -m evaluation.evaluator
```

Or customize input and output paths:

```powershell
python -m evaluation.evaluator `
  --dataset "dataset/processed/cnn_dailymail_test_1000_processed.jsonl" `
  --frequency "dataset/processed/frequency_summaries_1000.jsonl" `
  --tfidf "dataset/processed/tfidf_summaries_1000.jsonl" `
  --textrank "dataset/processed/textrank_summaries_1000.jsonl" `
  --output "dataset/processed/evaluation_results.json" `
  --article-output "dataset/processed/article_level_evaluation.jsonl" `
  --plot-dir "docs/figures"
```

---

## 📊 Output Artifacts

- **Overall Results JSON**: `dataset/processed/evaluation_results.json`
- **Article-Level JSONL**: `dataset/processed/article_level_evaluation.jsonl`
- **Visual Plots**:
  - `docs/figures/rouge_comparison.png`
  - `docs/figures/rouge_precision_recall.png`
  - `docs/figures/summary_length_comparison.png`
- **Analysis Report**: `docs/evaluation_analysis.md`
