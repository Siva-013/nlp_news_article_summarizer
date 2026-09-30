# Phase 6 Analysis: Quantitative Evaluation and Algorithm Comparison

## 1. Objective

Phase 6 implements a comprehensive, first-principles quantitative benchmarking suite to evaluate and compare the three classical extractive summarization algorithms implemented across Phases 3, 4, and 5:

1. **Frequency-Based Extractive Summarization** (Phase 3)
2. **TF-IDF-Based Extractive Summarization** (Phase 4)
3. **TextRank Graph-Based Extractive Summarization** (Phase 5)

The primary goal is to answer objectively and empirically:

> **Which classical extractive summarization approach performs best on the CNN/DailyMail benchmark dataset, and what structural trade-offs govern their performance?**

All metrics, n-gram extractors, and dynamic programming LCS algorithms are implemented from scratch without black-box third-party libraries, ensuring complete academic explainability and reproducibility for thesis and viva presentation.

---

## 2. Dataset and Corpus Details

The evaluation is conducted on the deterministic test partition established in Phase 1:

- **Dataset**: CNN/DailyMail (`abisee/cnn_dailymail`, v3.0.0)
- **Split**: Test set
- **Sample Size**: 1,000 news articles
- **Random Seed**: 42
- **Preprocessed Source**: `dataset/processed/cnn_dailymail_test_1000_processed.jsonl`
- **Reference Targets**: Human-written multi-sentence bulleted highlights extracted from the dataset.

The three summarization systems were executed under identical baseline constraints:
- **Target Summary Length**: Exactly 3 sentences extracted verbatim from the source text.
- **Order Preservation**: Extracted sentences were restored to their original chronological narrative order.

---

## 3. Evaluation Methodology

### 3.1 Dual-Representation Evaluation Pipeline
Following the project's core architecture:
1. **Verbatim Preservation**: The generated summaries evaluated are the unmodified, exact sentences selected by each algorithm.
2. **Standardized Tokenization for Matching**:
   - Lowercasing and Unicode normalization.
   - Punctuation removal while preserving intra-word hyphens and contractions.
   - Matching is performed on alphanumeric word tokens against the tokenized reference highlights.

### 3.2 Evaluation Metrics Overview
We report Precision, Recall, and F1-score across three standard metrics:
- **ROUGE-1**: Unigram overlap (vocabulary coverage and informational overlap).
- **ROUGE-2**: Bigram overlap (local phrasal coherence and syntactic structure).
- **ROUGE-L**: Longest Common Subsequence (sentence-level word order preservation).

---

## 4. Mathematical Formulation

### 4.1 ROUGE-N (Unigram & Bigram Matching)
For $N \in \{1, 2\}$, let $\text{Cand}$ be the candidate summary and $\text{Ref}$ be the reference highlight:

$$\text{Recall}_{\text{ROUGE-}N} = \frac{\sum_{\text{gram}_n \in \text{Ref}} \min(\text{count}_{\text{cand}}(\text{gram}_n), \text{count}_{\text{ref}}(\text{gram}_n))}{\sum_{\text{gram}_n \in \text{Ref}} \text{count}_{\text{ref}}(\text{gram}_n)}$$

$$\text{Precision}_{\text{ROUGE-}N} = \frac{\sum_{\text{gram}_n \in \text{Cand}} \min(\text{count}_{\text{cand}}(\text{gram}_n), \text{count}_{\text{ref}}(\text{gram}_n))}{\sum_{\text{gram}_n \in \text{Cand}} \text{count}_{\text{cand}}(\text{gram}_n)}$$

$$\text{F1}_{\text{ROUGE-}N} = \frac{2 \times \text{Precision}_{\text{ROUGE-}N} \times \text{Recall}_{\text{ROUGE-}N}}{\text{Precision}_{\text{ROUGE-}N} + \text{Recall}_{\text{ROUGE-}N}}$$

*Note: The clipping term $\min(\text{count}_{\text{cand}}, \text{count}_{\text{ref}})$ guarantees that repeated words in candidate summaries cannot artificially inflate overlap.*

### 4.2 ROUGE-L (Longest Common Subsequence)
Let $\text{LCS}(\text{Cand}, \text{Ref})$ be the length of the longest subsequence of tokens common to both candidate and reference (preserving in-sequence order without requiring contiguous positions):

$$\text{Recall}_{\text{ROUGE-L}} = \frac{\text{LCS}(\text{Cand}, \text{Ref})}{|\text{Ref}|}$$

$$\text{Precision}_{\text{ROUGE-L}} = \frac{\text{LCS}(\text{Cand}, \text{Ref})}{|\text{Cand}|}$$

$$\text{F1}_{\text{ROUGE-L}} = \frac{2 \times \text{Precision}_{\text{ROUGE-L}} \times \text{Recall}_{\text{ROUGE-L}}}{\text{Precision}_{\text{ROUGE-L}} + \text{Recall}_{\text{ROUGE-L}}}$$

$\text{LCS}$ is computed via dynamic programming in $\mathcal{O}(|\text{Cand}| \cdot |\text{Ref}|)$ time and $\mathcal{O}(|\text{Ref}|)$ space.

---

## 5. Overall Quantitative Results

Macro-averages across all 1,000 articles on the CNN/DailyMail test set:

| Algorithm | ROUGE-1 Precision | ROUGE-1 Recall | ROUGE-1 F1 | ROUGE-2 Precision | ROUGE-2 Recall | ROUGE-2 F1 | ROUGE-L Precision | ROUGE-L Recall | ROUGE-L F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Frequency** | **0.2134** | 0.2886 | 0.2306 | 0.0573 | 0.0846 | 0.0644 | **0.1508** | 0.2024 | 0.1619 |
| **TF-IDF** | 0.1524 | 0.1128 | 0.1201 | 0.0238 | 0.0211 | 0.0205 | 0.1161 | 0.0837 | 0.0898 |
| **TextRank** | 0.1933 | **0.4124** | **0.2536** | **0.0631** | **0.1440** | **0.0850** | 0.1329 | **0.2859** | **0.1746** |

### Primary F1 Summary
- **ROUGE-1 F1 Winner**: **TextRank (0.2536)** > Frequency (0.2306) > TF-IDF (0.1201)
- **ROUGE-2 F1 Winner**: **TextRank (0.0850)** > Frequency (0.0644) > TF-IDF (0.0205)
- **ROUGE-L F1 Winner**: **TextRank (0.1746)** > Frequency (0.1619) > TF-IDF (0.0898)

---

## 6. Detailed Statistical Analysis

To prevent single-metric distortion, we evaluate full statistical distributions (Mean, Median, Standard Deviation, Min, Max) across all 1,000 articles:

### 6.1 ROUGE-1 F1 Distribution
| Algorithm | Mean | Median | Std Dev | Min | Max |
|---|:---:|:---:|:---:|:---:|:---:|
| **Frequency** | 0.2306 | 0.2180 | 0.1060 | 0.0000 | 0.6667 |
| **TF-IDF** | 0.1201 | 0.1034 | 0.0960 | 0.0000 | 0.6909 |
| **TextRank** | **0.2536** | **0.2424** | 0.1029 | 0.0000 | **0.7009** |

### 6.2 ROUGE-2 F1 Distribution
| Algorithm | Mean | Median | Std Dev | Min | Max |
|---|:---:|:---:|:---:|:---:|:---:|
| **Frequency** | 0.0644 | 0.0370 | 0.0809 | 0.0000 | 0.5455 |
| **TF-IDF** | 0.0205 | 0.0000 | 0.0561 | 0.0000 | **0.6038** |
| **TextRank** | **0.0850** | **0.0583** | 0.0863 | 0.0000 | 0.4882 |

### 6.3 ROUGE-L F1 Distribution
| Algorithm | Mean | Median | Std Dev | Min | Max |
|---|:---:|:---:|:---:|:---:|:---:|
| **Frequency** | 0.1619 | 0.1458 | 0.0817 | 0.0000 | **0.6667** |
| **TF-IDF** | 0.0898 | 0.0764 | 0.0720 | 0.0000 | 0.5238 |
| **TextRank** | **0.1746** | **0.1567** | 0.0830 | 0.0000 | 0.5983 |

---

## 7. Per-Article Winner Analysis

Evaluating how often each algorithm achieves the strictly highest F1 score on individual articles (1,000 articles total):

| Metric | Frequency Wins | TF-IDF Wins | TextRank Wins | Ties |
|---|:---:|:---:|:---:|:---:|
| **ROUGE-1 F1** | 352 (35.2%) | 81 (8.1%) | **507 (50.7%)** | 60 (6.0%) |
| **ROUGE-2 F1** | 283 (28.3%) | 80 (8.0%) | **464 (46.4%)** | 173 (17.3%) |
| **ROUGE-L F1** | 383 (38.3%) | 95 (9.5%) | **463 (46.3%)** | 59 (5.9%) |

TextRank emerges as the top-performing summarizer on over **50.7%** of articles for ROUGE-1, **46.4%** for ROUGE-2, and **46.3%** for ROUGE-L.

---

## 8. Summary Length Analysis and Trade-offs

A critical factor influencing ROUGE recall and precision is the length of extracted summaries. We measured word counts across all 1,000 articles:

| Source | Mean Word Count | Median | Std Dev | Min | Max |
|---|:---:|:---:|:---:|:---:|:---:|
| **Reference Highlights** | 34.47 | 34.0 | 9.67 | 14.0 | 80.0 |
| **Frequency Summarizer** | 43.51 | 43.0 | 17.35 | 3.0 | 100.0 |
| **TF-IDF Summarizer** | 21.65 | 20.0 | 10.92 | 3.0 | 79.0 |
| **TextRank Summarizer** | 69.72 | 69.0 | 22.72 | 3.0 | 191.0 |

### Analysis of the Precision-Recall Trade-off:
1. **TextRank (High Recall Driver)**:
   - By extracting substantive sentences with multiple similarity connections (mean 69.72 words vs. 34.47 reference words), TextRank captures significantly more reference concepts, yielding a commanding **0.4124 Recall on ROUGE-1** (compared to 0.2886 for Frequency and 0.1128 for TF-IDF). This recall advantage elevates its harmonic mean (F1).
2. **Frequency (High Precision Balance)**:
   - Frequency summaries average 43.51 words, which is closer to the reference summary length (34.47 words). Consequently, Frequency achieves the **highest Precision on ROUGE-1 (0.2134)** and **ROUGE-L (0.1508)**, avoiding unnecessary sentence bulk.
3. **TF-IDF (Length Deficit & Rare Token Penalty)**:
   - Intra-article TF-IDF length-normalization heavily penalized longer compound sentences, leading to short extractions (mean 21.65 words). Many extracted sentences were brief or contained rare dateline tokens (`(CNN)`), causing severe recall loss (0.1128).

---

## 9. Runtime and Computational Efficiency

Measured across the 1,000 article test set on the local evaluation workstation:

| Algorithm | Processing Time (1,000 articles) | Throughput | Complexity per Article |
|---|:---:|:---:|:---:|
| **Frequency Summarizer** | ~1.42 seconds | ~704 articles/sec | $\mathcal{O}(M)$ linear in token count |
| **TF-IDF Summarizer** | ~1.75 seconds | ~571 articles/sec | $\mathcal{O}(M)$ linear in token count |
| **TextRank Summarizer** | ~7.86 seconds | ~127 articles/sec | $\mathcal{O}(N^2)$ pairwise graph + PageRank |

*Frequency is ~5.5× faster than TextRank due to avoiding pairwise graph operations, but TextRank achieves higher ROUGE scores while remaining highly practical (127 articles/sec).*

---

## 10. Strengths and Weaknesses of Evaluated Algorithms

### Frequency-Based Summarizer
- **Strengths**: Ultra-fast ($\mathcal{O}(M)$), best precision (0.2134 on R-1), concise length closely matching human abstracts.
- **Weaknesses**: Prone to extracting repetitive sentences that cluster around high-frequency keywords (*"said"*, *"police"*, *"court"*).

### TF-IDF-Based Summarizer
- **Strengths**: Selects distinct vocabulary terms, avoids pure keyword repetition.
- **Weaknesses**: Lowest ROUGE scores across all metrics. Vulnerable to isolated datelines and rare words that possess high IDF but low topical salience.

### TextRank Graph Summarizer
- **Strengths**: Highest ROUGE-1, ROUGE-2, and ROUGE-L F1 scores. Global graph recommendation naturally suppresses isolated tokens and elevates topically cohesive sentences.
- **Weaknesses**: $\mathcal{O}(N^2)$ graph construction complexity; produces longer summaries that reduce precision.

---

## 11. Visualizations

Generated publication-quality charts are stored in `docs/figures/`:
1. `docs/figures/rouge_comparison.png`: Bar chart comparing ROUGE-1, ROUGE-2, and ROUGE-L F1 scores across all three algorithms.
2. `docs/figures/rouge_precision_recall.png`: Scatter plot mapping the Precision vs. Recall operating points of each model.
3. `docs/figures/summary_length_comparison.png`: Word length distributions showing mean and standard deviation compared to ground-truth highlights.

---

## 12. Academic Limitations

1. **Lexical ROUGE vs. Semantic Synthesis**:
   ROUGE relies on strict n-gram and subsequence overlap. Classical extractive summaries that express the exact same meaning as the reference highlights using synonymous vocabulary receive zero ROUGE credit.
2. **Abstractive Human References**:
   CNN/DailyMail reference highlights are written by human journalists as compact, abstractive bullet points. Purely extractive sentence selection is structurally capped in maximum possible ROUGE overlap.
3. **Fixed-Sentence Constraint**:
   All models operated under a fixed 3-sentence constraint. Dynamic length budgets (e.g., word count capping at 50 words) could alter the precision-recall balance.

---

## 13. Reproducibility Instructions

To replicate the complete Phase 6 evaluation:

```powershell
python -m evaluation.evaluator
```

- **Metrics Output**: `dataset/processed/evaluation_results.json`
- **Article-Level Output**: `dataset/processed/article_level_evaluation.jsonl`
- **Test Suite**:
  ```powershell
  python -m pytest tests/test_evaluation.py -v
  ```
