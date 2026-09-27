# CNN/DailyMail Dataset Documentation

## 1. Overview and Purpose
**CNN/DailyMail is a widely used benchmark dataset for news summarization.** Originally constructed for machine reading comprehension and question answering by Hermann et al. (2015), it was later adapted for abstractive and extractive text summarization by See et al. (2017).

In this project, CNN/DailyMail (version 3.0.0) serves as the empirical evaluation benchmark for our classical/traditional Natural Language Processing extractive summarization pipeline (Frequency-based baseline, TF-IDF ranking, and TextRank graph centrality). The human-written highlights provide the objective ground-truth summaries against which the system's generated extractive summaries are quantitatively evaluated using ROUGE metrics in later phases.

---

## 2. Dataset Specification
- **Dataset Name**: `CNN/DailyMail` (Hugging Face Hub identifier: `abisee/cnn_dailymail`)
- **Version**: `3.0.0` (non-anonymized text with original casing, punctuation, and sentence boundaries preserved)
- **Primary Domain**: Journalism, current events, international politics, sports, and culture.

---

## 3. Data Schema and Record Fields
Each record in the dataset consists of a JSON object containing three primary fields:

| Field | Data Type | Description | Role in System |
|:---|:---|:---|:---|
| `id` | `str` | Unique 40-character hexadecimal SHA-1 article hash. | Uniquely identifies articles across evaluation and user history tracking. |
| `article` | `str` | Full body text of the published news article. | Source input text for sentence segmentation, feature extraction, and extractive summary sentence selection. |
| `highlights` | `str` | Human-written bullet-point summary written by journalists. | Ground-truth reference summary for future ROUGE-1, ROUGE-2, and ROUGE-L evaluation. |

---

## 4. Dataset Splits (Full Corpus)
The complete CNN/DailyMail v3.0.0 corpus comprises over 300,000 paired articles and highlights:

- **Train Split**: 287,113 articles (used for statistical corpus vocabulary and IDF background calculation).
- **Validation Split**: 13,368 articles (used for threshold tuning and hyperparameter selection).
- **Test Split**: 11,490 articles (held-out evaluation set for final ROUGE score benchmarking).

---

## 5. Local Development and Sampling Strategy
The complete uncompressed CNN/DailyMail corpus requires several gigabytes of storage and substantial memory to load into memory at once. In production and academic environments, developing directly against monolithic corpora causes unnecessary computational overhead during local engineering and debugging.

To address this:
1. **Manageable Sample Selection**: We acquire reproducible subsets (e.g., $N = 100$ or $N = 1,000$ records from the `test` split) to enable rapid algorithm prototyping, fast unit testing, and immediate feedback loops.
2. **Streaming Acquisition**: Using Hugging Face's streaming mode (`streaming=True`), the downloader fetches records sequentially over HTTP without requiring the download or decompression of the full multi-gigabyte archive.
3. **Data Preservation**: The raw articles and highlights are saved in their unmodified source form in `dataset/raw/` using standard line-delimited JSON (`.jsonl`). Original capitalization, punctuation, and sentence wording are strictly maintained to ensure extractive summaries reflect genuine source text.

---

## 6. Reproducibility
Dataset acquisition and analysis are fully deterministic and reproducible via versioned parameters:

- **Target Split**: `test`
- **Sample Size**: `100` (or `1000`)
- **Random Seed**: `42`
- **Data Format**: Line-delimited JSON (`.jsonl`), UTF-8 encoded.

### Download Command
```powershell
python dataset/download_dataset.py --split test --samples 100 --seed 42
```

### Exploration & Analysis Command
```powershell
python dataset/explore_dataset.py --input dataset/raw/cnn_dailymail_test_100.jsonl
```

### Outputs Generated
- Raw data: `dataset/raw/cnn_dailymail_test_100.jsonl`
- Machine-readable statistics: `dataset/dataset_stats.json`
- Analytical report: `dataset/dataset_analysis.md`

---

## 7. Empirical Findings (100 Sample Benchmark)
From our empirical analysis of the 100-sample test partition:
- **Mean Article Length**: 540.9 words (~26.6 sentences)
- **Mean Reference Highlight Length**: 34.5 words (~2.6 sentences)
- **Mean Compression Ratio**: 0.0935 (9.35% of the original article length)
- **Lexical Diversity**: Mean Type-Token Ratio (TTR) is 0.5518, with an average of 277 unique word types per article.

---

## 8. Academic References
1. **Hermann, K. M., Kocisky, T., Grefenstette, E., Espeholt, L., Kay, W., Suleyman, M., & Blunsom, P. (2015).** Teaching machines to read and comprehend. In *Advances in Neural Information Processing Systems (NeurIPS 2015)* (pp. 1693-1701).
2. **See, A., Liu, P. J., & Manning, C. D. (2017).** Get To The Point: Summarization with Pointer-Generator Networks. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1073-1083). Association for Computational Linguistics.
