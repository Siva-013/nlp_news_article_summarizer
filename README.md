# NLP-Based News Article Summarization and Vocabulary Learning System

An end-to-end academic Natural Language Processing system for extractive news article summarization and vocabulary enrichment, built using foundational NLP algorithms (without external black-box generative models or closed APIs).

---

## 📌 Project Overview

This project implements classical and modern NLP techniques on the **CNN/DailyMail (v3.0.0)** dataset. It focuses on transparent, explainable summarization pipelines coupled with rigorous linguistic preprocessing, statistical evaluation, and reproducible benchmarks.

### Key Milestones Completed

- **Phase 1: Dataset Setup & Exploration**
  - Streaming acquisition of CNN/DailyMail test splits.
  - Comprehensive statistical profiling (token counts, compression ratios, vocabulary diversity / TTR).
  - Deterministic sampling (1,000 records, seed: 42).
- **Phase 2: Data Preprocessing Pipeline**
  - HTML stripping and artifact removal.
  - Unicode normalization (NFKC) and whitespace standardization.
  - Sentence segmentation via spaCy with custom hardened infix tokenization rules.
  - Morphological lemmatization, stop-word filtering, POS tagging, and Named Entity Recognition (NER) with exact relative character offsets.
  - **Dual Representation**: Preserves original sentence text for extractive output while maintaining normalized token streams for mathematical scoring.
- **Phase 3: Frequency-Based Extractive Summarization**
  - Luhn-inspired word-frequency extraction baseline.
  - Normalized word frequencies: $\text{NF}(w) = \frac{\text{freq}(w)}{\max_{w'} \text{freq}(w')}$.
  - Length-normalized sentence scoring: $\text{Score}(S) = \frac{\sum_{w \in C(S)} \text{NF}(w)}{|C(S)|}$.
  - Deterministic tie-breaking and chronological ordering of extracted sentences.
  - High-performance batch processing and standalone CLI.

---

## 📂 Project Structure

```text
D:\nlp project/
├── dataset/
│   ├── download_dataset.py           # Dataset acquisition script
│   ├── explore_dataset.py            # Statistical exploration script
│   ├── dataset_stats_1000.json       # Dataset metrics (1,000 articles)
│   ├── raw/
│   │   ├── cnn_dailymail_test_100.jsonl
│   │   └── cnn_dailymail_test_1000.jsonl
│   └── processed/
│       ├── preprocessing_stats.json  # Phase 2 pipeline metrics
│       └── frequency_summaries_1000.jsonl # Phase 3 baseline summaries
├── preprocessing/
│   ├── cleaner.py                    # HTML/text cleaning
│   ├── normalizer.py                 # Unicode & whitespace normalization
│   ├── sentence_splitter.py          # Sentence segmentation
│   ├── tokenizer.py                  # Hardened spaCy tokenizer
│   ├── stopwords.py                  # Stop-word filtering
│   ├── lemmatizer.py                 # Morphological lemmatization
│   ├── linguistic_features.py        # POS tagging and NER
│   └── pipeline.py                   # Master preprocessing pipeline
├── summarization/
│   ├── __init__.py                   # Package initialization
│   └── frequency.py                  # FrequencySummarizer class and CLI
├── tests/
│   ├── test_dataset.py               # Phase 1 unit tests (6 tests)
│   ├── test_preprocessing.py         # Phase 2 unit & regression tests (17 tests)
│   └── test_frequency_summarization.py # Phase 3 unit & integration tests (10 tests)
├── docs/
│   ├── preprocessing_analysis.md     # Phase 2 statistical analysis
│   └── frequency_baseline_analysis.md# Phase 3 baseline report & 5 CNN examples
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment configuration template
└── README.md                         # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment

- Python 3.10+ (Tested on Python 3.13)
- Windows / Linux / macOS

```bash
# Clone the repository
git clone https://github.com/Siva-013/nlp_news_article_summarizer.git
cd nlp_news_article_summarizer

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies and download spaCy model
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Running Preprocessing

```bash
python -m preprocessing.pipeline --input dataset/raw/cnn_dailymail_test_1000.jsonl --output dataset/processed/cnn_dailymail_test_1000_processed.jsonl
```

### 3. Running Frequency Summarizer CLI

```bash
# Summarize direct text:
python -m summarization.frequency --article "Artificial intelligence is changing the world..." --sentences 3

# Summarize a text file:
python -m summarization.frequency --article-file my.txt --sentences 3

# Batch process entire dataset:
python -m summarization.frequency --batch --input dataset/processed/cnn_dailymail_test_1000_processed.jsonl --output dataset/processed/frequency_summaries_1000.jsonl --sentences 3
```

### 4. Running the Test Suite

```bash
python -m pytest tests/ -v
```

All 33 test cases pass across dataset acquisition, preprocessing pipeline, and frequency-based summarization.

---

## 🔬 Methodology & Formulations

### Frequency-Based Scoring

1. **Word Frequency Counter**:
   $$\text{freq}(w) = \text{Count of word } w \text{ in article content tokens}$$
2. **Normalized Frequency**:
   $$\text{NF}(w) = \frac{\text{freq}(w)}{\max_{v \in V} \text{freq}(v)}$$
3. **Sentence Score**:
   $$\text{Score}(S) = \frac{\sum_{w \in C(S)} \text{NF}(w)}{|C(S)|}$$
   where $C(S)$ is the set of content words (non-stopwords, alphabetic) in sentence $S$.
4. **Summary Construction**:
   Top $k$ scoring sentences are selected and reordered chronologically by original sentence index to preserve discourse flow.

---

## 📜 Academic Integrity & License

This project adheres to academic guidelines:
- Strictly extractive and transparent algorithms without generative hallucinations.
- All references and benchmarks follow standard NLP literature.