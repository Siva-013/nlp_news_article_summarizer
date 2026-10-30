# NLP-Based News Article Summarization and Vocabulary Learning System

An end-to-end academic Natural Language Processing system for extractive news article summarization and vocabulary enrichment, built using foundational NLP algorithms (without external black-box generative models or closed APIs).

---

## 📌 Project Overview

This project implements classical and modern NLP techniques on the **CNN/DailyMail (v3.0.0)** dataset. It focuses on transparent, explainable summarization pipelines coupled with rigorous linguistic preprocessing, statistical evaluation, and reproducible benchmarks — delivered as a complete full-stack web application with secure multi-user accounts, persistent notes and history, offline text-to-speech, and interactive ROUGE evaluation.

### Development Phases

The project progressed through thirteen phases, each building on the dual-representation
foundation of Phase 2 (normalized token streams for mathematical scoring, verbatim sentence
text for extractive output):

| Phase | Title | What Was Built | Key Artifacts |
|-------|-------|----------------|---------------|
| **1** | Dataset Setup & Exploration | Streaming acquisition of CNN/DailyMail test splits; statistical profiling (token counts, compression ratios, TTR); deterministic sampling of 1,000 records (seed 42). | `dataset/`, `docs/dataset_analysis_1000.md` |
| **2** | Data Preprocessing Pipeline | HTML stripping, Unicode NFKC normalization, hardened spaCy sentence segmentation and infix tokenization, lemmatization, stop-word filtering, POS tagging, NER with exact character offsets. | `preprocessing/`, `docs/preprocessing_analysis.md` |
| **3** | Frequency-Based Summarization | Luhn-inspired baseline: normalized word frequencies, length-normalized sentence scoring, deterministic tie-breaking, chronological reordering, batch CLI. | `summarization/frequency.py`, `docs/frequency_baseline_analysis.md` |
| **4** | TF-IDF Summarization | Intra-article TF-IDF: sentence-level term frequency, inverse sentence-document frequency `IDF(t) = ln(N / DF(t))`, 8-stage scoring pipeline computed on content tokens. | `summarization/tfidf.py`, `docs/tfidf_baseline_analysis.md` |
| **5** | TextRank Summarization | Graph-based centrality from first principles: sentences as nodes, TF-IDF cosine similarity as weighted edges, PageRank-style iterative ranking (damping 0.85) — no networkx or prebuilt rankers. | `summarization/textrank.py`, `docs/textrank_baseline_analysis.md` |
| **6** | Quantitative Evaluation | ROUGE-1/2/L implemented from scratch (precision / recall / F1); macro benchmark of all three summarizers over 1,000 articles; per-article win statistics and summary-length analysis. | `evaluation/`, `docs/evaluation_analysis.md` |
| **7** | Vocabulary Extraction | Salient content words and named entities with morphological, syntactic and positional metadata plus a deterministic, explainable importance score; five-stage extraction pipeline. | `vocabulary/extractor.py`, `docs/vocabulary_extraction_analysis.md` |
| **8** | New-Word Detection | Relative-novelty classification against a learner's known vocabulary via canonical normalization and morphological lemma matching (inflected forms match base lemmas); multi-word phrase and entity-aware matching. | `vocabulary/new_word_detector.py`, `docs/new_word_detection_analysis.md` |
| **9** | Notes & History Persistence | SQLite-backed storage for vocabulary notes and every summarization run; typeahead-friendly schema; WAL mode; JSON-serializable records designed for a future REST layer. | `database/`, `docs/notes_history_analysis.md` |
| **10** | Text-to-Speech Integration | Offline TTS studio (voice picker, speed/volume, WAV download). Each synthesis runs in an **isolated subprocess** so a stuck SAPI/COM call can never block the API. | `speech/`, `docs/text_to_speech_analysis.md` |
| **11** | FastAPI Backend Integration | Versioned REST API (`/api/v1`) connecting Phases 1–10: clean routing (`api/routers/`), domain services, Pydantic V2 validation, dependency injection, path-traversal hardening. | `api/`, `docs/phase11_api_analysis.md` |
| **12** | Lexica Web Application | React + Vite single-page research console with six workspaces: Summarize (three methods side by side, per-sentence scores, top-word charts), Vocabulary, Notes, History, Listen, and Evaluate — with a live backend health indicator. | `frontend/` |
| **13** | Authentication & User Isolation | Per-user accounts with bcrypt-hashed passwords and signed JWT bearer tokens; every notes/history endpoint is user-scoped at the query level, so no account can read or modify another user's data. | `api/security.py`, `api/routers/auth.py`, `tests/test_auth.py` |

**Headline benchmark** (Phase 6, mean ROUGE F1 over 1,000 articles): TextRank 0.254 / 0.085 / 0.175 (R1/R2/L), Frequency 0.231 / 0.064 / 0.162, TF-IDF 0.120 / 0.021 / 0.090 — TextRank is the strongest method on every metric.

---

## 📂 Project Structure

```text
D:\nlp project/
├── api/                              # Phase 11: FastAPI service layer
│   ├── main.py                       # App instance, lifespan, CORS, health
│   ├── schemas.py                    # Pydantic V2 request/response models
│   ├── security.py                   # bcrypt hashing + JWT issuance/verification
│   ├── services.py                   # Domain orchestration of NLP modules
│   ├── dependencies.py               # DI: DB session, TTS engine, audio dir
│   └── routers/                      # auth, summarize, vocabulary, notes,
│                                     # history, speech, evaluate
├── database/                         # Phase 9: SQLite persistence
│   ├── notes_history.py              # Notes/history data access layer
│   └── notes_history.db              # SQLite database (created at runtime)
├── dataset/                          # Phase 1: acquisition + exploration
│   ├── download_dataset.py           # Dataset acquisition script
│   ├── explore_dataset.py            # Statistical exploration script
│   ├── dataset_stats_1000.json       # Dataset metrics (1,000 articles)
│   ├── raw/                          # Staged JSONL splits
│   └── processed/                    # Preprocessed records, summaries, detections
├── docs/                             # Per-phase analysis reports (Phases 1–11)
├── evaluation/                       # Phase 6: ROUGE scoring + comparison
│   ├── rouge.py                      # ROUGE-1/2/L from first principles
│   ├── evaluator.py                  # Batch evaluation driver
│   └── comparison.py                 # Cross-method win statistics
├── frontend/                         # Phases 12–13: React (Vite) "Lexica" web UI
├── preprocessing/                    # Phase 2: linguistic pipeline
│   ├── cleaner.py                    # HTML/text cleaning
│   ├── normalizer.py                 # Unicode & whitespace normalization
│   ├── sentence_splitter.py          # Sentence segmentation
│   ├── tokenizer.py                  # Hardened spaCy tokenizer
│   ├── stopwords.py                  # Stop-word filtering
│   ├── lemmatizer.py                 # Morphological lemmatization
│   ├── linguistic_features.py        # POS tagging and NER
│   └── pipeline.py                   # Master preprocessing pipeline
├── speech/                           # Phase 10: offline TTS (subprocess-isolated)
├── static/audio/                     # Synthesized WAV output directory
├── summarization/                    # Phases 3–5: extractive summarizers
│   ├── frequency.py                  # FrequencySummarizer class and CLI
│   ├── tfidf.py                      # TF-IDF summarizer
│   └── textrank.py                   # TextRank summarizer
├── tests/                            # 236 automated tests across all phases
├── vocabulary/                       # Phases 7–8: term extraction + new-word detection
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

All 236 test cases pass across dataset acquisition, preprocessing, the three summarizers,
vocabulary extraction and new-word detection, notes/history persistence, speech synthesis,
user authentication, and the FastAPI layer.

### 5. Running the Full Application (API + Web UI)

```bash
# Terminal 1 — FastAPI backend (http://localhost:8000, docs at /docs):
.venv\Scripts\activate
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — React frontend (http://localhost:5173):
cd frontend
npm install
npm run dev
```

The web UI ("Lexica") is a dark research-console front end with six workspaces:

- **User accounts** — every visitor signs in with their own username and password.
  Passwords are bcrypt-hashed; sessions use signed JWT bearer tokens. Notes and
  article history are stored per user — nobody can read or modify another
  user's data. Set `JWT_SECRET` in the environment for production-grade token
  signing (a documented dev fallback is used when unset).

- **Summarize** — extractive summarization with Frequency / TF–IDF / TextRank,
  side-by-side comparison of all three methods, per-sentence scores, top-word
  charts, and automatic archiving to history.
- **Vocabulary** — rule-based term extraction with importance scores, POS and
  entity tags, known-word chips, and new-word detection.
- **Notes** — SQLite-backed vocabulary notebook with typeahead search.
- **History** — searchable archive of every summarization run, with method
  filters, expandable source text, and pagination.
- **Listen** — offline text-to-speech studio (voice picker, speed/volume
  controls, WAV download). Each synthesis runs in an isolated subprocess so a
  stuck SAPI/COM call can never block the API.
- **Evaluate** — ROUGE-1/2/L scoring against a reference summary, with
  precision/recall/F1 and overlap verdicts.

The top bar shows a live backend health indicator (polls `/health` every 20 s).
On narrow screens the navigation moves to a bottom tab bar.

---

## 🔬 Methodology & Formulations

### Frequency-Based Scoring (Phase 3)

1. **Word Frequency Counter**:
   $$\text{freq}(w) = \text{Count of word } w \text{ in article content tokens}$$
2. **Normalized Frequency**:
   $$\text{NF}(w) = \frac{\text{freq}(w)}{\max_{v \in V} \text{freq}(v)}$$
3. **Sentence Score**:
   $$\text{Score}(S) = \frac{\sum_{w \in C(S)} \text{NF}(w)}{|C(S)|}$$
   where $C(S)$ is the set of content words (non-stopwords, alphabetic) in sentence $S$.
4. **Summary Construction**:
   Top $k$ scoring sentences are selected and reordered chronologically by original sentence index to preserve discourse flow.

### TF-IDF Scoring (Phase 4)

Each sentence is treated as a document within its article. With $N$ sentences and
$\text{DF}(t)$ the number of sentences containing term $t$:

1. **Inverse Document Frequency**: $\text{IDF}(t) = \ln\left(\frac{N}{\text{DF}(t)}\right)$
2. **Sentence Term Frequency**: $\text{TF}(t, S) = \frac{\text{count}(t \in S)}{|C(S)|}$
3. **Weight & Sentence Score**: $\text{Score}(S) = \frac{\sum_{t \in C(S)} \text{TF}(t, S) \times \text{IDF}(t)}{|C(S)|}$

Recurring terms present in nearly every sentence are discounted; terms concentrated
in a few sentences are boosted. Computed strictly on content tokens with deterministic
tie-breaking (lower sentence index wins).

### TextRank (Phase 5)

The article is modeled as a weighted undirected graph $G = (V, E)$: sentences are nodes and
edges carry TF-IDF cosine similarity over content-token vectors. PageRank-style iterative
propagation (damping $d = 0.85$) from a uniform start distribution yields sentence centrality:

$$\text{WS}(V_i) = (1 - d) + d \times \sum_{j} \frac{W(i, j)}{\sum_{k} W(j, k)} \times \text{WS}(V_j)$$

A sentence is recommended by the sentences it overlaps with, so centrality — not raw term
counts — drives selection. Graph construction and the PageRank solver are implemented from
scratch (lexical similarity only; no embedding-based semantics by design).

### Evaluation (Phase 6)

ROUGE-1/2/L precision, recall and F1 are implemented from first principles and computed
against the CNN/DailyMail reference highlights. Macro results over 1,000 articles:

| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|--------|-----------|-----------|-----------|
| Frequency | 0.231 | 0.064 | 0.162 |
| TF-IDF | 0.120 | 0.021 | 0.090 |
| **TextRank** | **0.254** | **0.085** | **0.175** |

**Findings.** TextRank wins on every metric — graph centrality recovers
reference-highlight-like sentences more often than raw frequency. The Frequency baseline is
a solid, efficient second. TF-IDF underperforms because its distinctiveness penalty
discounts exactly the recurring topical terms that news highlights reuse. These scores are
typical for purely extractive, unsupervised methods and are reported as a reproducible
classical baseline.

---

## 📜 Academic Integrity & License

This project adheres to academic guidelines:
- Strictly extractive and transparent algorithms without generative hallucinations.
- All references and benchmarks follow standard NLP literature.
