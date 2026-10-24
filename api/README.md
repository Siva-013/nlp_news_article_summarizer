# Phase 11: FastAPI Backend Integration

## Overview

The `api` package exposes the complete NLP pipeline of the **NLP-Based News Article Summarization and Vocabulary Learning System** via a clean, high-performance, testable RESTful API built with **FastAPI**.

This API strictly integrates classical, explainable NLP algorithms implemented in Phases 1–10:
- **Phase 3, 4, 5 Extractive Summarization**: Frequency, TF-IDF, and TextRank methods.
- **Phase 6 Evaluation**: First-principles ROUGE-1, ROUGE-2, and ROUGE-L computation.
- **Phase 7 Vocabulary Extraction**: Deterministic POS and frequency-based salience scoring.
- **Phase 8 New-Word Detection**: Canonical morphological and normalized term novelty classification.
- **Phase 9 Notes and Article History**: Parameterized SQLite persistence with foreign keys (`ON DELETE SET NULL`).
- **Phase 10 Speech Synthesis (TTS)**: Offline WAV audio generation via `pyttsx3`.

---

## Directory Structure

```
api/
├── __init__.py           # Package declaration & version
├── dependencies.py       # Dependency injection providers & safe path validators
├── main.py               # FastAPI application, CORS middleware, lifespan & health routes
├── README.md             # API documentation and usage guide
├── schemas.py            # Pydantic V2 models for requests, responses, and validations
├── services.py           # Domain service wrappers orchestrating NLP modules
└── routers/
    ├── __init__.py       # Routers package
    ├── evaluate.py       # POST /api/v1/evaluate
    ├── history.py        # CRUD on /api/v1/history
    ├── notes.py          # CRUD on /api/v1/notes
    ├── speech.py         # /api/v1/speech (synthesize, audio stream, voices)
    ├── summarize.py      # POST /api/v1/summarize
    └── vocabulary.py     # POST /api/v1/vocabulary/extract & /detect-new
```

---

## Running the API Server

From the project root (`D:\nlp project`):

```powershell
# Using the project's virtual environment
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Once running:
- **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## API Endpoints Reference

### 1. Root & Health
- `GET /`: Returns API metadata, status, version, and endpoint sitemap.
- `GET /health`: Verifies database connectivity and service availability.

### 2. Extractive Summarization (`/api/v1/summarize`)
- `POST /api/v1/summarize`
  - **Request Body**:
    ```json
    {
      "text": "London is the capital and largest city of England...",
      "method": "frequency",
      "num_sentences": 3
    }
    ```
  - **Supported Methods**: `"frequency"`, `"tfidf"`, `"textrank"`.
  - **Response**: Extracted summary, selected sentences with individual scores, and rankings.

### 3. Vocabulary Processing (`/api/v1/vocabulary/*`)
- `POST /api/v1/vocabulary/extract`
  - Extracts high-salience content words and named entities with deterministic heuristic scores.
  - Parameters: `text`, `max_items` (default: 30), `include_entities` (default: true).
- `POST /api/v1/vocabulary/detect-new`
  - Classifies vocabulary into `known_words` and `new_words` against a user's known vocabulary list.
  - Accepts either raw `text` or a pre-extracted Phase 7 `extraction_record`.

### 4. Article History Persistence (`/api/v1/history/*`)
- `POST /api/v1/history`: Save summary, article text, and vocabulary metadata (HTTP 201).
- `GET /api/v1/history`: List articles with pagination (`limit`, `offset`) and method filter.
- `GET /api/v1/history/search`: Full-text keyword search across titles, summaries, and text.
- `GET /api/v1/history/{history_id}`: Retrieve a specific history record by ID.
- `DELETE /api/v1/history/{history_id}`: Delete an article history record (sets FK on linked notes to NULL).

### 5. Vocabulary Notes Persistence (`/api/v1/notes/*`)
- `POST /api/v1/notes`: Create note with content, term, example sentence, and optional article link (HTTP 201).
- `GET /api/v1/notes`: List notes with pagination and term/article filters.
- `GET /api/v1/notes/search`: Search across note contents, terms, and example sentences.
- `GET /api/v1/notes/{note_id}`: Retrieve note by ID.
- `PUT /api/v1/notes/{note_id}`: Update note content, term, or associations.
- `DELETE /api/v1/notes/{note_id}`: Delete note by ID.

### 6. Text-to-Speech (`/api/v1/speech/*`)
- `POST /api/v1/speech/synthesize`: Synthesizes summary text into an offline WAV file, returning an `audio_id` and download URL.
- `GET /api/v1/speech/audio/{audio_id}`: Streams or downloads the generated WAV audio file (guarantees path traversal prevention).
- `GET /api/v1/speech/voices`: Lists available system speech synthesis voices.

### 7. Quantitative Evaluation (`/api/v1/evaluate`)
- `POST /api/v1/evaluate`: Computes exact precision, recall, and F1 for ROUGE-1, ROUGE-2, and ROUGE-L between candidate and reference summaries.

---

## Example Usage (PowerShell & Curl)

### Summarize an Article
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/summarize" -Method Post `
  -ContentType "application/json" `
  -Body '{"text": "London is the capital of the UK. It is on the River Thames. It is ancient.", "method": "textrank", "num_sentences": 2}'
```

### Extract Vocabulary
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/vocabulary/extract" -Method Post `
  -ContentType "application/json" `
  -Body '{"text": "London is the capital of England. Dogs love running in parks.", "max_items": 10}'
```

### Compute ROUGE Scores
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/evaluate" -Method Post `
  -ContentType "application/json" `
  -Body '{"candidate_summary": "London is England capital.", "reference_summary": "London is the capital of England."}'
```
