# Phase 9: Notes and Article History Module

## 1. Scope and Objective

This module implements **Phase 9: Notes and Article History** for the academic project:
**“NLP-Based News Article Summarization and Vocabulary Learning System.”**

The objective of Phase 9 is to provide a reliable, lightweight local persistence layer to:
1. **Archive Article History**: Save previously processed news articles, their generated extractive summaries (Frequency, TF-IDF, TextRank), and associated vocabulary learning metadata from Phase 7 (extracted vocabulary) and Phase 8 (new vocabulary).
2. **Manage Vocabulary Notes**: Create, retrieve, paginate, search, update, and delete personal vocabulary study notes and example sentences with optional associations to saved article history entries.

### Non-Goals & Boundaries:
- **No FastAPI or MongoDB Yet**: In accordance with the project roadmap, FastAPI backend endpoints and MongoDB document persistence belong to Phases 11 and 12. This module provides pure, reusable Python data access objects using standard library SQLite.
- **No User Authentication**: Multi-tenant user auth belongs to Phase 14.
- **No Text-to-Speech (TTS)**: Belongs to Phase 10.
- **Strict Classical NLP**: Zero LLMs, transformers, neural embeddings, or generative APIs.

---

## 2. Persistence Architecture & Design Decisions

- **Engine**: SQLite via Python's standard-library `sqlite3` module.
- **Why SQLite**: Zero external database server dependencies, ACID transaction compliance, cross-platform file storage, and instant memory-database support (`:memory:`) for perfectly isolated unit testing.
- **Configurable Database Path**: Default: `database/notes_history.db`.
- **SQL Injection Prevention**: 100% of queries use parameterized SQL bindings (`?` placeholders).
- **Relational Integrity**: Foreign-key constraints are explicitly activated via `PRAGMA foreign_keys = ON;`. The note-to-article relationship uses `ON DELETE SET NULL`, ensuring personal notes are never accidentally lost if a parent article history record is removed.
- **Non-Mutating Execution**: Never modifies caller-supplied dictionaries or objects.

---

## 3. Database Schema

### Table: `article_history`

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Unique history record identifier |
| `source_article_id` | `TEXT` | `NULLABLE` | Original CNN/DailyMail or external dataset article ID |
| `title` | `TEXT` | `NULLABLE` | Headline or article title |
| `article_text` | `TEXT` | `NULLABLE` | Full article text or text excerpt |
| `summary` | `TEXT` | `NOT NULL` | Generated extractive summary text |
| `method` | `TEXT` | `NOT NULL` | Summarization method (`textrank`, `tfidf`, `frequency`) |
| `processing_timestamp`| `TEXT` | `NOT NULL` | UTC ISO 8601 timestamp string |
| `extracted_vocabulary`| `TEXT` | `NULLABLE` | JSON-serialized vocabulary list from Phase 7 |
| `new_vocabulary` | `TEXT` | `NULLABLE` | JSON-serialized new words list from Phase 8 |
| `metadata` | `TEXT` | `NULLABLE` | JSON-serialized evaluation metrics and configuration |

**Indexes**:
- `idx_ah_source_id`: ON `article_history(source_article_id)`
- `idx_ah_method`: ON `article_history(method)`
- `idx_ah_timestamp`: ON `article_history(processing_timestamp DESC)`

---

### Table: `vocabulary_notes`

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Unique note identifier |
| `term` | `TEXT` | `NULLABLE` | Target vocabulary term (optional for general notes) |
| `content` | `TEXT` | `NOT NULL` | Note content, definition, or learning tip |
| `example_sentence` | `TEXT` | `NULLABLE` | Example usage sentence |
| `article_history_id` | `INTEGER` | `NULLABLE` | Foreign key referencing `article_history(id) ON DELETE SET NULL` |
| `created_at` | `TEXT` | `NOT NULL` | UTC ISO 8601 creation timestamp |
| `updated_at` | `TEXT` | `NOT NULL` | UTC ISO 8601 last update timestamp |

**Indexes**:
- `idx_notes_term`: ON `vocabulary_notes(term)`
- `idx_notes_article_id`: ON `vocabulary_notes(article_history_id)`
- `idx_notes_created_at`: ON `vocabulary_notes(created_at DESC)`

---

## 4. Public Python API

Exported via `database.notes_history` and `database`:

```python
from database import NotesHistoryManager

# 1. Initialize manager (default: database/notes_history.db)
manager = NotesHistoryManager()

# Or in-memory for testing:
# manager = NotesHistoryManager(db_path=":memory:")
```

### 4.1 Article History Operations

#### Save Article History
```python
history_id = manager.save_article_history(
    summary="Extractive summary text here.",
    method="textrank",
    source_article_id="f001ec5c4704938247d27a44948eebb37ae98d01",
    title="International Criminal Court Palestine Proceedings",
    article_text="Full news text...",
    extracted_vocabulary=[{"term": "ICC", "pos": "PROPN", "importance_score": 0.5545}],
    new_vocabulary=[{"term": "ICC", "pos": "PROPN", "importance_score": 0.5545}],
    metadata={"num_sentences_selected": 3, "rouge1_f1": 0.285},
)
```

#### Retrieve by History ID
```python
record = manager.get_article_history(history_id)
# Returns dictionary with automatically deserialized JSON fields, or None
```

#### List with Pagination & Method Filter
```python
# Page 1, limit 10, filtered by 'textrank'
records = manager.list_article_history(limit=10, offset=0, method="textrank")
total_count = manager.count_article_history(method="textrank")
```

#### Search Across Text Fields
```python
# Parameterized search across title, article_text, summary, source_article_id
results = manager.search_article_history(query="Palestine", limit=20)
```

#### Delete History Entry
```python
deleted = manager.delete_article_history(history_id)  # Returns True if deleted
```

#### Clear History (Protected)
```python
# Requires explicit confirmation to prevent accidental data loss
manager.clear_article_history(confirm=True)
```

---

### 4.2 Vocabulary Notes Operations

#### Create Note
```python
# Associated with a vocabulary word and article history
note_id = manager.create_note(
    term="jurisdiction",
    content="The official power to make legal decisions and judgments.",
    example_sentence="The court affirmed its territorial jurisdiction.",
    article_history_id=history_id,
)

# General study note (without term or article)
general_id = manager.create_note(
    content="Review legal terminology before the upcoming NLP viva.",
)
```

#### Retrieve Note
```python
note = manager.get_note(note_id)
```

#### List Notes with Pagination & Filters
```python
# Filter by vocabulary term (case-insensitive)
notes_term = manager.list_notes(term="jurisdiction")

# Filter by associated article history ID
notes_article = manager.list_notes(article_history_id=history_id)
```

#### Search Notes
```python
# Searches content, term, and example_sentence
results = manager.search_notes("territorial")
```

#### Update Note
```python
updated = manager.update_note(
    note_id=note_id,
    content="Revised definition: Authority to interpret and apply the law.",
    example_sentence="Federal courts have jurisdiction over federal offenses.",
)
```

#### Delete Note
```python
deleted = manager.delete_note(note_id)
```

---

## 5. Testing and Verification

Run the dedicated Phase 9 test suite:
```powershell
pytest tests/test_notes_history.py -v
```

Run the complete project test suite (Phases 1 through 9):
```powershell
pytest -v
```
All 157 tests must pass.

---

## 6. Future Integration Path (Phases 11 & 12)

This module has been designed for clean transition to the upcoming web and database tiers:
1. **FastAPI (Phase 11)**: The `NotesHistoryManager` API methods map directly to REST endpoints (`GET /api/history`, `POST /api/history`, `GET /api/notes`, `POST /api/notes`, `PUT /api/notes/{id}`).
2. **MongoDB (Phase 12)**: The JSON-serializable dictionaries returned by `get_article_history()` and `get_note()` are schema-compatible with MongoDB documents (`BSON`).
