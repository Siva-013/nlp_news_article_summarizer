# Phase 8: New-Word Detection Module

## 1. Purpose and Scope

This module implements **Phase 8: Classical New-Word Detection** for the academic project:
**“NLP-Based News Article Summarization and Vocabulary Learning System.”**

While Phase 7 answers the question: *“What salient vocabulary occurs in this news article?”*, Phase 8 answers:
> **“Which of those extracted vocabulary items are unfamiliar to a particular user, based on their explicitly supplied known-vocabulary list?”**

### Strict Scope & Non-Goals:
- **Relative Novelty Only**: A word is classified as "new" strictly if it is absent from the user's supplied known-vocabulary list under canonical normalization.
- **No Difficulty Classifiers**: The system does not invent a synthetic readability or difficulty score.
- **No User Profiles or Databases**: User state tracking, MongoDB persistence, spaced repetition, and frontend flashcard reviews belong to later phases (Phases 9–14).
- **Strict Classical NLP**: Zero LLMs, transformers, neural embeddings, or external AI APIs.

---

## 2. Public API Architecture

The module provides the `NewWordDetector` class in `vocabulary/new_word_detector.py` (lazily re-exported via `vocabulary/__init__.py`).

### Instantiation
```python
from vocabulary.new_word_detector import NewWordDetector

# 1. From an in-memory iterable of terms or dicts
detector = NewWordDetector(known_vocabulary=["say", "court", "today", "cnn"])

# 2. From a file on disk (.json, .txt, or .jsonl)
detector = NewWordDetector.from_file("path/to/known_vocabulary.json")
```

### Core Methods
- `load_known_vocabulary(items)`: Loads and normalizes an iterable of terms/metadata dictionaries.
- `load_from_file(file_path)`: Loads known terms from plain text, JSON list, or JSON dictionary.
- `is_known(item: Dict[str, Any]) -> bool`: Evaluates whether an individual Phase 7 item matches the known vocabulary.
- `detect_new_words(extraction_record: Dict[str, Any]) -> Dict[str, Any]`: Classifies all items in an extraction record into `known_words` and `new_words`. Non-mutating and JSON-serializable.
- `batch_detect_file(input_jsonl, output_jsonl, limit=None) -> Tuple[int, Dict[str, Any]]`: Streams a Phase 7 JSONL dataset and writes the Phase 8 output dataset with progress reporting and aggregated statistics.

---

## 3. Canonical Matching and Normalization Rules

To ensure deterministic, explainable classification:

1. **Case-Insensitive & Whitespace-Stripped Matching**:
   All user inputs and candidate terms are lowercased and stripped of leading/trailing whitespace.
2. **Dual-Key Lookup**:
   A candidate item from Phase 7 is classified as **`known_words`** if:
   - Its lowercased `normalized_form` is in the known vocabulary list.
   - **OR** its canonical morphological `lemma` is in the known vocabulary list (e.g., candidate inflected form *"announced"* matches if the user knows lemma *"announce"*).
   - **OR** its lowercased original surface `term` matches a known entry.
3. **Multi-Word Phrase Matching**:
   Multi-word phrases (e.g., *"United Nations"*, *"machine learning"*) match exact normalized multi-word entries.
4. **Named Entity Awareness**:
   If a known entry specifies an explicit entity label (e.g., `{"term": "Amazon", "entity_label": "ORG"}`), it matches candidates bearing that exact text and label.
5. **No Fuzzy Matching**:
   Strict exact-string matching prevents false-positive classifications that could erroneously mark unfamiliar words as known.

---

## 4. Input & Output Schemas

### Input (Phase 7 Extraction Record)
```json
{
  "article_id": "f001ec5c4704938247d27a44948eebb37ae98d01",
  "method": "classical_vocabulary_extraction",
  "items": [
    {
      "term": "said",
      "normalized_form": "said",
      "lemma": "say",
      "pos": "VERB",
      "frequency": 11,
      "sentence_frequency": 10,
      "sentence_indices": [7, 8, 10, 12, 14, 16, 17, 18, 20, 23],
      "entity_label": null,
      "importance_score": 0.5429,
      "selection_reason": "Salient VERB occurring 11 time(s) across 10 sentence(s) (coverage: 35.7%)."
    }
  ]
}
```

### Output (Phase 8 Detection Record)
```json
{
  "article_id": "f001ec5c4704938247d27a44948eebb37ae98d01",
  "method": "classical_new_word_detection",
  "matching_policy": "canonical_lemma_and_normalized_matching",
  "total_vocabulary_count": 30,
  "known_word_count": 7,
  "new_word_count": 23,
  "known_words": [
    {
      "term": "said",
      "normalized_form": "said",
      "lemma": "say",
      "pos": "VERB",
      "frequency": 11,
      "sentence_frequency": 10,
      "sentence_indices": [7, 8, 10, 12, 14, 16, 17, 18, 20, 23],
      "entity_label": null,
      "importance_score": 0.5429,
      "selection_reason": "Salient VERB occurring 11 time(s) across 10 sentence(s) (coverage: 35.7%)."
    }
  ],
  "new_words": [
    {
      "term": "ICC",
      "normalized_form": "icc",
      "lemma": "icc",
      "pos": "PROPN",
      "frequency": 7,
      "sentence_frequency": 7,
      "sentence_indices": [3, 4, 6, 10, 14, 18, 20],
      "entity_label": "ORG",
      "importance_score": 0.5545,
      "selection_reason": "Named entity (ORG) occurring 7 time(s) across 7 sentence(s) (coverage: 25.0%)."
    }
  ],
  "validation": {
    "reconciled": true,
    "difference": 0
  }
}
```

### Exact Accounting Guarantee
Every eligible candidate entry is accounted for exactly once:
$$\text{total\_vocabulary\_count} = \text{known\_word\_count} + \text{new\_word\_count}$$

---

## 5. Command-Line Interface (CLI)

### Single Article Inspection by ID
```powershell
python -m vocabulary.new_word_detector --input dataset/processed/vocabulary_extractions_1000.jsonl --known-vocab dataset/processed/synthetic_known_vocabulary.json --id f001ec5c4704938247d27a44948eebb37ae98d01
```

### Using Inline Known Words
```powershell
python -m vocabulary.new_word_detector --input dataset/processed/vocabulary_extractions_1000.jsonl --known-words "said,court,today,cnn,wednesday" --id f001ec5c4704938247d27a44948eebb37ae98d01
```

### Full Batch Execution (1,000 Articles)
```powershell
python -m vocabulary.new_word_detector --input dataset/processed/vocabulary_extractions_1000.jsonl --known-vocab dataset/processed/synthetic_known_vocabulary.json --output dataset/processed/new_word_detections_1000.jsonl --batch
```

---

## 6. Testing

The dedicated test suite covers 24 verification points:
```powershell
pytest tests/test_new_word_detection.py -v
```

Run the entire system test suite (Phases 1 through 8):
```powershell
pytest -v
```
All 133 tests must pass.

---

## 7. Known Limitations

1. **Relative to Input List**: The accuracy of detection depends directly on the completeness of the supplied known vocabulary. If a user does not include a word they know in the list, the system will classify it as "new".
2. **Context-Independent Known Words**: Polysemous words (e.g., *bank* as financial institution vs. river bank) are matched purely lexically, in accordance with classical NLP token matching.
