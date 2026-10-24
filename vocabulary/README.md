# Phase 7: Vocabulary Extraction Module

## 1. Objective and Scope

This module implements **Phase 7: Classical Rule-Based Vocabulary Extraction** for the academic project:
**“NLP-Based News Article Summarization and Vocabulary Learning System.”**

The primary objective of Phase 7 is to extract informative, high-salience vocabulary items (lexical content words and recognized named entities) from news articles using classical, explainable NLP methods. It enriches extracted terms with morphological lemmas, grammatical parts of speech, token and sentence frequencies, sentence positions, and transparent heuristic importance scores.

> **Strict Scope Notice**:
> This module does **not** include user vocabulary profiles, spaced repetition, flashcard tracking, or new-word detection (deferred to Phase 8). It relies entirely on classical NLP and does not use LLMs, embeddings, transformers, or external AI APIs.

---

## 2. Files Added

- `vocabulary/extractor.py`: Core extraction engine (`VocabularyExtractor`), scoring formulas, batch processing, and CLI.
- `vocabulary/__init__.py`: Package exports with lazy attribute resolution to prevent `runpy` module warnings.
- `vocabulary/README.md`: This comprehensive technical guide and usage manual.
- `tests/test_vocabulary_extraction.py`: 25 comprehensive unit and integration tests.
- `docs/vocabulary_extraction_analysis.md`: Empirical analysis report on 1,000 CNN/DailyMail articles.
- `dataset/processed/vocabulary_extractions_1000.jsonl`: 1,000-article batch extraction dataset.

---

## 3. Input Schema and Expected Fields

The extractor accepts two input modes:

### A. Phase 2 Processed Article Record (`Dict[str, Any]`)
Processes pre-tokenized and annotated records without re-running spaCy:
- `id` (`str`): Unique article identifier.
- `processed_sentences` (`List[Dict[str, Any]]`): List of sentence dictionaries containing:
  - `sentence_index` (`int`): 0-based index of sentence in article.
  - `tokens` (`List[str]`): Raw token strings.
  - `normalized_tokens` (`List[str]`): Lowercased tokens.
  - `lemmas` (`List[str]`): Canonical morphological roots.
  - `pos` (`List[str]`): Coarse Universal Dependencies POS tags (`NOUN`, `VERB`, etc.).
  - `token_details` (`List[Dict[str, Any]]`, optional): Rich token annotations with `is_stop`, `is_punct`, `normalized`, `pos`, etc.
  - `entities` (`List[Dict[str, Any]]`, optional): Named entity annotations with `text`, `label`, `start_char`, `end_char`.

### B. Raw Text String (`str`)
Passed to `extract_from_text()`, which lazy-loads `PreprocessingPipeline` to generate the complete Phase 2 representation on the fly.

---

## 4. Extraction Pipeline

```
Raw Text OR Processed Record
             │
             ▼
   [Sentence Traversal]
             │
    ┌────────┴──────────────────────────┐
    ▼                                   ▼
Content Words Extraction        Named Entity Recognition
- Filter stops & punctuation    - Multiword / single spans
- Keep NOUN, PROPN, VERB,       - Preserve exact NER label
  ADJ, ADV                      - Preserve entity surface
    │                                   │
    └────────┬──────────────────────────┘
             ▼
[Linguistic Aggregation]
- Content Words: aggregated by (lemma, POS)
- Named Entities: aggregated by (entity_text_lower, entity_label)
- Accumulate: frequency, sentence_indices, surface_form counts
             │
             ▼
[Heuristic Importance Scoring]
- Normalized frequency
- Sentence coverage ratio
- Named entity bonus & POS weights
             │
             ▼
[Deterministic Ranking & Pruning]
- Sort: score DESC -> freq DESC -> first_sent ASC -> term ASC
- Return top-K items in structured JSON format
```

---

## 5. Content-Word Filtering Rules

1. **Stop Words & Punctuation**: Tokens where `is_stop == True` or `is_punct == True` are excluded.
2. **Grammatical Categories**: Accepted Universal Dependencies tags:
   - `NOUN` (common nouns)
   - `PROPN` (proper nouns)
   - `VERB` (lexical verbs, excluding modal/auxiliary verbs marked as stop words)
   - `ADJ` (adjectives)
   - `ADV` (adverbs)
3. **Alphanumeric Check**: Tokens must contain at least one alphanumeric character (`any(c.isalnum() for c in token)`).
4. **Minimum Term Length**: Default `min_term_length = 2` characters.

---

## 6. Lemma and Entity Handling

- **Inflection Aggregation**: Inflected variants of ordinary words (e.g., *announces*, *announced*, *announcing*) aggregate under their morphological canonical lemma (*announce*).
- **Representative Surface Form**: The most frequently occurring capitalization and surface form in the article is returned as `term`, ensuring human readability.
- **Named Entities**: Multi-word and single-word entities are retained with their full surface text and categorized by their exact NER label (`PERSON`, `ORG`, `GPE`, `DATE`, `EVENT`, etc.). Entities are given an entity bonus and are kept distinct from generic content words.

---

## 7. Frequency Definitions

- **Token Frequency ($TF$)**: Total number of occurrences of the term/entity across the entire article.
- **Sentence Frequency ($SF$)**: Total number of distinct sentences in which the term appears (counted at most once per sentence).
- **Sentence Coverage ($SC$)**: Ratio of sentences containing the term:
  $$SC = \frac{SF}{N_{\text{sentences}}}$$

---

## 8. Importance Scoring Formula and Default Weights

The importance score is a deterministic, explainable heuristic:

$$\text{score}(t) = \left(\alpha \cdot \frac{TF(t)}{\max_{w} TF(w)} + \beta \cdot \frac{SF(t)}{N_{\text{sentences}}} + \gamma \cdot \text{EntityBonus}(t)\right) \times \text{POSMultiplier}(\text{pos}(t))$$

### Default Parameters
- $\alpha = 0.4$ (Term frequency weight)
- $\beta = 0.4$ (Sentence coverage weight)
- $\gamma = 0.2$ (Named entity bonus: $1.0$ if named entity, else $0.0$)
- **Constraint**: $\alpha + \beta + \gamma = 1.0$
- **POS Multipliers**: `NOUN`: 1.0, `PROPN`: 1.0, `VERB`: 1.0, `ADJ`: 1.0, `ADV`: 0.8.

---

## 9. Deterministic Ranking & Tie-Breaking

To guarantee reproducibility across runs and platforms, candidates are sorted by:
1. `importance_score` (Descending)
2. `frequency` (Descending)
3. `_first_sentence_idx` (Ascending — earlier appearance in article)
4. `_first_token_offset` (Ascending)
5. `normalized_form` (Ascending — alphabetical tie-breaker)

---

## 10. Output Schema

```json
{
  "article_id": "f001ec5c4704938247d27a44948eebb37ae98d01",
  "method": "classical_vocabulary_extraction",
  "configuration": {
    "max_items": 30,
    "include_entities": true,
    "weights": {
      "alpha_frequency": 0.4,
      "beta_coverage": 0.4,
      "gamma_entity": 0.2
    }
  },
  "total_candidate_items": 238,
  "returned_item_count": 30,
  "items": [
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
    },
    {
      "term": "said",
      "normalized_form": "said",
      "lemma": "say",
      "pos": "VERB",
      "frequency": 11,
      "sentence_frequency": 10,
      "sentence_indices": [1, 3, 5, 7, 9, 11, 15, 17, 19, 21],
      "entity_label": null,
      "importance_score": 0.5429,
      "selection_reason": "Salient VERB occurring 11 time(s) across 10 sentence(s) (coverage: 35.7%)."
    }
  ]
}
```

---

## 11. Installation & Environment

This module runs within the project's virtual environment:
```powershell
# Virtual environment activation (Windows)
.\.venv\Scripts\Activate.ps1
```

Dependencies used:
- Python 3.13+
- `spacy` (`en_core_web_sm`)
- `tqdm`

---

## 12. CLI Usage Examples

### Single Article from Preprocessed Dataset by ID
```powershell
python -m vocabulary.extractor --dataset dataset/processed/cnn_dailymail_test_1000_processed.jsonl --id f001ec5c4704938247d27a44948eebb37ae98d01
```

### Raw Text String
```powershell
python -m vocabulary.extractor --text "Artificial intelligence is advancing rapidly. Researchers at Stanford University published groundbreaking papers on machine learning."
```

### Raw Text File
```powershell
python -m vocabulary.extractor --article-file sample_article.txt --max-items 20
```

### Batch Extraction across 1,000 Articles
```powershell
python -m vocabulary.extractor --dataset dataset/processed/cnn_dailymail_test_1000_processed.jsonl --output dataset/processed/vocabulary_extractions_1000.jsonl --batch
```

---

## 13. Python API Usage

```python
from vocabulary.extractor import VocabularyExtractor

# Initialize extractor
extractor = VocabularyExtractor(max_items=25, include_entities=True)

# 1. From preprocessed record (instantaneous, no spaCy reload)
record = {...} # Dict from cnn_dailymail_test_1000_processed.jsonl
result = extractor.extract_from_processed_record(record)

# 2. From raw string
raw_text = "The Federal Reserve raised interest rates yesterday..."
result = extractor.extract_from_text(raw_text, article_id="fed_news")

for item in result["items"]:
    print(f"{item['term']} ({item['pos']}) - Score: {item['importance_score']:.4f}")
```

---

## 14. Testing Instructions

Run the dedicated Phase 7 test suite:
```powershell
pytest tests/test_vocabulary_extraction.py -v
```

Run the entire project test suite (Phases 1 through 7):
```powershell
pytest -v
```
All 109 tests across all project phases must pass.

---

## 15. Known Limitations

1. **Corpus-Level IDF**: Phase 7 scores terms based on within-article salience and coverage. Cross-article IDF can further downweight high-frequency domain words (e.g., *said*, *told*).
2. **Ambiguous POS in Headlines**: Capitalized short headlines without full sentence structure occasionally misclassify words as `PROPN`.
3. **No Dictionary Definitions**: In accordance with the classical NLP constraints, external dictionary definitions are not retrieved in this phase.

---

## 16. Academic Scope Confirmation

**Phase 7 is strictly confined to vocabulary extraction and statistical feature scoring.**
No user vocabulary models, proficiency levels, flashcards, or Phase 8 features have been implemented.
