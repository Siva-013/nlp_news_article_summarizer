# NLP Preprocessing Pipeline (Phase 2 & Hardening)

## 1. Purpose of Preprocessing
In natural language processing, raw text harvested from journalism platforms, web scrapers, and digital feeds contains irregularities such as HTML markup, non-standard Unicode encodings, erratic whitespace, and unsegmented prose. 

The purpose of this pipeline is to transform unstructured raw news articles into linguistically structured, machine-interpretable representations while strictly preserving the verbatim original wording for downstream extractive summarization.

---

## 2. Why Preprocessing is Essential
Traditional/classical NLP methods (such as TF-IDF scoring and TextRank graph centrality) rely heavily on exact term matching, grammatical category filtering, and graph adjacency. Without rigorous preprocessing:
- HTML entities and tags distort vocabulary frequencies and artificially inflate term weights.
- Decomposed Unicode characters cause identical words to fail string equality comparisons.
- Sentence boundary ambiguity leads to fragmented or incoherent summary candidate sentences.
- Inflected word variants (e.g., *announces*, *announced*, *announcing*) fragment lexical statistics that should converge on a single lemma (*announce*).

---

## 3. Linguistic Stages and Architectural Modules

### Stage 1: HTML & Markup Cleaning (`cleaner.py`)
- **Mechanism**: Strips HTML tags (`<p>`, `<b>`, `<a>`, etc.) and decodes standard HTML entities (`&nbsp;`, `&amp;`, `&quot;`, `&#39;`).
- **Boundary Preservation**: Block-level transition tags (`</p>`, `</div>`, `<br>`, `</li>`) are replaced with boundary spaces to prevent prose from colliding across paragraphs.
- **Academic Rationale**: Does not indiscriminately strip punctuation or numbers, ensuring sentence boundaries, quotations, and financial figures remain intact.

### Stage 2: Unicode Normalization (`normalizer.py`)
- **Mechanism**: Standardizes text into Unicode Normalization Form C (`NFC`) using Python's `unicodedata.normalize`.
- **Academic Rationale**: Resolves decomposed diacritics and composite glyphs into canonical single codepoints, guaranteeing uniform vocabulary hashing.

### Stage 3: Whitespace Normalization (`normalizer.py`)
- **Mechanism**: Replaces Windows carriage returns, multi-space clusters, erratic tabs, and orphan newlines with single spaces and strips leading/trailing margins.
- **Academic Rationale**: Preserves clean paragraph continuity without creating orphan sentence fragments.

### Stage 4: Sentence Segmentation (`sentence_splitter.py`)
- **Mechanism**: Utilizes spaCy's dependency-driven sentence boundary detector (`doc.sents`).
- **Academic Rationale**: Preserves the complete, original sentence wording. Extractive summarization selects genuine sentences from this list to assemble the final summary.

### Stage 5: Hardened Linguistic Tokenization (`tokenizer.py`)
- **Mechanism**: Applies spaCy's rule-based tokenizer configured with **enhanced infix regex rules** (`configure_spacy_tokenizer`).
- **Hardened Boundaries**: Standard spaCy tokenization leaves closing parentheses and brackets attached to subsequent words when there is no intervening space (e.g. `(CNN)The` $\rightarrow$ `['(', 'CNN)The']`). Our hardened tokenizer introduces custom infix rules to correctly split parentheses, brackets, and quotes:
  $$\text{"(CNN)The"} \longrightarrow \text{['(', 'CNN', ')', 'The']}$$
  $$\text{'word,"hello'} \longrightarrow \text{['word', ',', '"', 'hello']}$$
- **Academic Rationale**: Avoids crude whitespace splitting (`text.split()`). Accurately isolates clitics, punctuation, contractions (*"don't"* $\rightarrow$ *"do"*, *"n't"*), numbers (*"123rd"*, *"$100"*), and hyphenated expressions without hardcoded string hacks.

### Stage 6: Text Normalization (`normalizer.py`)
- **Mechanism**: Produces lowercased analytical forms (`normalize_token()`) for vocabulary matching without mutating the original casing of the source sentence.

### Stage 7: Stop-Word Identification (`stopwords.py`)
- **Mechanism**: References spaCy's English stop-word lexicon (`nlp.Defaults.stop_words`).
- **Academic Rationale**: Stop words are **never deleted** from the primary sentence. Instead, boolean flags (`is_stop: bool`) are recorded at token level, and filtered content token lists (`content_tokens`) are maintained exclusively for semantic scoring and graph construction.

### Stage 8: Lemmatization (`lemmatizer.py`)
- **Mechanism**: spaCy's morphological lemmatizer generates canonical dictionary headwords (*"announced"* $\rightarrow$ *"announce"*, *"running"* $\rightarrow$ *"run"*, *"cars"* $\rightarrow$ *"car"*).
- **Academic Rationale**: Normalizes inflectional variation for feature calculation while keeping original sentences readable and unaltered.

### Stage 9: Part-of-Speech Tagging (`linguistic_features.py`)
- **Universal Dependencies (Coarse POS)**: `token.pos_` (`NOUN`, `VERB`, `ADJ`, `ADV`, `PROPN`). Used for high-level semantic filtering.
- **Penn Treebank (Detailed Tag)**: `token.tag_` (`NN`, `NNS`, `VBD`, `VBG`, `JJ`, `JJR`). Captures tense, number, and syntactic inflection.

### Stage 10: Named Entity Recognition (`linguistic_features.py`)
- **Mechanism**: Extracts categorized named entity spans (`PERSON`, `ORG`, `GPE`, `LOC`, `DATE`, `MONEY`, `EVENT`, etc.).
- **Relative Offset Alignment**: Character offsets (`start_char`, `end_char`) are calculated strictly relative to each sentence's original text, guaranteeing that `sentence_text[start_char:end_char] == entity_text`.

---

## 4. Phase 1 vs. Phase 2 Sentence Count Methodology

In Phase 1 exploratory data analysis, the reported mean sentence count was approximately **31.83** sentences/article. In Phase 2, the linguistic segmentation reports approximately **34.2** sentences/article.

### Methodological Distinction:
1. **Phase 1 Method**: Used a simple, lightweight regex split on terminal punctuation boundaries (`re.split(r'(?<=[.!?])\s+')`). This naive approach fails to segment datelines, bullet points without terminal periods, parenthetical clauses, or multi-clause quotes.
2. **Phase 2 Method**: Utilizes spaCy's statistical transition-based dependency parser and rule-based sentence boundary detector (`doc.sents`), hardened with custom dateline parenthesis splitting. For example, introductory datelines like `(CNN)` are accurately identified as distinct sentence boundaries rather than merged into the first paragraph sentence.

Phase 2 sentence segmentation is the **authoritative linguistic segmentation** for all subsequent extractive summarization and feature extraction phases.

---

## 5. The Dual Representation Architecture

Extractive summarization demands a strict architectural separation:

$$\text{Processed Representation (Tokens, Lemmas, POS, NER)} \longrightarrow \text{Scoring \& Feature Extraction}$$
$$\text{Original Representation (Verbatim Sentences)} \longrightarrow \text{Final Extractive Summary Output}$$

Every record in `dataset/processed/` retains both representations:
```json
{
  "id": "f001ec5c4704938247d27a44948eebb37ae98d01",
  "original_article": "(CNN)The Palestinian Authority...",
  "cleaned_article": "(CNN)The Palestinian Authority...",
  "original_sentences": [
    "(CNN)",
    "The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories."
  ],
  "processed_sentences": [
    {
      "sentence_index": 0,
      "original_text": "(CNN)",
      "tokens": ["(", "CNN", ")"],
      "normalized_tokens": ["(", "cnn", ")"],
      "content_tokens": ["cnn"],
      "lemmas": ["(", "cnn", ")"],
      "pos": ["PUNCT", "PROPN", "PUNCT"],
      "tags": ["-LRB-", "NNP", "-RRB-"],
      "token_details": [...],
      "entities": [{"text": "CNN", "label": "ORG", "start_char": 1, "end_char": 4}]
    },
    {
      "sentence_index": 1,
      "original_text": "The Palestinian Authority officially became the 123rd member...",
      "tokens": ["The", "Palestinian", "Authority", "officially", "became", "..."],
      "normalized_tokens": ["the", "palestinian", "authority", "officially", "became", "..."],
      "content_tokens": ["palestinian", "authority", "officially", "123rd", "member", "international", "criminal", "court"],
      "lemmas": ["the", "palestinian", "authority", "officially", "become", "..."],
      "pos": ["DET", "PROPN", "PROPN", "ADV", "VERB", "..."],
      "tags": ["DT", "NNP", "NNP", "RB", "VBD", "..."],
      "token_details": [...],
      "entities": [
        {"text": "The Palestinian Authority", "label": "ORG", "start_char": 0, "end_char": 25},
        {"text": "123rd", "label": "ORDINAL", "start_char": 48, "end_char": 53},
        {"text": "the International Criminal Court", "label": "ORG", "start_char": 64, "end_char": 96},
        {"text": "Wednesday", "label": "DATE", "start_char": 100, "end_char": 109}
      ]
    }
  ],
  "highlights": "Membership gives the ICC jurisdiction over alleged crimes committed..."
}
```

---

## 6. Execution and Verification

### Run the Pipeline via CLI
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run batch preprocessing on 1,000 CNN/DailyMail records
python -m preprocessing.pipeline --input dataset/raw/cnn_dailymail_test_1000.jsonl --output dataset/processed/cnn_dailymail_test_1000_processed.jsonl
```

### Run Unit Tests
```powershell
python -m pytest -v tests/
```
