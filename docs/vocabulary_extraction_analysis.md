# Phase 7: Vocabulary Extraction Analysis Report

## 1. Executive Summary

This report documents the implementation, empirical execution, and quantitative analysis of **Phase 7: Classical Vocabulary Extraction** for the academic project:
**“NLP-Based News Article Summarization and Vocabulary Learning System.”**

Building upon the preprocessed representations established in Phase 2 and the extractive summarization baselines in Phases 3–6, Phase 7 extracts salient lexical content words and recognized named entities from articles. It associates each extracted term with morphological, syntactic, and positional metadata, accompanied by a deterministic, explainable heuristic importance score.

---

## 2. Motivation and Academic Context

News articles are rich in domain-specific terminology, key topical entities, and lexical descriptors. For a vocabulary learning system, extracting every token indiscriminately is counter-productive:
1. **Functional Noise**: Stop words, determiners, auxiliaries, and punctuation dominate frequency counts but have minimal lexical learning value.
2. **Contextual Salience**: Readers benefit most from terms central to the article's core narrative.
3. **Linguistic Transparency**: Presenting canonical lemmas, parts of speech, and sentence locations enables structured vocabulary reviews without black-box language models.

Phase 7 establishes this feature extraction pipeline strictly via **classical NLP techniques**, ensuring that every extracted item and rank score can be explained from first principles.

---

## 3. Extraction Methodology and Architectural Design

### 3.1 Pipeline Flow
The extraction process executes in five deterministic stages:

1. **Dual Representation Traversal**:
   Reads the `processed_sentences` array from the Phase 2 dataset or generates it via `PreprocessingPipeline` for raw text.
2. **Content-Word Filtering**:
   - Excludes tokens where `is_stop == True` or `is_punct == True`.
   - Restricts primary lexical extraction to Universal Dependencies content categories: `NOUN`, `PROPN`, `VERB`, `ADJ`, and `ADV`.
   - Filters out non-alphanumeric noise and tokens shorter than `min_term_length = 2`.
3. **Named Entity Processing**:
   - Spans identified by the Named Entity Recognition (NER) model are categorized by their exact label (`PERSON`, `ORG`, `GPE`, `DATE`, `NORP`, etc.).
   - Multiword and single-word entities are preserved in their original casing and aggregated distinctly from ordinary vocabulary.
4. **Morphological Aggregation**:
   - Lexical content words are aggregated by canonical morphological lemma (`lemma`, `pos`), preserving the most frequent original surface form as the representative display term.
   - Named entities are aggregated by `(normalized_text, entity_label)`.
5. **Deterministic Ranking and Tie-Breaking**:
   Sorts candidates by:
   $$\text{Importance Score} \downarrow \;\longrightarrow\; \text{Token Frequency} \downarrow \;\longrightarrow\; \text{First Sentence Index} \uparrow \;\longrightarrow\; \text{First Token Offset} \uparrow \;\longrightarrow\; \text{Normalized Form} \uparrow$$

---

## 4. Importance Scoring Formulation

To balance lexical frequency with topical distribution across the narrative, the extractor computes an explainable heuristic score:

$$\text{score}(t) = \left(\alpha \cdot \frac{TF(t)}{\max_{w} TF(w)} + \beta \cdot \frac{SF(t)}{N_{\text{sentences}}} + \gamma \cdot \text{EntityBonus}(t)\right) \times \text{POSMultiplier}(\text{pos}(t))$$

### Parameter Rationale:
- **Normalized Term Frequency ($\frac{TF}{\max TF}$, $\alpha = 0.4$)**: Captures the term's raw repetition within the text.
- **Sentence Coverage Ratio ($\frac{SF}{N_{\text{sentences}}}$, $\beta = 0.4$)**: Rewards terms that recur across multiple independent sentences, penalizing bursty words confined to a single sentence.
- **Named Entity Bonus ($\gamma = 0.2$)**: Provides a modest boost to recognized people, organizations, and geopolitical entities that anchor news reporting.
- **POS Multipliers**: `NOUN`: 1.0, `PROPN`: 1.0, `VERB`: 1.0, `ADJ`: 1.0, `ADV`: 0.8.

> [!NOTE]
> This score is explicitly a heuristic ranking index for topical salience within an individual article. It is **not** a probability, cross-corpus IDF score, or user-specific difficulty metric.

---

## 5. Empirical Batch Results (1,000 CNN/DailyMail Articles)

The batch extraction was executed across the full test corpus of 1,000 processed articles:
- **Input Corpus**: `dataset/processed/cnn_dailymail_test_1000_processed.jsonl`
- **Output Dataset**: `dataset/processed/vocabulary_extractions_1000.jsonl`
- **Execution Mode**: Streaming JSONL line-by-line (constant memory footprint)

### 5.1 Macro Corpus Statistics

| Metric | Measured Value |
| :--- | :---: |
| **Total Articles Processed** | 1,000 |
| **Total Vocabulary Items Extracted** | 30,000 |
| **Mean Candidates Evaluated per Article** | 252.33 |
| **Mean Selected Items per Article** | 30.00 |
| **Mean Candidate Salience Score** | 0.3086 |
| **Total Batch Execution Time** | 8.42 seconds |
| **Throughput** | 118.7 articles / second |

### 5.2 Part-of-Speech (POS) Distribution of Selected Items

| POS Category | Description | Count | Percentage |
| :--- | :--- | :---: | :---: |
| **PROPN** | Proper Nouns (including Named Entities) | 25,846 | 86.15% |
| **NOUN** | Common Nouns | 2,723 | 9.08% |
| **VERB** | Lexical Action / Stative Verbs | 1,059 | 3.53% |
| **ADJ** | Descriptive Adjectives | 363 | 1.21% |
| **ADV** | Key Adverbs | 9 | 0.03% |
| **Total** | | **30,000** | **100.0%** |

### 5.3 Named Entity Category Distribution

The top entity categories extracted across the corpus reflect the journalistic nature of CNN/DailyMail news:

| Entity Label | Entity Description | Extracted Count |
| :--- | :--- | :---: |
| **PERSON** | People (politicians, witnesses, athletes, etc.) | 4,522 |
| **ORG** | Companies, agencies, institutions | 4,491 |
| **DATE** | Absolute or relative dates/periods | 4,246 |
| **GPE** | Geopolitical entities (countries, cities, states) | 3,292 |
| **CARDINAL** | Cardinal numerals of topical significance | 2,566 |
| **NORP** | Nationalities, religious or political groups | 1,417 |
| **TIME** | Specific times of day | 572 |
| **ORDINAL** | Ordinal ranks (e.g., *first*, *second*) | 489 |

---

## 6. Representative Case Studies from Real Articles

### Case 1: Article `f001ec5c4704938247d27a44948eebb37ae98d01`
- **Topic**: International Criminal Court (ICC) membership regarding Palestine.
- **Total Candidates Evaluated**: 238
- **Top Extracted Terms**:

| Term | POS | TF | SF | Score | Type / Entity | Selection Reason |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **ICC** | `PROPN` | 7 | 7 | 0.5545 | `ORG` | Named entity (ORG) occurring 7 time(s) across 7 sentence(s) (coverage: 25.0%). |
| **said** | `VERB` | 11 | 10 | 0.5429 | `-` | Salient VERB occurring 11 time(s) across 10 sentence(s) (coverage: 35.7%). |
| **Palestine**| `PROPN` | 6 | 6 | 0.5039 | `GPE` | Named entity (GPE) occurring 6 time(s) across 6 sentence(s) (coverage: 21.4%). |
| **Palestinian**| `PROPN` | 4 | 4 | 0.4026 | `NORP`| Named entity (NORP) occurring 4 time(s) across 4 sentence(s) (coverage: 14.3%). |
| **court** | `NOUN` | 7 | 7 | 0.3545 | `-` | Salient NOUN occurring 7 time(s) across 7 sentence(s) (coverage: 25.0%). |

*Observation*: The extractor accurately captured the central institutional actor (`ICC`), the geopolitical focus (`Palestine`), the community (`Palestinian`), and the judicial setting (`court`).

---

### Case 2: Article `230c522854991d053fe98a718b1defa077a8efef`
- **Topic**: Dog rescue story featuring Mellado and dog Theia.
- **Total Candidates Evaluated**: 184
- **Top Extracted Terms**:

| Term | POS | TF | SF | Score | Type / Entity | Selection Reason |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Mellado** | `PROPN` | 4 | 4 | 0.5509 | `PERSON` | Named entity (PERSON) occurring 4 time(s) across 4 sentence(s) (coverage: 21.1%). |
| **dog** | `NOUN` | 6 | 5 | 0.5053 | `-` | Salient NOUN occurring 6 time(s) across 5 sentence(s) (coverage: 26.3%). |
| **Theia** | `PROPN` | 3 | 3 | 0.4632 | `ORG` | Named entity (ORG) occurring 3 time(s) across 3 sentence(s) (coverage: 15.8%). |
| **help** | `VERB` | 4 | 3 | 0.3298 | `-` | Salient VERB occurring 4 time(s) across 3 sentence(s) (coverage: 15.8%). |

*Observation*: In this narrative piece, common noun `dog` and proper nouns `Mellado` and `Theia` dominate the importance ranking alongside key action verb `help`.

---

## 7. Limitations and Future Directions

1. **Reporting Verbs**: General news attribution verbs such as *said*, *told*, *reported* frequently rank high due to article repetition. While they represent valid verbs in Phase 7, downweighting reporting verbs or integrating corpus-level IDF can improve vocabulary novelty.
2. **Named Entity Resolution**: SpaCy's NER occasionally mislabels non-standard names (e.g., classifying dog name *Theia* as `ORG`).
3. **Decoupled User Profiles**: In accordance with the project roadmap, Phase 7 does not incorporate user-specific vocabulary knowledge. Determining whether an extracted word is "new" or "difficult" is reserved for Phase 8.

---

## 8. Reproducibility Instructions

The empirical results and batch dataset can be reproduced with:
```powershell
# Run the 25-test suite
pytest tests/test_vocabulary_extraction.py -v

# Run batch extraction
python -m vocabulary.extractor --dataset dataset/processed/cnn_dailymail_test_1000_processed.jsonl --output dataset/processed/vocabulary_extractions_1000.jsonl --batch
```
