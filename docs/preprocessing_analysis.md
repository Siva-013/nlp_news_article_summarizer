# Phase 2 Preprocessing Analysis Report (Hardened & Verified)

## 1. Execution Overview
- **Linguistic Engine**: `spaCy (en_core_web_sm 3.8.0)` with hardened infix tokenization rules
- **Input Corpus**: `D:\nlp project\dataset\raw\cnn_dailymail_test_1000.jsonl`
- **Output Corpus**: `D:\nlp project\dataset\processed\cnn_dailymail_test_1000_processed.jsonl`
- **Articles Processed**: 1,000
- **Total Tokens Evaluated**: 739,860

## 2. Preprocessing Metrics Summary
| Metric | Mean | Median | Min | Max |
|:---|:---:|:---:|:---:|:---:|
| **Sentences per Article** | **33.47** | 30.0 | 3 | 112 |
| **Total Tokens per Article** | **739.86** | 671.5 | 94 | 2,486 |
| **Content Tokens per Article** | **325.24** | 291.0 | 39 | 1,357 |
| **Stop Words per Article** | **311.81** | 281.0 | 34 | 910 |
| **Named Entities per Article** | **61.72** | 53.0 | 4 | 499 |

## 3. Normalization and Lexical Distribution
- **Tokens Modified by Lowercasing/Normalization**: 103,881 (14.04%)
- **Stop-Word Ratio**: Approximately 42.1% of article tokens are grammatical function words (stop words).
- **Information Density**: Content tokens constitute approximately 44.0% of total words, providing dense semantic features for future TF-IDF and TextRank ranking.

## 4. Phase 1 vs. Phase 2 Sentence Count Methodology
- **Phase 1 Exploration (Lightweight Method)**: Mean = 31.83 sentences/article.
  *Methodology*: Used a lightweight regex heuristic splitting exclusively on terminal punctuation (`.!?` followed by whitespace). This method misses introductory parenthetical datelines, isolated headers, and multi-clause parenthetical quotes.
- **Phase 2 Pipeline (Authoritative Linguistic Method)**: Mean = 33.47 sentences/article.
  *Methodology*: Uses spaCy's statistical transition-based dependency parser and rule-based sentence boundary detector (`doc.sents`), enhanced with hardened infix tokenization rules. For instance, introductory news datelines such as `(CNN)` are accurately identified as distinct sentence boundaries rather than merged into the initial paragraph sentence.
- **Academic Conclusion**: Phase 2's spaCy sentence segmentation is the authoritative linguistic standard used across all subsequent scoring, ranking, and evaluation phases.

## 5. Tokenizer Hardening & Regression Verification
- **Issue Corrected**: In unconfigured spaCy, closing parentheses and adjacent punctuation bordering words without intervening whitespace (e.g. `(CNN)The` or `word,"hello`) were grouped into single malformed tokens (`CNN)The`).
- **Correction Applied**: Hardened infix regex patterns via `spacy.util.compile_infix_regex` on `nlp.tokenizer.infix_finditer`.
- **Outcome**: Tokens are cleanly and linguistically separated:
  $$\text{"(CNN)The Palestinian Authority"} \longrightarrow \text{['(', 'CNN', ')', 'The', 'Palestinian', 'Authority']}$$
  $$\text{'word,"hello'} \longrightarrow \text{['word', ',', '"', 'hello']}$$
- **Named Entity Alignment**: All entity character spans (`start_char`, `end_char`) are calculated relative to each sentence's original text, ensuring $100\%$ character alignment (`sentence_text[start:end] == entity_text`).

## 6. Dual Representation Guarantee
- Every processed record strictly preserves the verbatim `original_article` and `original_sentences` alongside the linguistically enriched `processed_sentences`.
- Extractive summarization algorithms in Phase 3, 4, and 5 can score sentences using the processed representation (lemmas, content words, POS, entities) while directly returning unmodified original sentences in the final summary.
