# Phase 8: New-Word Detection Analysis Report

## 1. Executive Summary

This report documents the implementation, empirical batch execution, and statistical analysis of **Phase 8: New-Word Detection** for the academic project:
**“NLP-Based News Article Summarization and Vocabulary Learning System.”**

Phase 8 bridges extractive summarization and vocabulary acquisition by evaluating the vocabulary items extracted in Phase 7 against a user's known-vocabulary list. Using classical, deterministic canonical normalization and morphological lemma matching, the detector classifies every extracted item into either `known_words` or `new_words`.

---

## 2. Research Objective and Methodology

### 2.1 The Concept of Relative Novelty
In classical educational NLP, determining whether a word is "difficult" or "new" must not rely on subjective developer impressions or opaque black-box classifiers. Novelty is an inherently **relative property**:
> A word is **new** if and only if it is absent from the specific learner's current repository of known vocabulary under canonical linguistic matching.

### 2.2 Matching and Normalization Policy
The system applies a deterministic canonical matching policy (`canonical_lemma_and_normalized_matching`):
1. **Canonical Normalization**: All terms are lowercased and stripped of boundary whitespace.
2. **Morphological Lemma Matching**:
   $$\text{is\_known}(t) \iff \text{normalized}(t) \in V_{\text{known}} \;\lor\; \text{lemma}(t) \in V_{\text{known}} \;\lor\; \text{surface}(t) \in V_{\text{known}}$$
   This allows inflected surface forms (*announced*, *creating*, *cars*) to match known base lemmas (*announce*, *create*, *car*).
3. **Multi-Word Phrase Matching**: Named entities and compound terms (e.g., *United Nations*) match exact multi-word known entries.
4. **Entity Awareness**: Entity labels (`PERSON`, `ORG`, `GPE`, etc.) are preserved and matched against entity-specific entries when provided.
5. **Exact Accounting Guarantee**:
   $$\text{total\_vocabulary\_count} = \text{known\_word\_count} + \text{new\_word\_count}$$
   Every input candidate is categorized exactly once, with zero loss of Phase 7 metadata or rank scoring.

---

## 3. Experimental Setup & Synthetic Benchmark Baseline

To perform a reproducible, quantitative evaluation on the 1,000-article corpus, a **reproducible synthetic known-vocabulary baseline** was constructed:
- **Baseline File**: `dataset/processed/synthetic_known_vocabulary.json`
- **Specification**: 603 foundational core English words and common entities (e.g., *say, tell, court, police, year, today, government, city, help, good, new, cnn, united states*).
- **Declaration of Synthetic Nature**: This vocabulary list serves as a standardized academic benchmark to test throughput, boundary conditions, and distribution metrics. It does **not** represent a real individual learner's psychological vocabulary profile.

---

## 4. Empirical Batch Results (1,000 CNN/DailyMail Articles)

The batch detector processed the full Phase 7 output dataset:
- **Input Dataset**: `dataset/processed/vocabulary_extractions_1000.jsonl` (1,000 articles, 30,000 total candidate items)
- **Known Vocabulary Baseline**: 603 canonical entries
- **Output Dataset**: `dataset/processed/new_word_detections_1000.jsonl`

### 4.1 Macro Execution Metrics

| Metric | Measured Value |
| :--- | :---: |
| **Total Articles Processed** | 1,000 |
| **Total Vocabulary Items Evaluated** | 30,000 |
| **Known Words Identified** | 6,496 (21.65%) |
| **New Words Identified (Learning Targets)** | 23,504 (78.35%) |
| **Reconciliation Consistency Rate** | 100.0% (30,000 / 30,000) |
| **Batch Runtime** | 0.42 seconds |
| **Throughput** | 2,388.14 articles / second |

### 4.2 Per-Article Distribution Statistics

| Distribution Statistic | Known Words per Article | New Words per Article |
| :--- | :---: | :---: |
| **Mean** | 6.50 | 23.50 |
| **Median** | 6.00 | 24.00 |
| **Minimum** | 1.00 | 14.00 |
| **Maximum** | 16.00 | 29.00 |
| **Standard Deviation** | 2.45 | 2.45 |

*Key Takeaway*: Under the 603-word foundational baseline, each 30-word article summary yielded an average of **23.5 unfamiliar target vocabulary words** and **6.5 familiar anchor words**.

---

## 5. Illustrative Case Studies from Real Articles

### Case Study 1: Article `f001ec5c4704938247d27a44948eebb37ae98d01`
- **Topic**: International Criminal Court (ICC) Palestine proceedings.
- **Total Vocabulary Items**: 30 | **Known**: 7 (23.3%) | **New**: 23 (76.7%)

| Classification | Term | POS | Lemma | Type / Entity | Score | Why Classified |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **KNOWN** | `said` | `VERB` | `say` | `-` | 0.5429 | Lemma `say` is in known vocabulary |
| **KNOWN** | `court` | `NOUN` | `court` | `-` | 0.3545 | Direct match for known term `court` |
| **KNOWN** | `today` | `PROPN` | `today` | `DATE` | 0.3519 | Direct match for known term `today` |
| **KNOWN** | `CNN` | `PROPN` | `cnn` | `ORG` | 0.3013 | Direct match for known term `cnn` |
| **NEW** | `ICC` | `PROPN` | `icc` | `ORG` | 0.5545 | Absent from known vocabulary |
| **NEW** | `Palestine` | `PROPN` | `palestine` | `GPE` | 0.5039 | Absent from known vocabulary |
| **NEW** | `Palestinian`| `PROPN` | `palestinian`| `NORP`| 0.4026 | Absent from known vocabulary |
| **NEW** | `Israel` | `PROPN` | `israel` | `GPE` | 0.3519 | Absent from known vocabulary |

---

### Case Study 2: Article `230c522854991d053fe98a718b1defa077a8efef`
- **Topic**: Animal rescue narrative featuring Mellado and dog Theia.
- **Total Vocabulary Items**: 30 | **Known**: 7 (23.3%) | **New**: 23 (76.7%)

| Classification | Term | POS | Lemma | Type / Entity | Score | Why Classified |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **KNOWN** | `help` | `VERB` | `help` | `-` | 0.3298 | Direct match for known term `help` |
| **KNOWN** | `Washington` | `PROPN` | `washington` | `GPE` | 0.2877 | Direct match for known entity `washington` |
| **KNOWN** | `found` | `VERB` | `find` | `-` | 0.2807 | Lemma `find` is in known vocabulary |
| **NEW** | `Mellado` | `PROPN` | `mellado` | `PERSON` | 0.5509 | Absent from known vocabulary |
| **NEW** | `dog` | `NOUN` | `dog` | `-` | 0.5053 | Absent from known vocabulary |
| **NEW** | `Theia` | `PROPN` | `theia` | `ORG` | 0.4632 | Absent from known vocabulary |

---

## 6. Limitations

1. **Synthetic Vocabulary Boundary**: The empirical statistics reflect comparison against a 603-word synthetic list. Actual user performance will vary based on user-provided known lists.
2. **Context-Free Lexical Matching**: Words with multiple senses (polysemy) match whenever the surface/lemma matches, irrespective of contextual semantics.
3. **Absence of User State**: Phase 8 does not persist user history or track words learned over time; this functionality is deferred to Phase 9.

---

## 7. Verification & Reproducibility Commands

The experiment and tests can be reproduced with:
```powershell
# Run the 24-test Phase 8 test suite
pytest tests/test_new_word_detection.py -v

# Run batch detection over the 1,000 Phase 7 records
python -m vocabulary.new_word_detector --input dataset/processed/vocabulary_extractions_1000.jsonl --known-vocab dataset/processed/synthetic_known_vocabulary.json --output dataset/processed/new_word_detections_1000.jsonl --batch
```
