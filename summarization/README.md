# Frequency-Based Extractive Summarization (Phase 3 Baseline)

## 1. What Extractive Summarization Means
In text summarization, there are two fundamental paradigms:
- **Abstractive Summarization**: Generates novel phrases, sentences, and paraphrases that were not explicitly present in the source text (conventionally requiring large sequence-to-sequence models or LLMs).
- **Extractive Summarization**: Identifies and extracts the most salient, representative sentences verbatim directly from the original document, preserving exact wording, factual veracity, and grammatical integrity.

This project focuses strictly on **classical extractive summarization**, where algorithmic scoring models (Frequency, TF-IDF, TextRank) rank candidate sentences and output the top-scoring original sentences without generative hallucination.

---

## 2. What Frequency-Based Summarization Means
The frequency-based extractive summarizer is the foundational baseline algorithm in modern text summarization research (Luhn, 1958; Edmundson, 1969). It operates on the core linguistic hypothesis that:
> *Words that appear frequently throughout an article reflect the central topics and core themes of the reporting.*

Sentences containing high concentrations of these frequent topical words are statistically more likely to convey central narrative information than sentences containing solely peripheral details.

---

## 3. Algorithmic Pipeline & Mathematical Formulations

```
Raw Article
   ↓
Phase 2 Linguistic Preprocessing (Cleaning, Normalization, Segmentation, Tokenization, POS, NER)
   ↓
Content Token Extraction (Nouns, Lexical Verbs, Adjectives; Stop Words Filtered)
   ↓
Word Frequency Counting: freq(w)
   ↓
Frequency Normalization: NF(w) = freq(w) / max(freq)
   ↓
Sentence Scoring: Score(S) = sum(NF(w)) / |ContentWords(S)|
   ↓
Sentence Ranking (Deterministic Tie-Breaking)
   ↓
Top-K Sentence Selection
   ↓
Chronological Order Restoration (Sort by Sentence Index Ascending)
   ↓
Final Extractive Summary (Verbatim Source Sentences)
```

### 3.1 Content Word Representation
Raw frequency must not count punctuation symbols or grammatical function words (*the*, *is*, *at*, *which*). Using Phase 2 preprocessing, we isolate meaningful content words:
$$C(A) = \{ w \in \text{Tokens}(A) \mid \neg \text{is\_stop}(w) \wedge \neg \text{is\_punct}(w) \}$$

### 3.2 Raw Word Frequency
$$\text{freq}(w) = \sum_{t \in C(A)} \mathbb{I}(t = w)$$

### 3.3 Frequency Normalization
To prevent article length from distorting scale and to keep weights interpretable within $[0, 1]$:
$$\text{NF}(w) = \frac{\text{freq}(w)}{\max_{w' \in C(A)} \text{freq}(w')}$$
The most frequent content word receives an importance score of $1.0$.

### 3.4 Sentence Salience Scoring
For each candidate sentence $S$, its score is calculated as the average normalized frequency of its content words:
$$\text{Score}(S) = \frac{\sum_{w \in C(S)} \text{NF}(w)}{\max(|C(S)|, 1)}$$
- If a sentence contains zero content words (e.g. isolated punctuation or conversational filler), $\text{Score}(S) = 0.0$.
- Normalizing by sentence content length $|C(S)|$ ensures long sentences with many words do not unfairly dominate shorter, information-dense sentences.

### 3.5 Deterministic Tie-Breaking
When two sentences achieve identical salience scores, ties are broken strictly by chronological position:
$$\text{Rank}(S_i) > \text{Rank}(S_j) \iff (\text{Score}(S_i) > \text{Score}(S_j)) \lor (\text{Score}(S_i) = \text{Score}(S_j) \wedge i < j)$$

### 3.6 Chronological Order Restoration
Extracting top-scoring sentences according to raw score can disrupt the narrative timeline. The top $K$ selected sentences are sorted by their original document index:
$$\text{Order}(S_i) < \text{Order}(S_j) \iff i < j$$
This preserves readability and chronological coherence in the final summary.

---

## 4. Known Academic Limitations of the Frequency Baseline

While frequency-based summarization provides a transparent, explainable benchmark, it possesses key algorithmic limitations that motivate subsequent phases:

1. **Lack of Corpus Specificity (No IDF)**: High-frequency terms within an article often include generic journalistic verbs (*said*, *told*, *reported*, *year*) that appear frequently across *all* news articles. The frequency baseline cannot downweight these ubiquitous words. (*Resolved in Phase 4 using TF-IDF*).
2. **Ignorance of Cross-Sentence Similarity**: The algorithm evaluates each sentence in isolation. If two sentences repeat the exact same high-frequency terms, both may receive high scores, leading to informational redundancy. (*Resolved in Phase 5 using TextRank graph centrality*).
3. **No Discourse or Position Modeling**: Frequency alone does not account for the journalistic "inverted pyramid" structure, where the opening paragraphs traditionally carry the greatest factual weight.
4. **Keyword Bias**: Sentences with repeated occurrences of a single dominant keyword can achieve inflated salience over diverse, multi-concept sentences.

---

## 5. Usage & CLI Examples

### Summarize a Direct Text String
```powershell
python -m summarization.frequency --article "The United Nations announced a climate accord today in Geneva. Global leaders pledged carbon reductions. Financial institutions committed investments." --num-sentences 2
```

### Summarize an Article by ID from Processed Dataset
```powershell
python -m summarization.frequency --dataset dataset/processed/cnn_dailymail_test_1000_processed.jsonl --id f001ec5c4704938247d27a44948eebb37ae98d01 --num-sentences 3
```

### Run Batch Experiment Over 1,000 Articles
```powershell
python -m summarization.frequency --batch --input dataset/processed/cnn_dailymail_test_1000_processed.jsonl --output dataset/processed/frequency_summaries_1000.jsonl --num-sentences 3
```
