# Phase 4 Baseline Analysis: TF-IDF Extractive Summarization

## 1. Objective

Phase 4 implements a classical **Term Frequency-Inverse Document Frequency (TF-IDF)** extractive summarization algorithm from first principles. While the Phase 3 frequency baseline scored sentences based on the overall prevalence of words throughout the document, TF-IDF measures term **salience and distinctiveness** by balancing local term frequency within a sentence against its sentence-level document frequency across the article.

The primary objectives of this phase are:
- Implement fully explainable, mathematically explicit TF-IDF formulations suitable for academic scrutiny and viva defense.
- Maintain **dual representation**: compute TF, DF, IDF, and sentence scores strictly on processed linguistic content tokens while extracting verbatim, unmodified sentences from the original source text.
- Establish an intra-article TF-IDF baseline over the 1,000 CNN/DailyMail test set articles without relying on external generative models, LLMs, or black-box libraries.

---

## 2. Algorithm Description

The TF-IDF summarizer operates through an 8-stage pipeline:

```
Preprocessed Article Record
           ↓
1. Extract content tokens per sentence: C(S_i)
           ↓
2. Compute Document Frequency (DF) across sentences: DF(t)
           ↓
3. Compute Inverse Document Frequency (IDF): IDF(t) = ln(N / DF(t))
           ↓
4. Compute Sentence Term Frequency (TF): TF(t, S_i) = count(t ∈ S_i) / |C(S_i)|
           ↓
5. Compute TF-IDF weights: TFIDF(t, S_i) = TF(t, S_i) × IDF(t)
           ↓
6. Compute Sentence Score: Score(S_i) = sum(TFIDF(t, S_i)) / |C(S_i)|
           ↓
7. Rank sentences descending by score (deterministic tie-breaking: lower index wins)
           ↓
8. Select Top-K sentences, sort ascending by original sentence index, and return original text
```

---

## 3. Mathematical Formulations

### 3.1 Term Frequency (TF)
For term $t$ in sentence $d$:
$$\text{TF}(t, d) = \frac{\text{count}(t \in d)}{|C(d)|}$$
where $C(d)$ denotes the list of content tokens (non-stopwords, lemmatized, alphabetic) in sentence $d$. If $|C(d)| = 0$, then $\text{TF}(t, d) = 0$.

### 3.2 Document Frequency (DF)
For an article composed of $N$ segmented sentences $\{S_1, S_2, \dots, S_N\}$:
$$\text{DF}(t) = \sum_{i=1}^N \mathbb{I}(t \in S_i)$$
Each term is counted at most once per sentence.

### 3.3 Inverse Document Frequency (IDF)
$$\text{IDF}(t) = \ln\left(\frac{N}{\text{DF}(t)}\right)$$
- If a term appears in every sentence ($\text{DF}(t) = N$), then $\text{IDF}(t) = \ln(1) = 0.0$.
- If a term appears in only 1 sentence ($\text{DF}(t) = 1$), then $\text{IDF}(t) = \ln(N)$ (maximum distinctiveness).
- An optional smoothed variant is also implemented: $\text{IDF}_{\text{smooth}}(t) = \ln\left(1 + \frac{N}{\text{DF}(t)}\right)$. The standard formulation without smoothing is used for the default baseline.

### 3.4 TF-IDF Weight
$$\text{TFIDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

### 3.5 Sentence Importance Score
$$\text{SentenceScore}(S) = \frac{\sum_{t \in C(S)} \text{TFIDF}(t, S)}{|C(S)|}$$
- Normalization by $|C(S)|$ ensures that sentence length does not artificially inflate the score, producing the average TF-IDF importance of the sentence's content terms.
- If $|C(S)| = 0$, $\text{SentenceScore}(S) = 0.0$.

### 3.6 Deterministic Tie-Breaking
When two sentences achieve identical scores:
$$\text{Rank}(S_a) < \text{Rank}(S_b) \iff \text{SentenceIndex}(S_a) < \text{SentenceIndex}(S_b)$$

---

## 4. Configuration

| Parameter | Value | Description |
|---|---|---|
| **Dataset** | CNN/DailyMail v3.0.0 | Academic news summarization benchmark |
| **Split** | Test set | Random seed 42 |
| **Sample Size** | 1,000 articles | Deterministic sample established in Phase 1 |
| **Summary Length** | 3 sentences | Standard initial baseline configuration |
| **IDF Scope** | Intra-article ($N = \text{sentences in article}$) | Computes sentence salience relative to article context |
| **Smoothing** | Disabled (`smooth=False`) | Exact formula $\ln(N / \text{DF}(t))$ |

---

## 5. Five Real CNN/DailyMail Examples

### Example 1
- **Article ID**: `f001ec5c4704938247d27a44948eebb37ae98d01`
- **Original sentence count**: 28
- **Requested summary length**: 3
- **Selected sentence indices**: `[0, 12, 19]`
- **Selected sentence scores**: `[2.6391, 0.5786, 0.4619]`
- **Generated summary**:
  > (CNN) These are substantive commitments, which cannot be taken lightly," she said. It urged the warring sides to resolve their differences through direct negotiations.
- **Reference highlights**:
  > Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June .\nIsrael and the United States opposed the move, which could open the door to war crimes investigations against Israelis .

### Example 2
- **Article ID**: `230c522854991d053fe98a718b1defa077a8efef`
- **Original sentence count**: 19
- **Requested summary length**: 3
- **Selected sentence indices**: `[0, 10, 13]`
- **Selected sentence scores**: `[0.5889, 0.5449, 0.441]`
- **Generated summary**:
  > (CNN)Never mind cats having nine lives. She's also created a Facebook page to keep supporters updated. I agreed to foster her until she finally found a loving home."
- **Reference highlights**:
  > Theia, a bully breed mix, was apparently hit by a car, whacked with a hammer and buried in a field .\n"She's a true miracle dog and she deserves a good life," says Sara Mellado, who is looking for a home for Theia .

### Example 3
- **Article ID**: `4495ba8f3a340d97a9df1476f8a35502bcce1f69`
- **Original sentence count**: 38
- **Requested summary length**: 3
- **Selected sentence indices**: `[8, 15, 27]`
- **Selected sentence scores**: `[0.9193, 0.8661, 0.9094]`
- **Generated summary**:
  > But there are some facts about Zarif that are less well-known. "The man who was perceived to be denying it is now gone. So he is 54, 55 or maybe even 56.
- **Reference highlights**:
  > Mohammad Javad Zarif has spent more time with John Kerry than any other foreign minister .\nHe once participated in a takeover of the Iranian Consulate in San Francisco .\nThe Iranian foreign minister tweets in English .

### Example 4
- **Article ID**: `a38e72fed88684ec8d60dd5856282e999dc8c0ca`
- **Original sentence count**: 11
- **Requested summary length**: 3
- **Selected sentence indices**: `[2, 6, 9]`
- **Selected sentence scores**: `[1.1989, 0.3611, 0.4241]`
- **Generated summary**:
  > The others have already gone home. As of Monday, that health care worker is in fair condition. Almost all the deaths have been in Guinea, Liberia and Sierra Leone.
- **Reference highlights**:
  > 17 Americans were exposed to the Ebola virus while in Sierra Leone in March .\nAnother person was diagnosed with the disease and taken to hospital in Maryland .\nNational Institutes of Health says the patient is in fair condition after weeks of treatment .

### Example 5
- **Article ID**: `c27cf1b136cc270023de959e7ab24638021bc43f`
- **Original sentence count**: 22
- **Requested summary length**: 3
- **Selected sentence indices**: `[10, 11, 15]`
- **Selected sentence scores**: `[0.751, 1.0257, 1.1432]`
- **Generated summary**:
  > This is no Duke we want. This is not the Duke we're here to experience. Two students were expelled.
- **Reference highlights**:
  > Student is no longer on Duke University campus and will face disciplinary review .\nSchool officials identified student during investigation and the person admitted to hanging the noose, Duke says .\nThe noose, made of rope, was discovered on campus about 2 a.m.

---

## 6. Observations

1. **Preference for Informational Distinctiveness**:
   Unlike Frequency summarization which favored central, highly repeated keywords across the whole article (such as `"said"`, `"court"`, `"dog"`), TF-IDF penalizes words that appear everywhere ($\text{IDF} \to 0$) and promotes sentences with distinctive, specific nouns, actions, and names.

2. **The Short-Sentence / Dateline Phenomenon**:
   In articles where a single dateline or short sentence contains a word that occurs nowhere else in the article (e.g. `(CNN)` where `"cnn"` has $\text{DF}=1$ out of 28 sentences, giving $\text{IDF} = \ln(28) \approx 3.33$), that single term dominates the average TF-IDF calculation for the sentence. Consequently, short sentences with unique terms can receive high average scores.

3. **Compression and Word Length**:
   The mean summary word count for TF-IDF 3-sentence extraction is **21.65 words**, compared to **43.51 words** in Frequency summarization. This occurs because length-normalization allows shorter, highly focused sentences to compete with longer compound sentences.

---

## 7. Limitations of TF-IDF Summarization

1. **No Semantic or Contextual Understanding**:
   TF-IDF treats terms as independent discrete tokens. Synonyms (e.g., *"doctor"* and *"physician"*) are treated as completely unrelated, and homonyms are conflated.

2. **Lack of Inter-Sentence Discourse Modeling**:
   TF-IDF evaluates each sentence in isolation against the document vocabulary. It does not model whether one sentence logically follows another or whether rhetorical transitions are preserved.

3. **Coreference Blindness**:
   Sentences containing unresolved pronouns (*"she"*, *"it"*, *"they"*) can be extracted out of context, leading to dangling references in the final summary.

4. **Vulnerability to Outlier / Rare Tokens**:
   Because $\text{IDF}(t) = \ln(N / \text{DF}(t))$ is maximized when $\text{DF}(t) = 1$, typos, isolated jargon, or dateline markers can receive disproportionate importance.

5. **Extractive Constraint**:
   TF-IDF cannot synthesize, rephrase, or merge ideas into compact abstractive statements.

> [!NOTE]
> Formal quantitative comparison between Frequency-based and TF-IDF-based summarization (e.g., ROUGE-1, ROUGE-2, ROUGE-L) will be performed rigorously in **Phase 6**. No claim is made at this stage that TF-IDF outperforms Frequency summarization.
