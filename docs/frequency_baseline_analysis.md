# Phase 3: Frequency-Based Summarization Baseline Report

## 1. Executive Summary
This report presents the empirical results of the **Frequency-Based Extractive Summarizer**, serving as the foundational baseline for the summarization system. Sentences are ranked and selected strictly using normalized term frequency from Phase 2 content tokens, with original sentence wording and chronological order strictly preserved.

- **Algorithm**: Classical Word-Frequency Scoring
- **Input Corpus**: `D:\nlp project\dataset\processed\cnn_dailymail_test_1000_processed.jsonl`
- **Output Corpus**: `D:\nlp project\dataset\processed\frequency_summaries_1000.jsonl`
- **Articles Processed**: 1000
- **Target Summary Length**: 3 sentences

---

## 2. Mathematical Formulation

### Word Frequency
$$\text{freq}(w) = \text{total occurrences of content word } w \text{ in article}$$

### Normalized Frequency
$$\text{NF}(w) = \frac{\text{freq}(w)}{\max_{w'} \text{freq}(w')}$$
where $\max_{w'} \text{freq}(w')$ is the maximum frequency of any content word in the article.

### Sentence Salience Score
$$\text{Score}(S) = \frac{\sum_{w \in C(S)} \text{NF}(w)}{|C(S)|}$$
where $C(S)$ represents the list of meaningful content words in sentence $S$. If $|C(S)| = 0$, $\text{Score}(S) = 0.0$.

### Selection & Reconstruction
1. Sentences are ranked descending by $\text{Score}(S)$.
2. Ties are broken deterministically in favor of the earlier sentence index.
3. The top $K$ sentences are selected and then reordered by their original sentence index ascending:
$$\text{Order}(S_i) < \text{Order}(S_j) \iff \text{Index}(S_i) < \text{Index}(S_j)$$

---

## 3. Quantitative Summary Statistics ($N = 1,000$ Articles)

| Metric | Mean | Median | Min | Max |
|:---|:---:|:---:|:---:|:---:|
| **Original Sentences per Article** | 33.47 | 30.0 | 3 | 112 |
| **Selected Sentences per Summary** | 3.0 | 3.0 | 3 | 3 |
| **Summary Word Count** | 43.51 | 43.0 | 3 | 100 |

*Note: Phase 1 showed reference highlights average approximately 34.47 words (2.63 sentences). The 3-sentence frequency summaries average 43.51 words, providing a strong baseline for future ROUGE evaluation in Phase 6.*

---

## 4. Real CNN/DailyMail Summary Examples

### Example 1 (ID: `f001ec5c4704938247d27a44948eebb37ae98d01`)
- **Original Sentences**: 28
- **Requested Sentences**: 3 (Selected: 3)
- **Selected Indices & Scores**: [(1, 0.3471), (17, 0.3857), (18, 0.4615)]

**Generated Frequency Summary**:
> The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The United States also said it "strongly" disagreed with the court's decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC," the State Department said in a statement.

**Ground-Truth Reference Highlights**:
> Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June .
Israel and the United States opposed the move, which could open the door to war crimes investigations against Israelis .

---

### Example 2 (ID: `230c522854991d053fe98a718b1defa077a8efef`)
- **Original Sentences**: 19
- **Requested Sentences**: 3 (Selected: 3)
- **Selected Indices & Scores**: [(6, 0.3333), (7, 0.375), (9, 0.4091)]

**Generated Frequency Summary**:
> "She's a true miracle dog and she deserves a good life." Theia is only one year old but the dog's brush with death did not leave her unscathed. The veterinary hospital's Good Samaritan Fund committee awarded some money to help pay for the dog's treatment, but Mellado has set up a fundraising page to help meet the remaining cost of the dog's care.

**Ground-Truth Reference Highlights**:
> Theia, a bully breed mix, was apparently hit by a car, whacked with a hammer and buried in a field .
"She's a true miracle dog and she deserves a good life," says Sara Mellado, who is looking for a home for Theia .

---

### Example 3 (ID: `4495ba8f3a340d97a9df1476f8a35502bcce1f69`)
- **Original Sentences**: 38
- **Requested Sentences**: 3 (Selected: 3)
- **Selected Indices & Scores**: [(8, 0.381), (14, 0.4107), (23, 0.375)]

**Generated Frequency Summary**:
> But there are some facts about Zarif that are less well-known. "Iran never denied it," Zarif tweeted back. The website of the Iranian Foreign Ministry, which Zarif runs, cannot even agree with itself on when he was born.

**Ground-Truth Reference Highlights**:
> Mohammad Javad Zarif has spent more time with John Kerry than any other foreign minister .
He once participated in a takeover of the Iranian Consulate in San Francisco .
The Iranian foreign minister tweets in English .

---

### Example 4 (ID: `a38e72fed88684ec8d60dd5856282e999dc8c0ca`)
- **Original Sentences**: 11
- **Requested Sentences**: 3 (Selected: 3)
- **Selected Indices & Scores**: [(0, 0.3889), (3, 0.4375), (5, 0.375)]

**Generated Frequency Summary**:
> (CNN)Five Americans who were monitored for three weeks at an Omaha, Nebraska, hospital after being exposed to Ebola in West Africa have been released, a Nebraska Medicine spokesman said in an email Wednesday. They were exposed to Ebola in Sierra Leone in March, but none developed the deadly virus. They all had contact with a colleague who was diagnosed with the disease and is being treated at the National Institutes of Health in Bethesda, Maryland.

**Ground-Truth Reference Highlights**:
> 17 Americans were exposed to the Ebola virus while in Sierra Leone in March .
Another person was diagnosed with the disease and taken to hospital in Maryland .
National Institutes of Health says the patient is in fair condition after weeks of treatment .

---

### Example 5 (ID: `c27cf1b136cc270023de959e7ab24638021bc43f`)
- **Original Sentences**: 22
- **Requested Sentences**: 3 (Selected: 3)
- **Selected Indices & Scores**: [(0, 0.4), (10, 0.6667), (11, 0.5556)]

**Generated Frequency Summary**:
> (CNN)A Duke student has admitted to hanging a noose made of rope from a tree near a student union, university officials said Thursday. This is no Duke we want. This is not the Duke we're here to experience.

**Ground-Truth Reference Highlights**:
> Student is no longer on Duke University campus and will face disciplinary review .
School officials identified student during investigation and the person admitted to hanging the noose, Duke says .
The noose, made of rope, was discovered on campus about 2 a.m.

---


## 5. Architectural Properties & Known Baseline Limitations
1. **Extractive Fidelity**: The summary exclusively contains verbatim source sentences. Lemmatized or stop-word-stripped representations were used solely for mathematical scoring.
2. **Chronological Coherence**: Re-sorting selected top-K sentences by their original appearance order preserves discourse flow.
3. **Inability to Model Term Specificity**: Raw frequency does not downweight common terms that appear across all news articles (e.g. *said*, *told*, *year*). This will be resolved in Phase 4 using **TF-IDF**.
4. **Lack of Sentence Inter-Relationships**: Sentence scoring considers terms independently rather than measuring inter-sentence cross-similarity. This will be addressed in Phase 5 using **TextRank**.
