# Phase 5 Baseline Analysis: TextRank Extractive Summarization

## 1. Objective

Phase 5 introduces graph-based sentence centrality to extractive text summarization by implementing the classical **TextRank** algorithm from first principles.

While Phase 3 (Frequency) relied on overall term repetition and Phase 4 (TF-IDF) focused on isolated term distinctiveness, TextRank models the entire news article as a **weighted lexical similarity graph**. Sentences serve as nodes, and edges represent lexical overlap between sentences weighted by TF-IDF representations. Sentence salience is derived not from isolated feature counts, but from **graph-based lexical recommendation** via PageRank.

The primary objectives of this phase are:
- Implement classical TextRank from scratch without black-box graph or PageRank libraries (such as `networkx.pagerank`).
- Maintain **dual representation**: construct TF-IDF vectors and similarity matrices strictly on Phase 2 processed content tokens, while preserving and outputting verbatim source sentences.
- Establish a rigorous, reproducible 1,000-article baseline for comparison with Frequency and TF-IDF baselines in Phase 6.

---

## 2. TextRank Concept

TextRank (Mihalcea & Tarau, 2004) adapts Google's PageRank algorithm to Natural Language Processing:

```
Sentence S_1 <=========> Sentence S_2
     ^                        ^
     | \                    / |
     |   \                /   |
     |     \            /     |
     v       \        /       v
Sentence S_3 <=========> Sentence S_4
```

1. **Nodes as Sentences**: Each sentence in an article corresponds to a vertex $V_i$ in graph $G = (V, E)$.
2. **Edges as Lexical Similarity**: An undirected edge exists between $V_i$ and $V_j$ ($i \ne j$), weighted by their lexical similarity $W(i, j) \in [0, 1]$ computed via TF-IDF cosine similarity.
3. **Graph Recommendation**: A sentence achieves high centrality if it is strongly connected to other sentences that themselves possess high centrality.
4. **Iterative Centrality**: Initial uniform probability mass ($1/N$) is iteratively propagated across weighted edges until the node scores stabilize (converge).

> [!NOTE]
> **Lexical vs. Semantic Distinction**: The similarity graph in this implementation represents **lexical similarity based on TF-IDF sentence representations** (exact lemma and token overlap weighted by distinctiveness). It does **not** model embedding-based semantic similarity (such as Word2Vec, GloVe, BERT, or sentence transformers), which are intentionally excluded from this classical NLP project.

---

## 3. Mathematical Formulation

### 3.1 Intra-Article Sentence TF-IDF Vectors
For term $t$ in sentence $S$ within an article of $N$ sentences:
$$\text{TF}(t, S) = \frac{\text{count}(t \in S)}{|C(S)|}$$
$$\text{DF}(t) = \sum_{i=1}^N \mathbb{I}(t \in S_i)$$
$$\text{IDF}(t) = \ln\left(\frac{N}{\text{DF}(t)}\right)$$
$$\text{TFIDF}(t, S) = \text{TF}(t, S) \times \text{IDF}(t)$$
where $C(S)$ is the sequence of linguistic content tokens (non-stopwords, lemmatized, alphabetic) in sentence $S$.

### 3.2 Pairwise Cosine Similarity
For two sentences represented by sparse TF-IDF vectors $\mathbf{u}$ and $\mathbf{v}$:
$$\text{cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \times \|\mathbf{v}\|} = \frac{\sum_{t \in \mathbf{u} \cap \mathbf{v}} u_t v_t}{\sqrt{\sum_{t} u_t^2} \sqrt{\sum_{t} v_t^2}}$$
- If $\|\mathbf{u}\| = 0$ or $\|\mathbf{v}\| = 0$, $\text{cosine}(\mathbf{u}, \mathbf{v}) = 0.0$ (safe zero-vector handling).
- Edge weights satisfy symmetry: $W[i][j] = W[j][i]$.
- No self-loops: $W[i][i] = 0.0$.
- All edge weights are non-negative: $W[i][j] \ge 0.0$.

### 3.3 PageRank Formulation with Dangling Node Handling
For sentence node $S_i$:
$$\text{PR}(S_i) = \frac{1 - d}{N} + d \left( \sum_{j \notin D, j \ne i} \frac{W[j][i]}{\sum_k W[j][k]} \text{PR}(S_j) + \frac{\sum_{j \in D} \text{PR}(S_j)}{N} \right)$$
Where:
- $d$: Damping factor ($d = 0.85$).
- $N$: Total number of sentences in the article.
- $D = \{j \mid \sum_k W[j][k] = 0\}$: The set of dangling nodes (sentences with zero outgoing similarity edges). Dangling mass is distributed uniformly ($1/N$) across all nodes, guaranteeing conservation of total probability mass ($\sum_i \text{PR}(S_i) = 1.0$).

### 3.4 Convergence Criterion
Power iterations update the PageRank vector until the $L_1$ norm difference satisfies:
$$\sum_{i=1}^N |\text{PR}^{(t+1)}(S_i) - \text{PR}^{(t)}(S_i)| < \tau \quad (\tau = 10^{-6})$$
or when the iteration count reaches $\text{max\_iterations} = 100$.

---

## 4. Implementation Details

The implementation in `summarization/textrank.py` follows a clean, modular structure:
- `calculate_tf()`, `calculate_df()`, `calculate_idf()`: First-principles sparse term-frequency statistics.
- `build_sentence_vectors()`: Generates sparse TF-IDF dictionaries per sentence.
- `cosine_similarity()`: Sparse dot-product and norm evaluation with zero-magnitude protection.
- `build_similarity_matrix()`: Symmetric $N \times N$ matrix construction with zero diagonal.
- `calculate_pagerank()`: Explicit manual PageRank power iteration with uniform dangling mass redistribution.
- `rank_sentences()`: Descending PageRank sorting with deterministic earlier-index tie-breaking.
- `select_top_sentences()`: Top-$K$ selection restored to original chronological sentence index.
- `batch_summarize_file()`: High-throughput streaming batch pipeline reading Phase 2 preprocessed JSONL directly without redundant re-preprocessing.

---

## 5. Configuration

| Parameter | Value | Description |
|---|---|---|
| **Dataset** | CNN/DailyMail v3.0.0 | Academic benchmark |
| **Split** | Test set | Random seed 42 |
| **Article Count** | 1,000 articles | Deterministic sample established in Phase 1 |
| **Summary Length ($K$)** | 3 sentences | Standard initial baseline configuration |
| **Damping Factor ($d$)** | 0.85 | Classical PageRank parameter |
| **Max Iterations** | 100 | Convergence iteration limit |
| **Tolerance ($\tau$)** | $1 \times 10^{-6}$ | $L_1$ convergence threshold |
| **Tie-Breaking** | Lower sentence index | Deterministic reproducibility |
| **Self-Loops** | Excluded ($W[i][i] = 0$) | Prevents artificial self-endorsement |
| **Dangling Nodes** | Uniform ($1/N$) | Preserves total probability mass |

---

## 6. Verified Hand-Calculated Graph Example

To illustrate the exact mechanics of graph construction and PageRank scoring, consider a 3-sentence document ($N = 3$):
- **$S_0$**: `"cats chase mice"` $\rightarrow C(S_0) = \text{['cats', 'chase', 'mice']}$ (3 tokens)
- **$S_1$**: `"cats hunt mice"` $\rightarrow C(S_1) = \text{['cats', 'hunt', 'mice']}$ (3 tokens)
- **$S_2$**: `"football teams won matches"` $\rightarrow C(S_2) = \text{['football', 'teams', 'won', 'matches']}$ (4 tokens)

### 6.1 TF-IDF Vectors
- **Document Frequencies**:
  $\text{DF}(\text{cats}) = 2, \text{DF}(\text{mice}) = 2, \text{DF}(\text{chase}) = 1, \text{DF}(\text{hunt}) = 1$, and all $S_2$ terms have $\text{DF} = 1$.
- **Inverse Document Frequencies** ($\text{IDF}(t) = \ln(3 / \text{DF}(t))$):
  - $\text{IDF}(\text{cats}) = \ln(3/2) \approx 0.405465$
  - $\text{IDF}(\text{mice}) = \ln(3/2) \approx 0.405465$
  - $\text{IDF}(\text{chase}) = \text{IDF}(\text{hunt}) = \ln(3/1) \approx 1.098612$
  - All $S_2$ terms have $\text{IDF} = \ln(3) \approx 1.098612$
- **Resulting Sparse Vectors**:
  - $\mathbf{v}_0 = \{\text{cats}: 0.135155, \text{chase}: 0.366204, \text{mice}: 0.135155\}$
  - $\mathbf{v}_1 = \{\text{cats}: 0.135155, \text{hunt}: 0.366204, \text{mice}: 0.135155\}$
  - $\mathbf{v}_2 = \{\text{football}: 0.274653, \text{teams}: 0.274653, \text{won}: 0.274653, \text{matches}: 0.274653\}$

### 6.2 Exact Similarity Matrix
- $S_0$ and $S_1$ share `cats` and `mice`:
  $$\mathbf{v}_0 \cdot \mathbf{v}_1 = 0.135155^2 + 0.135155^2 \approx 0.036534$$
  $$\|\mathbf{v}_0\| = \|\mathbf{v}_1\| = \sqrt{0.135155^2 + 0.366204^2 + 0.135155^2} \approx 0.413085$$
  $$\text{cosine}(\mathbf{v}_0, \mathbf{v}_1) = \frac{0.036534}{0.413085^2} \approx \mathbf{0.214099}$$
- $S_2$ shares zero terms with $S_0$ and $S_1$, so $\text{cosine}(\mathbf{v}_0, \mathbf{v}_2) = \text{cosine}(\mathbf{v}_1, \mathbf{v}_2) = 0.0$.

$$W = \begin{bmatrix} 0.0 & 0.214099 & 0.0 \\ 0.214099 & 0.0 & 0.0 \\ 0.0 & 0.0 & 0.0 \end{bmatrix}$$

### 6.3 PageRank Centrality
$S_2$ has row sum 0 and is treated as a dangling node. Iterative PageRank converges in 12 iterations to:
- $\text{PR}(S_0) = \mathbf{0.465116}$
- $\text{PR}(S_1) = \mathbf{0.465116}$
- $\text{PR}(S_2) = \mathbf{0.069768}$

$S_0$ and $S_1$ mutually endorse each other, receiving over $93\%$ of the graph's centrality mass, while the isolated $S_2$ receives only baseline teleportation mass.

---

## 7. Five Real CNN/DailyMail Examples

The following records from `dataset/processed/textrank_summaries_1000.jsonl` verify performance on real news articles:

### Example 1
- **Article ID**: `f001ec5c4704938247d27a44948eebb37ae98d01`
- **Original sentence count**: 28
- **Selected sentence indices**: `[1, 8, 11]`
- **Selected TextRank scores**: `[0.066026, 0.054031, 0.054301]`
- **Generated summary**:
  > The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. "As Palestine formally becomes a State Party to the Rome Statute today, the world is also a step closer to ending a long era of impunity and injustice," he said, according to an ICC news release. "As the Rome Statute today enters into force for the State of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Statute.
- **Reference highlights**:
  > Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June .\nIsrael and the United States opposed the move, which could open the door to war crimes investigations against Israelis .

### Example 2
- **Article ID**: `230c522854991d053fe98a718b1defa077a8efef`
- **Original sentence count**: 19
- **Selected sentence indices**: `[2, 3, 9]`
- **Selected TextRank scores**: `[0.093328, 0.076412, 0.138965]`
- **Generated summary**:
  > That's according to Washington State University, where the dog -- a friendly white-and-black bully breed mix now named Theia -- has been receiving care at the Veterinary Teaching Hospital. Four days after her apparent death, the dog managed to stagger to a nearby farm, dirt-covered and emaciated, where she was found by a worker who took her to a vet for help. The veterinary hospital's Good Samaritan Fund committee awarded some money to help pay for the dog's treatment, but Mellado has set up a fundraising page to help meet the remaining cost of the dog's care.
- **Reference highlights**:
  > Theia, a bully breed mix, was apparently hit by a car, whacked with a hammer and buried in a field .\n"She's a true miracle dog and she deserves a good life," says Sara Mellado, who is looking for a home for Theia .

### Example 3
- **Article ID**: `4495ba8f3a340d97a9df1476f8a35502bcce1f69`
- **Original sentence count**: 38
- **Selected sentence indices**: `[1, 23, 34]`
- **Selected TextRank scores**: `[0.045848, 0.058469, 0.052562]`
- **Generated summary**:
  > He is, of course, the Iranian foreign minister. The website of the Iranian Foreign Ministry, which Zarif runs, cannot even agree with itself on when he was born. Later, the website says, Zarif went to make a similar protest at the Iranian mission to the United Nations.
- **Reference highlights**:
  > Mohammad Javad Zarif has spent more time with John Kerry than any other foreign minister .\nHe once participated in a takeover of the Iranian Consulate in San Francisco .\nThe Iranian foreign minister tweets in English .

### Example 4
- **Article ID**: `a38e72fed88684ec8d60dd5856282e999dc8c0ca`
- **Original sentence count**: 11
- **Selected sentence indices**: `[0, 3, 5]`
- **Selected TextRank scores**: `[0.142530, 0.169746, 0.129068]`
- **Generated summary**:
  > (CNN)Five Americans who were monitored for three weeks at an Omaha, Nebraska, hospital after being exposed to Ebola in West Africa have been released, a Nebraska Medicine spokesman said in an email Wednesday. They were exposed to Ebola in Sierra Leone in March, but none developed the deadly virus. They all had contact with a colleague who was diagnosed with the disease and is being treated at the National Institutes of Health in Bethesda, Maryland.
- **Reference highlights**:
  > 17 Americans were exposed to the Ebola virus while in Sierra Leone in March .\nAnother person was diagnosed with the disease and taken to hospital in Maryland .\nNational Institutes of Health says the patient is in fair condition after weeks of treatment .

### Example 5
- **Article ID**: `c27cf1b136cc270023de959e7ab24638021bc43f`
- **Original sentence count**: 22
- **Selected sentence indices**: `[0, 3, 10]`
- **Selected TextRank scores**: `[0.095406, 0.082422, 0.080265]`
- **Generated summary**:
  > (CNN)A Duke student has admitted to hanging a noose made of rope from a tree near a student union, university officials said Thursday. The student was identified during an investigation by campus police and the office of student affairs and admitted to placing the noose on the tree early Wednesday, the university said. This is no Duke we want.
- **Reference highlights**:
  > Student is no longer on Duke University campus and will face disciplinary review .\nSchool officials identified student during investigation and the person admitted to hanging the noose, Duke says .\nThe noose, made of rope, was discovered on campus about 2 a.m.

---

## 8. Observations

1. **Effect of Graph Connectivity on Rare Tokens**:
   In Phase 4 (TF-IDF), isolated dateline markers such as `(CNN)` received high scores because unique tokens yielded high IDF values. In TextRank, a sentence containing isolated tokens typically exhibits weak or zero lexical similarity with subsequent sentences. Because PageRank scores reflect incoming edge weights, isolated sentences receive limited graph reinforcement. However, graph connectivity reduces rather than entirely eliminates rare-token influence (e.g., if a dateline sentence also contains common article vocabulary).
2. **Topical Clustering**:
   Sentences sharing core vocabulary (e.g., *dog*, *treatment*, *veterinary* in Example 2) mutually reinforce each other across the graph, elevating their PageRank centrality relative to peripheral sentences.
3. **Word Count and Sentence Length**:
   Under the implemented TextRank configuration, the selected 3-sentence summaries contained **69.72 words on average**, compared to **43.51 words** for Frequency summarization and **21.65 words** for TF-IDF summarization across the same 1,000 articles. Sentences with higher content token counts tend to participate in more non-zero similarity edges, which can lead to higher overall graph connectivity.

---

## 9. Academic Limitations

1. **Lexical Overlap Dependency**:
   Cosine similarity relies entirely on shared lexical content tokens and lemmas. Semantic equivalents (e.g., *"doctor"* vs *"physician"*, *"treat"* vs *"therapy"*) have zero similarity unless captured by shared vocabulary.
2. **Coreference / Anaphora Blindness**:
   The method does not resolve pronouns (*"he"*, *"they"*, *"that"*). A high-centrality sentence may refer to entities introduced in non-selected prior sentences.
3. **Pairwise Comparison Complexity**:
   Constructing the similarity matrix requires evaluating every unique sentence pair:
   $$\frac{N(N - 1)}{2} \text{ comparisons}$$
   Consequently, graph construction scales as $\mathcal{O}(N^2)$ with the number of sentences $N$.
4. **Discourse Order Disconnect**:
   While selected sentences are restored to original chronological order, TextRank evaluates sentences as an unordered set of nodes without modeling narrative or rhetorical structure.
5. **Extractive Limitation**:
   TextRank selects existing sentences verbatim; it cannot rephrase, synthesize, or compress sentences.

---

## 10. Performance and Complexity

### 10.1 Computational Complexity
- **Sentence-pair similarity construction**: $\mathcal{O}(N^2 \cdot |V|)$ where $N$ is the number of sentences and $|V|$ is the average number of unique terms per sentence pair.
- **PageRank iteration**: $\mathcal{O}(I \cdot |E|)$ where $I$ is the number of power iterations until convergence (mean: 26.52) and $|E| \le N^2$ is the number of non-zero similarity edges.

### 10.2 Measured Benchmark (Local Run)
- **Total Articles**: 1,000
- **Total Sentences Selected**: 3,000 (3.0 per article)
- **Average PageRank Iterations to Convergence**: 26.52
- **Processing Time**: 7.86 seconds
- **Throughput**: 127.16 articles / second

*(Note: Measured runtime reflects local execution on this specific workstation and hardware environment; it illustrates computational feasibility rather than absolute hardware-independent performance).*

---

## 11. Relationship to Phases 3 and 4

This project establishes three distinct classical extractive summarization baselines:

1. **Phase 3 (Frequency)**:
   Scored sentences by overall term frequency across the entire article ($\text{NF}(w) = \text{freq}(w)/\max$). Favors globally recurrent topical keywords.
2. **Phase 4 (TF-IDF)**:
   Scored sentences by length-normalized term distinctiveness ($\text{TF} \times \text{IDF}$). Favors informative, locally distinctive vocabulary within the article.
3. **Phase 5 (TextRank)**:
   Scores sentences by global graph centrality via PageRank on a lexical similarity network. Favors sentences endorsed by mutual content overlap across the document.

---

## 12. Reproducibility

The 1,000-article baseline was generated using:

```powershell
python -m summarization.textrank --dataset "dataset/processed/cnn_dailymail_test_1000_processed.jsonl" --num-sentences 3
```

- **Output File**: `dataset/processed/textrank_summaries_1000.jsonl`
- **Output Record Count**: 1,000 JSON lines

---

## 13. Phase 6 Evaluation Note

> [!IMPORTANT]
> Formal quantitative evaluation (ROUGE-1, ROUGE-2, and ROUGE-L) comparing Frequency, TF-IDF, and TextRank summarization will be conducted in **Phase 6**. No claim is made at this stage regarding which algorithm achieves superior summarization quality.
