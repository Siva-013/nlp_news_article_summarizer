# CNN/DailyMail Dataset Analysis Report

**Phase 1: Dataset Setup & Exploration**

## 1. Dataset Metadata
| Attribute | Value |
|:---|:---|
| **Dataset Name** | `cnn_dailymail` |
| **Version** | `3.0.0` |
| **Split** | `test` |
| **Sample Size** | `100` |
| **Random Seed** | `42` |
| **Source Loader** | `abisee/cnn_dailymail` (HuggingFace Hub) |

## 2. Integrity and Validation
- **Total Records Evaluated**: 100
- **Valid Records**: 100
- **Invalid Records**: 0
- **Missing / Empty Articles**: 0
- **Missing / Empty Highlights**: 0
- **Missing IDs**: 0
- **Duplicate IDs**: 0

## 3. Article and Highlight Distributions

### Article Statistics
- **Word Count**: Mean = 540.92, Median = 423.0, Range = [102 – 1481], Std = 340.05
- **Sentence Count**: Mean = 26.55, Median = 21.5, Range = [4 – 117], Std = 18.28

### Reference Highlight Statistics
- **Word Count**: Mean = 34.46, Median = 32.5, Range = [18 – 59], Std = 9.72
- **Sentence Count**: Mean = 2.62, Median = 3.0, Range = [1 – 5], Std = 0.69

## 4. Compression Ratio Analysis
$$\text{Compression Ratio} = \frac{\text{Highlight Word Count}}{\text{Article Word Count}}$$

- **Mean Compression Ratio**: 0.0935 (9.35%)
- **Median Compression Ratio**: 0.0688 (6.88%)
- **Min Compression Ratio**: 0.0185
- **Max Compression Ratio**: 0.2973

## 5. Lexical Diversity (Type-Token Ratio)
$$\text{TTR} = \frac{\text{Unique Word Types}}{\text{Total Tokens}}$$

- **Mean Unique Words per Article**: 277.19
- **Median Unique Words per Article**: 240.5
- **Mean Type-Token Ratio (TTR)**: 0.5518
- **Median Type-Token Ratio (TTR)**: 0.5587

## 6. Key Empirical Observations
1. **Summary Conciseness**: The human-written highlights average approximately 34.46 words compared to 540.92 words in the article, demonstrating an average compression ratio of ~9.3%.
2. **Extractive Target**: Articles typically contain ~26.55 sentences, while human highlights consist of ~2.62 bullet sentences, confirming that a 3 to 5 sentence extractive summary aligns closely with the reference length.
3. **Vocabulary Breadth**: An average TTR of ~0.5518 indicates substantial lexical variation across news reporting, providing a rich basis for vocabulary extraction and keyword scoring in subsequent phases.
