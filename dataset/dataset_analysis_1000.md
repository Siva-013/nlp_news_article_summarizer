# CNN/DailyMail Dataset Analysis Report

**Phase 1: Dataset Setup & Exploration**

## 1. Dataset Metadata
| Attribute | Value |
|:---|:---|
| **Dataset Name** | `cnn_dailymail` |
| **Version** | `3.0.0` |
| **Split** | `test` |
| **Sample Size** | `1000` |
| **Random Seed** | `42` |
| **Source Loader** | `abisee/cnn_dailymail` (HuggingFace Hub) |

## 2. Integrity and Validation
- **Total Records Evaluated**: 1000
- **Valid Records**: 1000
- **Invalid Records**: 0
- **Missing / Empty Articles**: 0
- **Missing / Empty Highlights**: 0
- **Missing IDs**: 0
- **Duplicate IDs**: 0

## 3. Article and Highlight Distributions

### Article Statistics
- **Word Count**: Mean = 623.02, Median = 566.0, Range = [73 – 1750], Std = 348.65
- **Sentence Count**: Mean = 31.83, Median = 28.0, Range = [3 – 117], Std = 18.93

### Reference Highlight Statistics
- **Word Count**: Mean = 34.47, Median = 34.0, Range = [14 – 80], Std = 9.66
- **Sentence Count**: Mean = 2.63, Median = 3.0, Range = [1 – 7], Std = 0.71

## 4. Compression Ratio Analysis
$$\text{Compression Ratio} = \frac{\text{Highlight Word Count}}{\text{Article Word Count}}$$

- **Mean Compression Ratio**: 0.0756 (7.56%)
- **Median Compression Ratio**: 0.061 (6.1%)
- **Min Compression Ratio**: 0.012
- **Max Compression Ratio**: 0.3291

## 5. Lexical Diversity (Type-Token Ratio)
$$\text{TTR} = \frac{\text{Unique Word Types}}{\text{Total Tokens}}$$

- **Mean Unique Words per Article**: 310.14
- **Median Unique Words per Article**: 292.5
- **Mean Type-Token Ratio (TTR)**: 0.5316
- **Median Type-Token Ratio (TTR)**: 0.5215

## 6. Key Empirical Observations
1. **Summary Conciseness**: The human-written highlights average approximately 34.47 words compared to 623.02 words in the article, demonstrating an average compression ratio of ~7.6%.
2. **Extractive Target**: Articles typically contain ~31.83 sentences, while human highlights consist of ~2.63 bullet sentences, confirming that a 3 to 5 sentence extractive summary aligns closely with the reference length.
3. **Vocabulary Breadth**: An average TTR of ~0.5316 indicates substantial lexical variation across news reporting, providing a rich basis for vocabulary extraction and keyword scoring in subsequent phases.
