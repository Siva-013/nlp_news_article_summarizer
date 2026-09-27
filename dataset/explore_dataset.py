"""
CNN/DailyMail Dataset Exploration & Statistical Analysis
Part of Phase 1: Dataset Setup & Exploration

This module inspects downloaded raw JSONL datasets, validates records,
computes essential text statistics (word count, sentence count, compression ratio,
Type-Token Ratio / lexical richness), outputs structured JSON statistics,
and prints representative article-highlight pairs.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
    import pandas as pd
except ImportError as err:
    print(
        f"[ERROR] Missing required dependencies: {err}.\n"
        "Please activate the virtual environment and ensure pandas and numpy are installed.",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for dataset exploration."""
    parser = argparse.ArgumentParser(
        description="Compute statistical characteristics for CNN/DailyMail JSONL dataset sample."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/raw/cnn_dailymail_test_100.jsonl"),
        help="Path to the input JSONL file (default: dataset/raw/cnn_dailymail_test_100.jsonl).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("dataset/dataset_stats.json"),
        help="Path to save the generated statistics JSON (default: dataset/dataset_stats.json).",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("dataset/dataset_analysis.md"),
        help="Path to save the Markdown analysis report (default: dataset/dataset_analysis.md).",
    )
    parser.add_argument(
        "--num-samples-display",
        type=int,
        default=3,
        help="Number of article/highlight sample pairs to print to terminal (default: 3).",
    )
    return parser.parse_args()


def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load raw records from a JSONL file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the file is empty or contains malformed JSON.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found at: {file_path.resolve()}")

    records: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f_in:
        for line_num, line in enumerate(f_in, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                records.append(record)
            except json.JSONDecodeError as jde:
                raise ValueError(
                    f"Malformed JSON on line {line_num} in '{file_path}': {jde}"
                ) from jde

    if not records:
        raise ValueError(f"Input file '{file_path}' contains zero records.")

    return records


def validate_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Validate dataset records and verify field types and presence.
    """
    seen_ids = set()
    valid_records: List[Dict[str, Any]] = []
    metrics = {
        "total": len(records),
        "valid": 0,
        "invalid": 0,
        "empty_articles": 0,
        "empty_highlights": 0,
        "missing_ids": 0,
        "duplicate_ids": 0,
    }

    for item in records:
        rec_id = item.get("id")
        article = item.get("article")
        highlights = item.get("highlights")

        is_valid = True

        if not rec_id or not isinstance(rec_id, str):
            metrics["missing_ids"] += 1
            is_valid = False
        elif rec_id in seen_ids:
            metrics["duplicate_ids"] += 1
            is_valid = False
        else:
            seen_ids.add(rec_id)

        if not isinstance(article, str) or len(article.strip()) == 0:
            metrics["empty_articles"] += 1
            is_valid = False

        if not isinstance(highlights, str) or len(highlights.strip()) == 0:
            metrics["empty_highlights"] += 1
            is_valid = False

        if is_valid:
            valid_records.append(item)
        else:
            metrics["invalid"] += 1

    metrics["valid"] = len(valid_records)
    return valid_records, metrics


def count_words(text: str) -> int:
    """
    Lightweight, explainable word count definition for Phase 1.
    Splits text on whitespace after stripping leading/trailing spaces.
    (Full morphological tokenization is explicitly deferred to Phase 2).
    """
    if not text:
        return 0
    return len(text.strip().split())


def count_sentences(text: str) -> int:
    """
    Lightweight, explainable sentence segmentation for Phase 1.
    Uses regex punctuation boundaries (.!? followed by whitespace)
    and handles bullet-style highlight boundaries.
    (Full NLP sentence segmentation pipeline belongs to Phase 2).
    """
    if not text or not text.strip():
        return 0
    # Split on terminal punctuation followed by space or newline
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return max(len(sentences), 1)


def calculate_statistics(values: List[float | int]) -> Dict[str, float]:
    """
    Calculate summary statistics (min, max, mean, median, std) for a numerical list.
    """
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "std": 0.0}

    arr = np.array(values, dtype=float)
    return {
        "min": int(np.min(arr)) if np.all(arr.astype(int) == arr) else float(round(float(np.min(arr)), 4)),
        "max": int(np.max(arr)) if np.all(arr.astype(int) == arr) else float(round(float(np.max(arr)), 4)),
        "mean": float(round(float(np.mean(arr)), 2)),
        "median": float(round(float(np.median(arr)), 2)),
        "std": float(round(float(np.std(arr)), 2)),
    }


def calculate_vocabulary_metrics(text: str) -> Tuple[int, int, float]:
    """
    Calculate lexical richness metrics for an individual text:
    - total_words: number of lowercased alphanumeric tokens
    - unique_words: vocabulary cardinality (unique tokens)
    - type_token_ratio (TTR): unique_words / total_words
    """
    # Simple alphanumeric token extraction for lexical diversity estimation
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b", text.lower())
    total = len(tokens)
    if total == 0:
        return 0, 0, 0.0
    unique = len(set(tokens))
    ttr = round(unique / total, 4)
    return total, unique, ttr


def compute_dataset_analysis(records: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Compute comprehensive statistical distributions across all records.
    """
    analysis_data = []

    for item in records:
        art_text = item["article"]
        hl_text = item["highlights"]

        art_words = count_words(art_text)
        art_sents = count_sentences(art_text)
        hl_words = count_words(hl_text)
        hl_sents = count_sentences(hl_text)

        # Compression ratio = highlight words / article words
        comp_ratio = round(hl_words / art_words, 4) if art_words > 0 else 0.0

        art_total_toks, art_unique_toks, art_ttr = calculate_vocabulary_metrics(art_text)

        analysis_data.append(
            {
                "id": item["id"],
                "article_words": art_words,
                "article_sentences": art_sents,
                "highlight_words": hl_words,
                "highlight_sentences": hl_sents,
                "compression_ratio": comp_ratio,
                "unique_words": art_unique_toks,
                "ttr": art_ttr,
            }
        )

    df = pd.DataFrame(analysis_data)

    stats_summary = {
        "articles": {
            "word_count": calculate_statistics(df["article_words"].tolist()),
            "sentence_count": calculate_statistics(df["article_sentences"].tolist()),
        },
        "highlights": {
            "word_count": calculate_statistics(df["highlight_words"].tolist()),
            "sentence_count": calculate_statistics(df["highlight_sentences"].tolist()),
        },
        "compression_ratio": {
            "min": float(round(float(df["compression_ratio"].min()), 4)),
            "max": float(round(float(df["compression_ratio"].max()), 4)),
            "mean": float(round(float(df["compression_ratio"].mean()), 4)),
            "median": float(round(float(df["compression_ratio"].median()), 4)),
        },
        "vocabulary": {
            "mean_unique_words": float(round(float(df["unique_words"].mean()), 2)),
            "median_unique_words": float(round(float(df["unique_words"].median()), 2)),
            "mean_ttr": float(round(float(df["ttr"].mean()), 4)),
            "median_ttr": float(round(float(df["ttr"].median()), 4)),
        },
    }

    return stats_summary, df


def display_samples(records: List[Dict[str, Any]], count: int = 3) -> None:
    """Print readable sample article/reference-summary pairs from the actual dataset."""
    display_count = min(count, len(records))
    print("\n" + "=" * 60)
    print(f"SAMPLE ARTICLE / REFERENCE HIGHLIGHT PAIRS ({display_count} SAMPLES)")
    print("=" * 60)

    for idx in range(display_count):
        rec = records[idx]
        print(f"\n============================================================")
        print(f"SAMPLE {idx + 1} (ID: {rec['id']})")
        print(f"============================================================")
        # Show first 500 characters of article with ellipsis
        art_preview = rec["article"]
        if len(art_preview) > 600:
            art_preview = art_preview[:600] + "\n... [TRUNCATED FOR TERMINAL DISPLAY] ..."
        print(f"ARTICLE:\n{art_preview}\n")
        print(f"REFERENCE HIGHLIGHT:\n{rec['highlights']}")
        print("-" * 60)


def display_terminal_report(
    meta: Dict[str, Any],
    val_metrics: Dict[str, int],
    stats: Dict[str, Any],
    output_json_path: Path,
) -> None:
    """Print the formatted summary report matching exact Phase 1 specification."""
    print("\n============================================================")
    print("CNN/DAILYMAIL DATASET EXPLORATION")
    print("============================================================")
    print(f"Dataset        : {meta['name']}")
    print(f"Version        : {meta['version']}")
    print(f"Split          : {meta['split']}")
    print(f"Samples        : {meta['sample_size']}")
    print(f"Random seed    : {meta['random_seed']}")
    print("------------------------------------------------------------")
    print("VALIDATION")
    print("------------------------------------------------------------")
    print(f"Valid records     : {val_metrics['valid']}")
    print(f"Invalid records   : {val_metrics['invalid']}")
    print(f"Empty articles    : {val_metrics['empty_articles']}")
    print(f"Empty highlights  : {val_metrics['empty_highlights']}")
    print(f"Missing IDs       : {val_metrics['missing_ids']}")
    print(f"Duplicate IDs     : {val_metrics['duplicate_ids']}")
    print("------------------------------------------------------------")
    print("ARTICLE STATISTICS")
    print("------------------------------------------------------------")
    print(f"Mean words        : {stats['articles']['word_count']['mean']}")
    print(f"Median words      : {stats['articles']['word_count']['median']}")
    print(f"Min words         : {stats['articles']['word_count']['min']}")
    print(f"Max words         : {stats['articles']['word_count']['max']}")
    print(f"Std words         : {stats['articles']['word_count']['std']}")
    print(f"Mean sentences    : {stats['articles']['sentence_count']['mean']}")
    print(f"Median sentences  : {stats['articles']['sentence_count']['median']}")
    print(f"Min sentences     : {stats['articles']['sentence_count']['min']}")
    print(f"Max sentences     : {stats['articles']['sentence_count']['max']}")
    print("------------------------------------------------------------")
    print("HIGHLIGHT STATISTICS")
    print("------------------------------------------------------------")
    print(f"Mean words        : {stats['highlights']['word_count']['mean']}")
    print(f"Median words      : {stats['highlights']['word_count']['median']}")
    print(f"Min words         : {stats['highlights']['word_count']['min']}")
    print(f"Max words         : {stats['highlights']['word_count']['max']}")
    print(f"Mean sentences    : {stats['highlights']['sentence_count']['mean']}")
    print(f"Median sentences  : {stats['highlights']['sentence_count']['median']}")
    print("------------------------------------------------------------")
    print("COMPRESSION")
    print("------------------------------------------------------------")
    print(f"Mean ratio        : {stats['compression_ratio']['mean']} ({round(stats['compression_ratio']['mean'] * 100, 2)}%)")
    print(f"Median ratio      : {stats['compression_ratio']['median']} ({round(stats['compression_ratio']['median'] * 100, 2)}%)")
    print(f"Min ratio         : {stats['compression_ratio']['min']}")
    print(f"Max ratio         : {stats['compression_ratio']['max']}")
    print("------------------------------------------------------------")
    print("VOCABULARY")
    print("------------------------------------------------------------")
    print(f"Mean unique words : {stats['vocabulary']['mean_unique_words']}")
    print(f"Median unique words: {stats['vocabulary']['median_unique_words']}")
    print(f"Mean TTR          : {stats['vocabulary']['mean_ttr']}")
    print(f"Median TTR        : {stats['vocabulary']['median_ttr']}")
    print("------------------------------------------------------------")
    print(f"Statistics saved to:\n{output_json_path.resolve()}\n")


def generate_markdown_report(
    meta: Dict[str, Any],
    val_metrics: Dict[str, int],
    stats: Dict[str, Any],
    output_md_path: Path,
) -> None:
    """Generate a readable markdown summary report for documentation."""
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    md_content = f"""# CNN/DailyMail Dataset Analysis Report

**Phase 1: Dataset Setup & Exploration**

## 1. Dataset Metadata
| Attribute | Value |
|:---|:---|
| **Dataset Name** | `{meta['name']}` |
| **Version** | `{meta['version']}` |
| **Split** | `{meta['split']}` |
| **Sample Size** | `{meta['sample_size']}` |
| **Random Seed** | `{meta['random_seed']}` |
| **Source Loader** | `abisee/cnn_dailymail` (HuggingFace Hub) |

## 2. Integrity and Validation
- **Total Records Evaluated**: {val_metrics['total']}
- **Valid Records**: {val_metrics['valid']}
- **Invalid Records**: {val_metrics['invalid']}
- **Missing / Empty Articles**: {val_metrics['empty_articles']}
- **Missing / Empty Highlights**: {val_metrics['empty_highlights']}
- **Missing IDs**: {val_metrics['missing_ids']}
- **Duplicate IDs**: {val_metrics['duplicate_ids']}

## 3. Article and Highlight Distributions

### Article Statistics
- **Word Count**: Mean = {stats['articles']['word_count']['mean']}, Median = {stats['articles']['word_count']['median']}, Range = [{stats['articles']['word_count']['min']} – {stats['articles']['word_count']['max']}], Std = {stats['articles']['word_count']['std']}
- **Sentence Count**: Mean = {stats['articles']['sentence_count']['mean']}, Median = {stats['articles']['sentence_count']['median']}, Range = [{stats['articles']['sentence_count']['min']} – {stats['articles']['sentence_count']['max']}], Std = {stats['articles']['sentence_count']['std']}

### Reference Highlight Statistics
- **Word Count**: Mean = {stats['highlights']['word_count']['mean']}, Median = {stats['highlights']['word_count']['median']}, Range = [{stats['highlights']['word_count']['min']} – {stats['highlights']['word_count']['max']}], Std = {stats['highlights']['word_count']['std']}
- **Sentence Count**: Mean = {stats['highlights']['sentence_count']['mean']}, Median = {stats['highlights']['sentence_count']['median']}, Range = [{stats['highlights']['sentence_count']['min']} – {stats['highlights']['sentence_count']['max']}], Std = {stats['highlights']['sentence_count']['std']}

## 4. Compression Ratio Analysis
$$\\text{{Compression Ratio}} = \\frac{{\\text{{Highlight Word Count}}}}{{\\text{{Article Word Count}}}}$$

- **Mean Compression Ratio**: {stats['compression_ratio']['mean']} ({round(stats['compression_ratio']['mean'] * 100, 2)}%)
- **Median Compression Ratio**: {stats['compression_ratio']['median']} ({round(stats['compression_ratio']['median'] * 100, 2)}%)
- **Min Compression Ratio**: {stats['compression_ratio']['min']}
- **Max Compression Ratio**: {stats['compression_ratio']['max']}

## 5. Lexical Diversity (Type-Token Ratio)
$$\\text{{TTR}} = \\frac{{\\text{{Unique Word Types}}}}{{\\text{{Total Tokens}}}}$$

- **Mean Unique Words per Article**: {stats['vocabulary']['mean_unique_words']}
- **Median Unique Words per Article**: {stats['vocabulary']['median_unique_words']}
- **Mean Type-Token Ratio (TTR)**: {stats['vocabulary']['mean_ttr']}
- **Median Type-Token Ratio (TTR)**: {stats['vocabulary']['median_ttr']}

## 6. Key Empirical Observations
1. **Summary Conciseness**: The human-written highlights average approximately {stats['highlights']['word_count']['mean']} words compared to {stats['articles']['word_count']['mean']} words in the article, demonstrating an average compression ratio of ~{round(stats['compression_ratio']['mean'] * 100, 1)}%.
2. **Extractive Target**: Articles typically contain ~{stats['articles']['sentence_count']['mean']} sentences, while human highlights consist of ~{stats['highlights']['sentence_count']['mean']} bullet sentences, confirming that a 3 to 5 sentence extractive summary aligns closely with the reference length.
3. **Vocabulary Breadth**: An average TTR of ~{stats['vocabulary']['mean_ttr']} indicates substantial lexical variation across news reporting, providing a rich basis for vocabulary extraction and keyword scoring in subsequent phases.
"""
    with output_md_path.open("w", encoding="utf-8") as f_out:
        f_out.write(md_content)


def main() -> None:
    """Main CLI entrypoint for dataset exploration."""
    args = parse_arguments()

    try:
        records = load_dataset(args.input)
        valid_records, val_metrics = validate_records(records)

        if not valid_records:
            print(f"[ERROR] No valid records found in '{args.input}'. Aborting exploration.", file=sys.stderr)
            sys.exit(1)

        # Derive metadata from filename or dataset defaults
        filename = args.input.stem
        # parse e.g. cnn_dailymail_test_100
        split = "test" if "test" in filename else ("train" if "train" in filename else "validation")
        meta = {
            "name": "cnn_dailymail",
            "version": "3.0.0",
            "split": split,
            "sample_size": len(valid_records),
            "random_seed": 42,
        }

        stats_summary, _ = compute_dataset_analysis(valid_records)

        # Assemble full JSON output matching prompt specification
        dataset_stats_json = {
            "dataset": meta,
            **stats_summary,
        }

        # Save structured JSON
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f_out:
            json.dump(dataset_stats_json, f_out, indent=2)

        # Generate markdown report
        generate_markdown_report(meta, val_metrics, stats_summary, args.output_md)

        # Display terminal report
        display_terminal_report(meta, val_metrics, stats_summary, args.output_json)

        # Display sample article/highlight pairs
        display_samples(valid_records, count=args.num_samples_display)

    except Exception as err:
        print(f"\n[EXPLORATION ERROR] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
