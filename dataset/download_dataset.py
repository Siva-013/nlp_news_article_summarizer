"""
CNN/DailyMail Dataset Downloader
Part of Phase 1: Dataset Setup & Exploration

This module acquires a reproducible subset of the CNN/DailyMail (v3.0.0) benchmark
dataset for news summarization using the official HuggingFace datasets repository
(abisee/cnn_dailymail) without needing to download the entire multi-gigabyte corpus.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress Hugging Face symlink warnings on Windows platforms
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError as err:
    print(
        f"[ERROR] Missing required dependencies: {err}.\n"
        "Please activate the virtual environment and run: pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for dataset acquisition."""
    parser = argparse.ArgumentParser(
        description="Download and validate a reproducible sample of the CNN/DailyMail (v3.0.0) dataset."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to sample from (default: test).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of records to download (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility when sampling (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/raw"),
        help="Output directory for raw JSONL files (default: dataset/raw).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the split before sampling using the given seed.",
    )
    return parser.parse_args()


def validate_records(records: List[Dict[str, Any]], requested_count: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Validate downloaded dataset records against strict integrity constraints.

    Checks:
    - Presence of 'id', 'article', and 'highlights'
    - String data types for all fields
    - Non-empty content
    - Uniqueness of record IDs

    Returns:
        Tuple of (valid_records, validation_metrics_dict)
    """
    seen_ids = set()
    valid_records: List[Dict[str, Any]] = []

    metrics = {
        "requested": requested_count,
        "loaded": len(records),
        "valid": 0,
        "empty_articles": 0,
        "empty_highlights": 0,
        "missing_ids": 0,
        "duplicate_ids": 0,
    }

    for item in records:
        record_id = item.get("id")
        article = item.get("article")
        highlights = item.get("highlights")

        is_valid = True

        # Check missing or malformed ID
        if not record_id or not isinstance(record_id, str):
            metrics["missing_ids"] += 1
            is_valid = False
        elif record_id in seen_ids:
            metrics["duplicate_ids"] += 1
            is_valid = False
        else:
            seen_ids.add(record_id)

        # Check article
        if not isinstance(article, str) or len(article.strip()) == 0:
            metrics["empty_articles"] += 1
            is_valid = False

        # Check highlights
        if not isinstance(highlights, str) or len(highlights.strip()) == 0:
            metrics["empty_highlights"] += 1
            is_valid = False

        if is_valid:
            valid_records.append(
                {
                    "id": str(record_id).strip(),
                    "article": article.strip(),
                    "highlights": highlights.strip(),
                }
            )

    metrics["valid"] = len(valid_records)
    return valid_records, metrics


def download_cnn_dailymail(
    split: str = "test",
    samples: int = 100,
    seed: int = 42,
    shuffle: bool = False,
) -> List[Dict[str, Any]]:
    """
    Stream and sample records from CNN/DailyMail v3.0.0.

    Uses streaming to download only the required sample size rather than
    fetching the complete multi-gigabyte dataset.
    """
    if samples <= 0:
        raise ValueError(f"Sample count must be positive, got {samples}")

    dataset_name = "abisee/cnn_dailymail"
    dataset_version = "3.0.0"

    print(f"Connecting to official repository '{dataset_name}' (version {dataset_version})...")
    print(f"Target split: '{split}', requested samples: {samples}, seed: {seed}")

    try:
        # Load dataset in streaming mode for memory and bandwidth efficiency
        streamed_dataset = load_dataset(
            dataset_name,
            dataset_version,
            split=split,
            streaming=True,
        )

        if shuffle:
            print(f"Applying deterministic shuffle with seed={seed} (buffer_size=1000)...")
            streamed_dataset = streamed_dataset.shuffle(seed=seed, buffer_size=1000)

        records: List[Dict[str, Any]] = []
        progress_bar = tqdm(total=samples, desc=f"Streaming {split} samples", unit="records")

        for sample in streamed_dataset:
            records.append(sample)
            progress_bar.update(1)
            if len(records) >= samples:
                break

        progress_bar.close()
        return records

    except Exception as exc:
        raise RuntimeError(
            f"Failed to acquire CNN/DailyMail dataset: {exc}\n"
            "Please verify your internet connection and ensure Hugging Face Hub is reachable."
        ) from exc


def save_to_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Save records to a standard line-delimited JSON (JSONL) file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_out:
        for record in records:
            file_out.write(json.dumps(record, ensure_ascii=False) + "\n")


def display_validation_summary(metrics: Dict[str, int]) -> None:
    """Print readable validation summary to the terminal."""
    print("\nDataset Validation")
    print("------------------")
    print(f"Requested records : {metrics['requested']}")
    print(f"Loaded records    : {metrics['loaded']}")
    print(f"Valid records     : {metrics['valid']}")
    print(f"Empty articles    : {metrics['empty_articles']}")
    print(f"Empty highlights  : {metrics['empty_highlights']}")
    print(f"Missing IDs       : {metrics['missing_ids']}")
    print(f"Duplicate IDs     : {metrics['duplicate_ids']}")
    print("------------------\n")


def main() -> None:
    """Main CLI entrypoint for dataset acquisition."""
    args = parse_arguments()

    output_filename = f"cnn_dailymail_{args.split}_{args.samples}.jsonl"
    output_path = args.output_dir / output_filename

    try:
        raw_records = download_cnn_dailymail(
            split=args.split,
            samples=args.samples,
            seed=args.seed,
            shuffle=args.shuffle,
        )

        valid_records, metrics = validate_records(raw_records, args.samples)
        display_validation_summary(metrics)

        if metrics["valid"] == 0:
            print("[ERROR] No valid records found in the downloaded batch. Aborting save.", file=sys.stderr)
            sys.exit(1)

        save_to_jsonl(valid_records, output_path)
        print(f"Successfully saved {len(valid_records)} valid records to: {output_path.resolve()}")

    except Exception as err:
        print(f"\n[EXECUTION ERROR] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
