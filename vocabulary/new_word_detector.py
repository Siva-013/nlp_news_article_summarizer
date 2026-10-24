"""
New-Word Detection Module
Part of Phase 8: New-Word Detection

This module classifies vocabulary items extracted by Phase 7 (Vocabulary Extraction)
into 'known_words' and 'new_words' based on an explicitly provided list of known
vocabulary.

Key Principles:
1. Strict Classical NLP: Deterministic canonical lookup without LLMs, embeddings,
   or black-box word difficulty classifiers.
2. Relative Novelty: A word is 'new' strictly if it is absent from the user's
   supplied known-vocabulary list under canonical normalization.
3. Metadata Integrity: Preserves all Phase 7 vocabulary fields, importance scores,
   and deterministic ranking without mutation.
4. Exact Accounting: Guarantees total_vocabulary_count == known_word_count + new_word_count.
5. High Performance: Streaming line-by-line batch execution with constant memory footprint.
"""

import argparse
from collections.abc import Iterable
import copy
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tqdm import tqdm


DEFAULT_MATCHING_POLICY = "canonical_lemma_and_normalized_matching"


class NewWordDetector:
    """
    Classical NLP New-Word Detector.
    Matches extracted Phase 7 vocabulary items against an explicit user-supplied
    known-vocabulary repository using canonical lowercasing, whitespace stripping,
    and morphological lemma comparison.
    """

    def __init__(
        self,
        known_vocabulary: Optional[Iterable[Union[str, Dict[str, Any]]]] = None,
        matching_policy: str = DEFAULT_MATCHING_POLICY,
    ):
        """
        Initialize NewWordDetector.

        Args:
            known_vocabulary: Iterable of known words (strings or metadata dicts).
            matching_policy: Description of the matching strategy applied.
        """
        self.matching_policy = matching_policy
        self.known_terms: Set[str] = set()
        self.known_lemmas: Set[str] = set()
        self.known_entities: Set[Tuple[str, str]] = set()

        if known_vocabulary is not None:
            self.load_known_vocabulary(known_vocabulary)

    def load_known_vocabulary(
        self,
        items: Iterable[Union[str, Dict[str, Any]]],
    ) -> int:
        """
        Load and normalize known vocabulary items into canonical lookup sets.
        Handles strings, dictionaries, casing, and leading/trailing whitespace.

        Args:
            items: Iterable of vocabulary items.

        Returns:
            Total count of canonical entries loaded into known sets.
        """
        added_count = 0
        for entry in items:
            if isinstance(entry, str):
                cleaned = entry.strip().lower()
                if cleaned:
                    self.known_terms.add(cleaned)
                    self.known_lemmas.add(cleaned)
                    added_count += 1
            elif isinstance(entry, dict):
                term = str(entry.get("term", "") or entry.get("word", "")).strip().lower()
                lemma = str(entry.get("lemma", "")).strip().lower()
                entity_label = entry.get("entity_label") or entry.get("label")

                if term:
                    self.known_terms.add(term)
                    added_count += 1
                if lemma:
                    self.known_lemmas.add(lemma)
                    added_count += 1
                if term and entity_label:
                    lbl = str(entity_label).strip()
                    self.known_entities.add((term, lbl))
                    added_count += 1
            elif entry is None:
                continue
            else:
                cleaned = str(entry).strip().lower()
                if cleaned:
                    self.known_terms.add(cleaned)
                    added_count += 1

        return added_count

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        matching_policy: str = DEFAULT_MATCHING_POLICY,
    ) -> "NewWordDetector":
        """
        Factory method to construct a NewWordDetector from a file.
        Supports:
        - Plain text file (.txt): one word per line
        - JSON file (.json): array of strings or dict with 'known_words' list
        - JSONL file (.jsonl): one JSON object per line

        Args:
            file_path: Path to known vocabulary file.
            matching_policy: Matching policy descriptor.

        Returns:
            Initialized NewWordDetector instance.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Known vocabulary file not found: {path.resolve()}")

        detector = cls(matching_policy=matching_policy)
        detector.load_from_file(path)
        return detector

    def load_from_file(self, file_path: Union[str, Path]) -> int:
        """
        Load known vocabulary from disk file.

        Args:
            file_path: Path to known vocabulary file.

        Returns:
            Number of distinct entries loaded.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Known vocabulary file not found: {path.resolve()}")

        suffix = path.suffix.lower()
        items_to_load: List[Any] = []

        if suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    items_to_load = content
                elif isinstance(content, dict):
                    # Check common dictionary keys
                    for key in ["known_words", "words", "vocabulary", "items"]:
                        if key in content and isinstance(content[key], list):
                            items_to_load = content[key]
                            break
                    if not items_to_load:
                        # Fallback: load keys or values
                        items_to_load = list(content.keys())
                else:
                    raise ValueError(f"Unsupported JSON format in {path.name}")
        elif suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        record = json.loads(line_str)
                        if isinstance(record, dict):
                            items_to_load.append(record)
                        elif isinstance(record, str):
                            items_to_load.append(record)
                    except json.JSONDecodeError:
                        items_to_load.append(line_str)
        else:
            # Plain text file or unknown extension: one word per line
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        items_to_load.append(line_str)

        return self.load_known_vocabulary(items_to_load)

    def is_known(self, item: Dict[str, Any]) -> bool:
        """
        Determine whether a single Phase 7 vocabulary item matches the user's
        known-vocabulary set under the canonical matching policy.

        Matching criteria:
        1. Normalized form matches known_terms or known_lemmas.
        2. Canonical lemma matches known_lemmas or known_terms.
        3. Representative surface term (lowercased) matches known_terms.
        4. If the item is a named entity, (normalized_text, entity_label) matches known_entities.

        Args:
            item: Vocabulary item dictionary.

        Returns:
            True if item is recognized as known, False otherwise.
        """
        if not isinstance(item, dict):
            return False

        norm_form = str(item.get("normalized_form", "")).strip().lower()
        lemma = str(item.get("lemma", "")).strip().lower()
        surface = str(item.get("term", "")).strip().lower()
        entity_label = item.get("entity_label")

        # 1. Direct normalized term match
        if norm_form and (norm_form in self.known_terms or norm_form in self.known_lemmas):
            return True

        # 2. Canonical morphological lemma match
        if lemma and (lemma in self.known_lemmas or lemma in self.known_terms):
            return True

        # 3. Surface text match (lowercased)
        if surface and (surface in self.known_terms or surface in self.known_lemmas):
            return True

        # 4. Entity-specific match with label
        if entity_label and norm_form:
            lbl = str(entity_label).strip()
            if (norm_form, lbl) in self.known_entities or (surface, lbl) in self.known_entities:
                return True

        return False

    def detect_new_words(
        self,
        extraction_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Classify all vocabulary items from a Phase 7 extraction record into
        'known_words' and 'new_words'.
        Guarantees that the input dictionary is NOT modified.

        Args:
            extraction_record: Phase 7 vocabulary extraction dictionary.

        Returns:
            Structured dictionary conforming to Phase 8 New-Word Detection schema.
        """
        article_id = str(extraction_record.get("article_id", ""))
        raw_items = extraction_record.get("items", [])

        known_words: List[Dict[str, Any]] = []
        new_words: List[Dict[str, Any]] = []

        for item in raw_items:
            # Deepcopy item so outputs never share mutable references with input
            copied_item = copy.deepcopy(item)
            if self.is_known(item):
                known_words.append(copied_item)
            else:
                new_words.append(copied_item)

        total_count = len(raw_items)
        known_count = len(known_words)
        new_count = len(new_words)

        # Basic consistency check
        reconciled = (total_count == known_count + new_count)

        return {
            "article_id": article_id,
            "method": "classical_new_word_detection",
            "matching_policy": self.matching_policy,
            "total_vocabulary_count": total_count,
            "known_word_count": known_count,
            "new_word_count": new_count,
            "known_words": known_words,
            "new_words": new_words,
            "validation": {
                "reconciled": reconciled,
                "difference": total_count - (known_count + new_count),
            },
        }

    def batch_detect_file(
        self,
        input_jsonl: Union[str, Path],
        output_jsonl: Union[str, Path],
        limit: Optional[int] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Stream a Phase 7 extraction JSONL dataset line-by-line, classify vocabulary
        for every article, and write the Phase 8 output dataset.

        Args:
            input_jsonl: Path to Phase 7 vocabulary extractions JSONL.
            output_jsonl: Destination path for Phase 8 new-word detections JSONL.
            limit: Optional maximum number of articles to process.

        Returns:
            Tuple of (processed_article_count, batch_statistics_dict).
        """
        in_path = Path(input_jsonl)
        out_path = Path(output_jsonl)

        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path.resolve()}")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Count total records for progress bar
        total_records = 0
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total_records += 1

        target_count = min(limit, total_records) if limit else total_records
        processed_count = 0
        total_vocab_classified = 0
        total_known_classified = 0
        total_new_classified = 0
        new_word_counts: List[int] = []
        known_word_counts: List[int] = []

        with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
            progress = tqdm(
                total=target_count,
                desc="Detecting new words",
                unit="articles",
            )

            for line in f_in:
                line_str = line.strip()
                if not line_str:
                    continue

                try:
                    record = json.loads(line_str)
                except json.JSONDecodeError as err:
                    print(f"[WARNING] Skipping malformed JSON line: {err}", file=sys.stderr)
                    continue

                detection = self.detect_new_words(record)
                f_out.write(json.dumps(detection, ensure_ascii=False) + "\n")
                processed_count += 1

                n_vocab = detection["total_vocabulary_count"]
                n_known = detection["known_word_count"]
                n_new = detection["new_word_count"]

                total_vocab_classified += n_vocab
                total_known_classified += n_known
                total_new_classified += n_new
                new_word_counts.append(n_new)
                known_word_counts.append(n_known)

                progress.update(1)
                if limit and processed_count >= limit:
                    break

            progress.close()

        def compute_stats(arr: List[int]) -> Dict[str, float]:
            if not arr:
                return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0, "std": 0.0}
            sorted_arr = sorted(arr)
            n = len(sorted_arr)
            mean_val = sum(sorted_arr) / n
            median_val = (
                sorted_arr[n // 2]
                if n % 2 != 0
                else (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2.0
            )
            variance = sum((x - mean_val) ** 2 for x in sorted_arr) / (n - 1 if n > 1 else 1)
            std_val = variance ** 0.5
            return {
                "mean": round(mean_val, 2),
                "median": round(float(median_val), 2),
                "min": int(sorted_arr[0]),
                "max": int(sorted_arr[-1]),
                "std": round(float(std_val), 2),
            }

        stats = {
            "total_articles_processed": processed_count,
            "total_vocabulary_classified": total_vocab_classified,
            "total_known_words": total_known_classified,
            "total_new_words": total_new_classified,
            "known_percentage": round(
                (total_known_classified / total_vocab_classified * 100), 2
            )
            if total_vocab_classified > 0
            else 0.0,
            "new_percentage": round(
                (total_new_classified / total_vocab_classified * 100), 2
            )
            if total_vocab_classified > 0
            else 0.0,
            "new_words_per_article": compute_stats(new_word_counts),
            "known_words_per_article": compute_stats(known_word_counts),
            "known_vocabulary_size": len(self.known_terms),
        }

        return processed_count, stats


def print_single_detection_result(result: Dict[str, Any], max_display: int = 10) -> None:
    """Print readable new-word detection result to terminal."""
    print("\n" + "=" * 65)
    print(f"NEW-WORD DETECTION RESULT (Article ID: {result['article_id']})")
    print("=" * 65)
    print(f"Method                : {result['method']}")
    print(f"Matching Policy       : {result['matching_policy']}")
    print(f"Total Vocabulary      : {result['total_vocabulary_count']}")
    print(f"Known Words           : {result['known_word_count']}")
    print(f"New Words (Target)    : {result['new_word_count']}")
    print(f"Reconciled            : {result['validation']['reconciled']}")
    print("-" * 65)
    print("TOP NEW WORDS IDENTIFIED (FOR LEARNING):")
    print(f"{'TERM':<20} {'POS':<8} {'FREQ':<6} {'SCORE':<8} {'TYPE / ENTITY'}")
    print("-" * 65)

    new_words = result.get("new_words", [])
    for itm in new_words[:max_display]:
        ent_desc = itm.get("entity_label") or "-"
        print(
            f"{itm['term'][:19]:<20} "
            f"{itm['pos']:<8} "
            f"{itm['frequency']:<6} "
            f"{itm['importance_score']:<8.4f} "
            f"{ent_desc}"
        )

    if len(new_words) > max_display:
        print(f"... and {len(new_words) - max_display} more new vocabulary items.")

    print("-" * 65)
    print(f"SAMPLE KNOWN WORDS MATCHED ({len(result.get('known_words', []))} total):")
    known_words = result.get("known_words", [])
    for itm in known_words[:5]:
        print(f"  - {itm['term']} (lemma: {itm['lemma']}, POS: {itm['pos']})")
    print("=" * 65 + "\n")


def main() -> None:
    """CLI entrypoint for Phase 8 New-Word Detector."""
    parser = argparse.ArgumentParser(
        description="Detect unfamiliar vocabulary items by comparing Phase 7 extractions against a known-vocabulary list."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/processed/vocabulary_extractions_1000.jsonl"),
        help="Path to Phase 7 vocabulary extractions JSONL dataset.",
    )
    parser.add_argument(
        "--known-vocab",
        type=Path,
        default=None,
        help="Path to known-vocabulary file (.txt, .json, or .jsonl).",
    )
    parser.add_argument(
        "--known-words",
        type=str,
        default=None,
        help="Inline comma-separated list of known words (e.g. 'say,announced,court,london').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/processed/new_word_detections_1000.jsonl"),
        help="Destination path for Phase 8 new-word detections JSONL output.",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Detect new words for a single article ID from the input dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of articles to process in batch mode.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch new-word detection across the input dataset.",
    )

    args = parser.parse_args()

    detector = NewWordDetector()

    # Load known vocabulary from file if provided
    if args.known_vocab:
        if not args.known_vocab.exists():
            print(f"[ERROR] Known vocabulary file not found: {args.known_vocab}", file=sys.stderr)
            sys.exit(1)
        count = detector.load_from_file(args.known_vocab)
        print(f"[INFO] Loaded {count} terms from {args.known_vocab.resolve()}")

    # Load known words from inline string if provided
    if args.known_words:
        inline_words = [w.strip() for w in args.known_words.split(",") if w.strip()]
        detector.load_known_vocabulary(inline_words)
        print(f"[INFO] Loaded {len(inline_words)} inline known words.")

    if not detector.known_terms and not detector.known_lemmas:
        print("[WARNING] Known-vocabulary list is empty. All vocabulary items will be classified as 'new_words'.")

    # Mode 1: Single article query by ID
    if args.id:
        target_record = None
        if not args.input.exists():
            print(f"[ERROR] Input dataset not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        with args.input.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("article_id") == args.id or rec.get("id") == args.id:
                    target_record = rec
                    break

        if not target_record:
            print(f"[ERROR] Article ID '{args.id}' not found in {args.input}", file=sys.stderr)
            sys.exit(1)

        result = detector.detect_new_words(target_record)
        print_single_detection_result(result)
        return

    # Mode 2: Batch detection (default if --batch or no specific ID requested)
    if args.batch or not args.id:
        print(f"Starting batch new-word detection...")
        print(f"Input:   {args.input.resolve()}")
        print(f"Output:  {args.output.resolve()}")
        print(f"Known Vocabulary Size: {len(detector.known_terms)} canonical entries\n")

        count, stats = detector.batch_detect_file(
            input_jsonl=args.input,
            output_jsonl=args.output,
            limit=args.limit,
        )

        print("\n============================================================")
        print("PHASE 8 NEW-WORD DETECTION BATCH SUMMARY")
        print("============================================================")
        print(f"Articles Processed           : {stats['total_articles_processed']}")
        print(f"Total Vocabulary Evaluated   : {stats['total_vocabulary_classified']}")
        print(f"Known Words Identified       : {stats['total_known_words']:,} ({stats['known_percentage']}%)")
        print(f"New Words Identified         : {stats['total_new_words']:,} ({stats['new_percentage']}%)")
        print("New Words per Article Distribution:")
        print(f"  - Mean   : {stats['new_words_per_article']['mean']}")
        print(f"  - Median : {stats['new_words_per_article']['median']}")
        print(f"  - Min/Max: {stats['new_words_per_article']['min']} / {stats['new_words_per_article']['max']}")
        print(f"  - Std Dev: {stats['new_words_per_article']['std']}")
        print(f"Output Dataset Saved To      : {args.output.resolve()}")
        print("============================================================\n")


if __name__ == "__main__":
    main()
