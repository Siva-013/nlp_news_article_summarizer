"""
Vocabulary Extraction Module
Part of Phase 7: Vocabulary Extraction

This module extracts informative vocabulary items (lexical content words and
named entities) from news articles using classical, rule-based NLP techniques.
For every candidate vocabulary item, it computes empirical token frequency,
sentence coverage, linguistic attributes (lemma, POS, entity label), and a
transparent, explainable heuristic importance score.

Key Principles:
1. Strict Classical NLP: No LLMs, embeddings, or black-box classifiers.
2. Dual Representation Harmony: Retains representative surface forms and
   sentence indices while aggregating by linguistic lemmas.
3. Transparent Scoring:
   score(term) = α * normalized_frequency + β * sentence_coverage + γ * entity_bonus
4. Deterministic Tie-Breaking:
   importance_score DESC -> frequency DESC -> first_sentence_idx ASC -> normalized_form ASC.
5. Non-Mutating & JSON-Serializable.
"""

import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm


# Default scoring weights: sum must equal 1.0
DEFAULT_WEIGHTS = {
    "alpha_frequency": 0.4,
    "beta_coverage": 0.4,
    "gamma_entity": 0.2,
}

# Standard eligible POS tags for classical content words
DEFAULT_ELIGIBLE_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

# Default POS salience multipliers (all 1.0 by default)
DEFAULT_POS_WEIGHTS = {
    "NOUN": 1.0,
    "PROPN": 1.0,
    "VERB": 1.0,
    "ADJ": 1.0,
    "ADV": 0.8,
}


class VocabularyExtractor:
    """
    Classical NLP Vocabulary Extractor.
    Extracts high-salience lexical content words and named entities from preprocessed
    article records or raw text input using deterministic frequency and coverage statistics.
    """

    def __init__(
        self,
        pipeline: Optional[Any] = None,
        max_items: int = 30,
        include_entities: bool = True,
        weights: Optional[Dict[str, float]] = None,
        pos_weights: Optional[Dict[str, float]] = None,
        eligible_pos: Optional[Set[str]] = None,
        min_term_length: int = 2,
    ):
        """
        Initialize the VocabularyExtractor.

        Args:
            pipeline: Optional PreprocessingPipeline instance for raw text processing.
            max_items: Maximum number of vocabulary items to return per article.
            include_entities: Whether to extract and include named entities.
            weights: Dictionary containing alpha_frequency, beta_coverage, and gamma_entity.
            pos_weights: Optional POS multiplier dictionary.
            eligible_pos: Set of coarse Universal Dependencies POS tags to accept.
            min_term_length: Minimum character length for candidate words.
        """
        self._pipeline = pipeline
        self.max_items = max(1, int(max_items))
        self.include_entities = bool(include_entities)
        self.min_term_length = max(1, int(min_term_length))

        # Validate and set weights
        self.weights = self._validate_weights(weights or DEFAULT_WEIGHTS)

        # Set POS configuration
        self.eligible_pos = set(eligible_pos or DEFAULT_ELIGIBLE_POS)
        self.pos_weights = dict(pos_weights or DEFAULT_POS_WEIGHTS)

    @property
    def pipeline(self) -> Any:
        """Lazy load PreprocessingPipeline if raw text processing is needed."""
        if self._pipeline is None:
            from preprocessing.pipeline import PreprocessingPipeline
            self._pipeline = PreprocessingPipeline(model_name="en_core_web_sm")
        return self._pipeline

    @staticmethod
    def _validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """
        Validate that scoring weights are non-negative and sum to 1.0 (within float tolerance).
        """
        required_keys = {"alpha_frequency", "beta_coverage", "gamma_entity"}
        if not required_keys.issubset(weights.keys()):
            missing = required_keys - set(weights.keys())
            raise ValueError(f"Missing required weight keys: {missing}")

        for k in required_keys:
            val = weights[k]
            if not isinstance(val, (int, float)) or math.isnan(val) or val < 0:
                raise ValueError(f"Weight '{k}' must be a non-negative finite number, got {val}")

        total = sum(weights[k] for k in required_keys)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Weights must sum to 1.0, got sum = {total:.6f}")

        return {k: float(weights[k]) for k in required_keys}

    def extract_from_processed_record(
        self,
        record: Dict[str, Any],
        max_items: Optional[int] = None,
        include_entities: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Extract vocabulary items from a Phase 2 processed article record.
        Guarantees that the input record is NOT modified in any way.

        Args:
            record: Processed article record containing 'processed_sentences'.
            max_items: Optional override for maximum items to return.
            include_entities: Optional override for entity inclusion.

        Returns:
            JSON-serializable dictionary with extracted vocabulary items and metadata.
        """
        effective_max_items = max(1, int(max_items)) if max_items is not None else self.max_items
        effective_include_entities = (
            bool(include_entities) if include_entities is not None else self.include_entities
        )

        article_id = str(record.get("id", ""))
        sentences = record.get("processed_sentences", [])
        total_sentences = len(sentences)

        if total_sentences == 0:
            return {
                "article_id": article_id,
                "method": "classical_vocabulary_extraction",
                "configuration": {
                    "max_items": effective_max_items,
                    "include_entities": effective_include_entities,
                    "weights": copy.deepcopy(self.weights),
                },
                "total_candidate_items": 0,
                "returned_item_count": 0,
                "items": [],
            }

        # Aggregation structures:
        # candidate_key -> dict of stats
        # For lexical content words: candidate_key = ("WORD", lemma_lower, pos_tag)
        # For named entities: candidate_key = ("ENTITY", normalized_entity_text, entity_label)
        candidates: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        # 1. Process Lexical Content Words from sentences
        for sent in sentences:
            sent_idx = sent.get("sentence_index", 0)
            token_details = sent.get("token_details")

            if token_details and isinstance(token_details, list):
                # Standard Phase 2 full linguistic representation
                for tok_offset, tok in enumerate(token_details):
                    if not isinstance(tok, dict):
                        continue
                    if tok.get("is_stop", False) or tok.get("is_punct", False):
                        continue

                    pos = tok.get("pos", "NOUN")
                    if pos not in self.eligible_pos:
                        continue

                    norm = tok.get("normalized", "").strip().lower()
                    if not norm or len(norm) < self.min_term_length:
                        continue
                    if not any(c.isalnum() for c in norm):
                        continue

                    lemma = tok.get("lemma", norm).strip().lower()
                    if not lemma:
                        lemma = norm

                    surface = tok.get("text", norm).strip()
                    key = ("WORD", lemma, pos)

                    if key not in candidates:
                        candidates[key] = {
                            "type": "WORD",
                            "key": key,
                            "lemma": lemma,
                            "pos": pos,
                            "normalized_form": norm,
                            "surface_forms": Counter(),
                            "frequency": 0,
                            "sentence_indices": set(),
                            "entity_label": None,
                            "first_sentence_idx": sent_idx,
                            "first_token_offset": tok_offset,
                        }

                    cand = candidates[key]
                    cand["frequency"] += 1
                    cand["sentence_indices"].add(sent_idx)
                    cand["surface_forms"][surface] += 1
            else:
                # Fallback path if token_details is missing
                tokens = sent.get("tokens", [])
                lemmas = sent.get("lemmas", [])
                pos_list = sent.get("pos", [])
                content_tokens = set(sent.get("content_tokens", []))

                for tok_offset, token_text in enumerate(tokens):
                    surface = str(token_text).strip()
                    norm = surface.lower()
                    if not norm or len(norm) < self.min_term_length:
                        continue
                    if not any(c.isalnum() for c in norm):
                        continue

                    pos = pos_list[tok_offset] if tok_offset < len(pos_list) else "NOUN"
                    if pos not in self.eligible_pos:
                        continue

                    # If content_tokens available, ensure token is a content token
                    if content_tokens and norm not in content_tokens:
                        continue

                    lemma = lemmas[tok_offset].lower() if tok_offset < len(lemmas) else norm
                    key = ("WORD", lemma, pos)

                    if key not in candidates:
                        candidates[key] = {
                            "type": "WORD",
                            "key": key,
                            "lemma": lemma,
                            "pos": pos,
                            "normalized_form": norm,
                            "surface_forms": Counter(),
                            "frequency": 0,
                            "sentence_indices": set(),
                            "entity_label": None,
                            "first_sentence_idx": sent_idx,
                            "first_token_offset": tok_offset,
                        }

                    cand = candidates[key]
                    cand["frequency"] += 1
                    cand["sentence_indices"].add(sent_idx)
                    cand["surface_forms"][surface] += 1

        # 2. Process Named Entities if enabled
        if effective_include_entities:
            for sent in sentences:
                sent_idx = sent.get("sentence_index", 0)
                entities = sent.get("entities", [])
                if not isinstance(entities, list):
                    continue

                for ent_offset, ent in enumerate(entities):
                    if not isinstance(ent, dict):
                        continue
                    ent_text = ent.get("text", "").strip()
                    ent_label = ent.get("label", "").strip()
                    if not ent_text or not ent_label:
                        continue

                    norm_ent = ent_text.lower()
                    if not any(c.isalnum() for c in norm_ent):
                        continue

                    key = ("ENTITY", norm_ent, ent_label)

                    if key not in candidates:
                        candidates[key] = {
                            "type": "ENTITY",
                            "key": key,
                            "lemma": norm_ent,
                            "pos": "PROPN",
                            "normalized_form": norm_ent,
                            "surface_forms": Counter(),
                            "frequency": 0,
                            "sentence_indices": set(),
                            "entity_label": ent_label,
                            "first_sentence_idx": sent_idx,
                            "first_token_offset": ent_offset,
                        }

                    cand = candidates[key]
                    cand["frequency"] += 1
                    cand["sentence_indices"].add(sent_idx)
                    cand["surface_forms"][ent_text] += 1

        total_candidates = len(candidates)
        if total_candidates == 0:
            return {
                "article_id": article_id,
                "method": "classical_vocabulary_extraction",
                "configuration": {
                    "max_items": effective_max_items,
                    "include_entities": effective_include_entities,
                    "weights": copy.deepcopy(self.weights),
                },
                "total_candidate_items": 0,
                "returned_item_count": 0,
                "items": [],
            }

        # 3. Compute Importance Scores
        max_freq = max(cand["frequency"] for cand in candidates.values())
        alpha = self.weights["alpha_frequency"]
        beta = self.weights["beta_coverage"]
        gamma = self.weights["gamma_entity"]

        candidate_list: List[Dict[str, Any]] = []

        for cand in candidates.values():
            freq = cand["frequency"]
            sent_freq = len(cand["sentence_indices"])
            sent_indices = sorted(list(cand["sentence_indices"]))

            # Normalized features
            norm_freq = freq / max_freq if max_freq > 0 else 0.0
            sent_cov = sent_freq / total_sentences if total_sentences > 0 else 0.0
            entity_bonus = 1.0 if cand["entity_label"] is not None else 0.0

            # Base linear score
            base_score = (alpha * norm_freq) + (beta * sent_cov) + (gamma * entity_bonus)

            # Apply POS salience multiplier
            pos = cand["pos"]
            pos_mult = self.pos_weights.get(pos, 1.0)
            importance_score = round(float(base_score * pos_mult), 4)

            # Pick representative surface form (most frequent surface form)
            best_surface = cand["surface_forms"].most_common(1)[0][0]

            # Generate short, transparent selection reason
            if cand["entity_label"]:
                reason = (
                    f"Named entity ({cand['entity_label']}) occurring {freq} time(s) "
                    f"across {sent_freq} sentence(s) (coverage: {sent_cov:.1%})."
                )
            else:
                reason = (
                    f"Salient {pos} occurring {freq} time(s) "
                    f"across {sent_freq} sentence(s) (coverage: {sent_cov:.1%})."
                )

            item = {
                "term": best_surface,
                "normalized_form": cand["normalized_form"],
                "lemma": cand["lemma"],
                "pos": pos,
                "frequency": freq,
                "sentence_frequency": sent_freq,
                "sentence_indices": sent_indices,
                "entity_label": cand["entity_label"],
                "importance_score": importance_score,
                "selection_reason": reason,
                # Metadata fields used for deterministic sorting
                "_first_sentence_idx": cand["first_sentence_idx"],
                "_first_token_offset": cand["first_token_offset"],
            }
            candidate_list.append(item)

        # 4. Deterministic Ranking
        # Priority:
        # 1. importance_score DESC (-score)
        # 2. frequency DESC (-freq)
        # 3. first_sentence_idx ASC
        # 4. normalized_form ASC
        candidate_list.sort(
            key=lambda x: (
                -x["importance_score"],
                -x["frequency"],
                x["_first_sentence_idx"],
                x["_first_token_offset"],
                x["normalized_form"],
            )
        )

        # Truncate to effective_max_items and clean internal sort keys
        selected_items: List[Dict[str, Any]] = []
        for raw_item in candidate_list[:effective_max_items]:
            cleaned_item = {k: v for k, v in raw_item.items() if not k.startswith("_")}
            selected_items.append(cleaned_item)

        return {
            "article_id": article_id,
            "method": "classical_vocabulary_extraction",
            "configuration": {
                "max_items": effective_max_items,
                "include_entities": effective_include_entities,
                "weights": copy.deepcopy(self.weights),
            },
            "total_candidate_items": total_candidates,
            "returned_item_count": len(selected_items),
            "items": selected_items,
        }

    def extract_from_text(
        self,
        raw_text: str,
        article_id: str = "raw_article",
        max_items: Optional[int] = None,
        include_entities: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Extract vocabulary from a raw text article string by passing it through
        the existing Phase 2 PreprocessingPipeline.

        Args:
            raw_text: Raw article string.
            article_id: Identifier for the article.
            max_items: Maximum items to return.
            include_entities: Whether to extract named entities.

        Returns:
            Structured extraction dictionary.
        """
        processed_record = self.pipeline.process_article(
            raw_article=raw_text,
            record_id=article_id,
        )
        return self.extract_from_processed_record(
            record=processed_record,
            max_items=max_items,
            include_entities=include_entities,
        )

    def extract_from_file(
        self,
        file_path: Path,
        max_items: Optional[int] = None,
        include_entities: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Extract vocabulary from a raw text file on disk.

        Args:
            file_path: Path to the text file.
            max_items: Maximum items to return.
            include_entities: Whether to extract named entities.

        Returns:
            Structured extraction dictionary.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Article file not found: {path.resolve()}")
        text = path.read_text(encoding="utf-8")
        return self.extract_from_text(
            raw_text=text,
            article_id=path.stem,
            max_items=max_items,
            include_entities=include_entities,
        )

    def batch_extract_file(
        self,
        input_jsonl: Path,
        output_jsonl: Path,
        max_items: Optional[int] = None,
        include_entities: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Process a preprocessed JSONL file line-by-line and stream vocabulary extractions.
        Avoids loading the entire dataset into memory.

        Args:
            input_jsonl: Path to Phase 2 processed JSONL dataset.
            output_jsonl: Destination path for vocabulary extraction JSONL.
            max_items: Maximum vocabulary items per article.
            include_entities: Whether to include named entities.
            limit: Optional maximum number of records to process.

        Returns:
            Tuple of (records_processed_count, batch_statistics_dict).
        """
        in_path = Path(input_jsonl)
        out_path = Path(output_jsonl)

        if not in_path.exists():
            raise FileNotFoundError(f"Input dataset not found: {in_path.resolve()}")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Count total records for progress bar
        total_records = 0
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total_records += 1

        target_count = min(limit, total_records) if limit else total_records
        processed_count = 0
        total_items_extracted = 0
        total_candidates_evaluated = 0
        pos_distribution: Dict[str, int] = defaultdict(int)
        entity_label_distribution: Dict[str, int] = defaultdict(int)
        scores_list: List[float] = []

        with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
            progress = tqdm(
                total=target_count,
                desc="Extracting vocabulary",
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

                extraction = self.extract_from_processed_record(
                    record=record,
                    max_items=max_items,
                    include_entities=include_entities,
                )

                f_out.write(json.dumps(extraction, ensure_ascii=False) + "\n")
                processed_count += 1

                # Aggregate batch metrics
                items = extraction.get("items", [])
                total_items_extracted += len(items)
                total_candidates_evaluated += extraction.get("total_candidate_items", 0)

                for itm in items:
                    pos_distribution[itm["pos"]] += 1
                    scores_list.append(itm["importance_score"])
                    if itm["entity_label"]:
                        entity_label_distribution[itm["entity_label"]] += 1

                progress.update(1)
                if limit and processed_count >= limit:
                    break

            progress.close()

        avg_items_per_article = (
            total_items_extracted / processed_count if processed_count > 0 else 0.0
        )
        avg_candidates_per_article = (
            total_candidates_evaluated / processed_count if processed_count > 0 else 0.0
        )
        avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0

        stats = {
            "total_articles_processed": processed_count,
            "total_vocabulary_items_extracted": total_items_extracted,
            "mean_candidates_per_article": round(avg_candidates_per_article, 2),
            "mean_items_per_article": round(avg_items_per_article, 2),
            "mean_importance_score": round(avg_score, 4),
            "pos_distribution": dict(pos_distribution),
            "entity_label_distribution": dict(entity_label_distribution),
        }

        return processed_count, stats


def print_single_extraction_result(extraction: Dict[str, Any], top_n: int = 15) -> None:
    """Print formatted vocabulary extraction result to terminal."""
    print("\n" + "=" * 65)
    print(f"VOCABULARY EXTRACTION RESULT (Article ID: {extraction['article_id']})")
    print("=" * 65)
    print(f"Method                : {extraction['method']}")
    print(f"Total Candidates      : {extraction['total_candidate_items']}")
    print(f"Returned Items        : {extraction['returned_item_count']}")
    config = extraction.get("configuration", {})
    print(f"Include Entities      : {config.get('include_entities', True)}")
    weights = config.get("weights", {})
    print(f"Weights (alpha/beta/gamma): alpha={weights.get('alpha_frequency')}, beta={weights.get('beta_coverage')}, gamma={weights.get('gamma_entity')}")
    print("-" * 65)
    print(f"{'TERM':<20} {'POS':<8} {'FREQ':<6} {'SENTS':<6} {'SCORE':<8} {'TYPE / ENTITY'}")
    print("-" * 65)

    items = extraction.get("items", [])
    display_items = items[:top_n]
    for itm in display_items:
        ent_desc = itm["entity_label"] if itm["entity_label"] else "-"
        print(
            f"{itm['term'][:19]:<20} "
            f"{itm['pos']:<8} "
            f"{itm['frequency']:<6} "
            f"{itm['sentence_frequency']:<6} "
            f"{itm['importance_score']:<8.4f} "
            f"{ent_desc}"
        )

    if len(items) > top_n:
        print(f"... and {len(items) - top_n} more vocabulary items.")
    print("=" * 65 + "\n")


def main() -> None:
    """CLI entrypoint for Phase 7 Vocabulary Extractor."""
    parser = argparse.ArgumentParser(
        description="Extract salient vocabulary items and named entities from news articles."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/processed/cnn_dailymail_test_1000_processed.jsonl"),
        help="Path to Phase 2 processed JSONL dataset.",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Extract vocabulary from a single article matching this ID in the dataset.",
    )
    parser.add_argument(
        "--article-file",
        type=Path,
        default=None,
        help="Path to raw text file to extract vocabulary from.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Raw text string to extract vocabulary from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/processed/vocabulary_extractions_1000.jsonl"),
        help="Destination path for batch extraction JSONL output.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=30,
        help="Maximum number of vocabulary items per article (default: 30).",
    )
    parser.add_argument(
        "--no-entities",
        action="store_true",
        help="Exclude named entities from extracted vocabulary.",
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
        help="Run batch extraction on the dataset file.",
    )

    args = parser.parse_args()
    include_entities = not args.no_entities

    extractor = VocabularyExtractor(
        max_items=args.max_items,
        include_entities=include_entities,
    )

    # Mode 1: Direct text string
    if args.text:
        res = extractor.extract_from_text(args.text, article_id="cli_text")
        print_single_extraction_result(res)
        return

    # Mode 2: Single text file
    if args.article_file:
        res = extractor.extract_from_file(args.article_file)
        print_single_extraction_result(res)
        return

    # Mode 3: Query specific ID from processed dataset
    if args.id:
        target_record = None
        if not args.dataset.exists():
            print(f"[ERROR] Dataset not found: {args.dataset}", file=sys.stderr)
            sys.exit(1)

        with args.dataset.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("id") == args.id:
                    target_record = rec
                    break

        if not target_record:
            print(f"[ERROR] Article ID '{args.id}' not found in {args.dataset}", file=sys.stderr)
            sys.exit(1)

        res = extractor.extract_from_processed_record(target_record)
        print_single_extraction_result(res)
        return

    # Mode 4: Batch dataset extraction (default if --batch or no specific item requested)
    if args.batch or (not args.text and not args.article_file and not args.id):
        print(f"Starting batch vocabulary extraction...")
        print(f"Input:   {args.dataset.resolve()}")
        print(f"Output:  {args.output.resolve()}")
        print(f"Options: max_items={args.max_items}, include_entities={include_entities}\n")

        count, stats = extractor.batch_extract_file(
            input_jsonl=args.dataset,
            output_jsonl=args.output,
            max_items=args.max_items,
            include_entities=include_entities,
            limit=args.limit,
        )

        print("\n============================================================")
        print("PHASE 7 VOCABULARY EXTRACTION BATCH SUMMARY")
        print("============================================================")
        print(f"Articles Processed           : {stats['total_articles_processed']}")
        print(f"Total Vocabulary Items Saved : {stats['total_vocabulary_items_extracted']}")
        print(f"Mean Candidates per Article  : {stats['mean_candidates_per_article']}")
        print(f"Mean Items per Article       : {stats['mean_items_per_article']}")
        print(f"Mean Salience Score          : {stats['mean_importance_score']}")
        print("POS Distribution of Extracted Items:")
        for pos, c in sorted(stats["pos_distribution"].items(), key=lambda x: -x[1]):
            print(f"  - {pos:<8}: {c:,}")
        if stats["entity_label_distribution"]:
            print("Top Entity Types Extracted:")
            for ent_label, c in sorted(stats["entity_label_distribution"].items(), key=lambda x: -x[1])[:8]:
                print(f"  - {ent_label:<8}: {c:,}")
        print(f"Output Dataset Saved To      : {args.output.resolve()}")
        print("============================================================\n")


if __name__ == "__main__":
    main()
