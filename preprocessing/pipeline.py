"""
Central NLP Preprocessing Pipeline
Part of Phase 2: Data Preprocessing Pipeline

This module coordinates the complete linguistic preprocessing pipeline:
1. HTML/Markup cleaning
2. Unicode normalization (NFC)
3. Whitespace normalization
4. Sentence segmentation
5. Tokenization
6. Text normalization
7. Stop-word identification & content token filtering
8. Lemmatization
9. Part-of-speech (POS) tagging (coarse & fine-grained)
10. Named Entity Recognition (NER)
11. Dual Representation assembly (Original sentences preserved for extractive summarization)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# Suppress Hugging Face symlink warnings on Windows platforms
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    import numpy as np
    import spacy
    from tqdm import tqdm
except ImportError as err:
    print(
        f"[ERROR] Missing required dependencies: {err}.\n"
        "Please activate the virtual environment and ensure spacy, numpy, and tqdm are installed.",
        file=sys.stderr,
    )
    sys.exit(1)

from preprocessing.cleaner import clean_html
from preprocessing.normalizer import normalize_unicode, normalize_whitespace, normalize_token
from preprocessing.sentence_splitter import SentenceSplitter
from preprocessing.tokenizer import SpacyTokenizer, configure_spacy_tokenizer
from preprocessing.stopwords import StopwordHandler
from preprocessing.lemmatizer import SpacyLemmatizer
from preprocessing.linguistic_features import LinguisticFeatureExtractor


class PreprocessingPipeline:
    """
    Central orchestration class for the classical NLP preprocessing pipeline.
    Maintains dual representation: original article text for extractive summarization
    and rich linguistic annotations for feature extraction and analytical scoring.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the preprocessing pipeline and load the spaCy English model.

        Args:
            model_name: spaCy linguistic model name (default: en_core_web_sm).
        """
        self.model_name = model_name
        try:
            # We use standard rule/dependency sentence splitting, tagger, lemmatizer, and ner
            self.nlp = spacy.load(model_name)
            configure_spacy_tokenizer(self.nlp)
        except OSError as exc:

            raise OSError(
                f"Model '{model_name}' is not installed.\n"
                f"Please install it in the virtual environment using:\n"
                f"python -m spacy download {model_name}"
            ) from exc

        # Initialize sub-modules sharing the same loaded nlp model
        self.sentence_splitter = SentenceSplitter(nlp=self.nlp)
        self.tokenizer = SpacyTokenizer(nlp=self.nlp)
        self.stopword_handler = StopwordHandler(nlp=self.nlp)
        self.lemmatizer = SpacyLemmatizer(nlp=self.nlp)
        self.feature_extractor = LinguisticFeatureExtractor(nlp=self.nlp)

    def clean_and_normalize_text(self, raw_text: str) -> str:
        """
        Apply HTML cleaning, Unicode normalization (NFC), and whitespace normalization.

        Args:
            raw_text: Raw article string.

        Returns:
            Cleaned and normalized text.
        """
        cleaned = clean_html(raw_text)
        unicode_norm = normalize_unicode(cleaned, form="NFC")
        whitespace_norm = normalize_whitespace(unicode_norm)
        return whitespace_norm

    def process_article(
        self,
        raw_article: str,
        record_id: str = "",
        highlights: str = "",
    ) -> Dict[str, Any]:
        """
        Process a single article through the complete linguistic preprocessing pipeline.

        Returns a structured dictionary strictly upholding the Dual Representation:
        - Original representation: raw article, cleaned article, original sentence list.
        - Processed representation: token metadata, normalized tokens, content tokens,
          lemmas, POS tags, and named entities per sentence.
        """
        # Step 1-3: Clean HTML, normalize Unicode (NFC), normalize whitespace
        cleaned_article = self.clean_and_normalize_text(raw_article)

        if not cleaned_article:
            return {
                "id": record_id,
                "original_article": raw_article,
                "cleaned_article": "",
                "original_sentences": [],
                "processed_sentences": [],
                "highlights": highlights,
            }

        # Step 4: Parse with spaCy for sentence segmentation, tagging, and NER
        doc = self.nlp(cleaned_article)

        original_sentences: List[str] = []
        processed_sentences: List[Dict[str, Any]] = []

        for sent_idx, sent_span in enumerate(doc.sents):
            sent_text = sent_span.text.strip()
            if not sent_text:
                continue

            original_sentences.append(sent_text)

            # Step 5: Tokenization
            tokens = [tok.text for tok in sent_span]

            # Step 6: Text Normalization (lowercased analytical form)
            normalized_tokens = [normalize_token(tok.text) for tok in sent_span]

            # Step 7: Stop-word filtering for content words
            content_tokens = self.stopword_handler.filter_content_tokens(list(sent_span))

            # Step 8: Lemmatization
            lemmas = [self.lemmatizer.lemmatize_token(tok) for tok in sent_span]

            # Step 9: POS Tagging (coarse and detailed)
            pos_dict = self.feature_extractor.extract_pos_tags(list(sent_span))

            # Step 10: Named Entity Recognition
            entities = self.feature_extractor.extract_entities(sent_span)

            # Assemble detailed token metadata
            token_details = []
            for tok in sent_span:
                token_details.append(
                    {
                        "text": tok.text,
                        "normalized": normalize_token(tok.text),
                        "lemma": self.lemmatizer.lemmatize_token(tok),
                        "pos": tok.pos_,
                        "tag": tok.tag_,
                        "is_stop": bool(tok.is_stop),
                        "is_punct": bool(tok.is_punct),
                    }
                )

            processed_sentences.append(
                {
                    "sentence_index": sent_idx,
                    "original_text": sent_text,
                    "tokens": tokens,
                    "normalized_tokens": normalized_tokens,
                    "content_tokens": content_tokens,
                    "lemmas": lemmas,
                    "pos": pos_dict["pos"],
                    "tags": pos_dict["tags"],
                    "token_details": token_details,
                    "entities": entities,
                }
            )

        return {
            "id": record_id,
            "original_article": raw_article,
            "cleaned_article": cleaned_article,
            "original_sentences": original_sentences,
            "processed_sentences": processed_sentences,
            "highlights": highlights,
        }

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        limit: Optional[int] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Process a JSONL file line-by-line using streaming to prevent high memory usage.

        Args:
            input_path: Path to raw input JSONL.
            output_path: Path to save processed JSONL.
            limit: Optional limit on the number of records to process.

        Returns:
            Tuple of (processed_records_count, aggregated_statistics_dict).
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found at: {input_path.resolve()}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Count total records for progress bar
        total_records = 0
        with input_path.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                if line.strip():
                    total_records += 1

        target_count = min(limit, total_records) if limit else total_records
        print(f"Starting batch preprocessing on {target_count} records...")
        print(f"Input:  {input_path.resolve()}")
        print(f"Output: {output_path.resolve()}\n")

        processed_count = 0
        stats_tracker = {
            "total_articles": 0,
            "sentence_counts": [],
            "token_counts": [],
            "content_token_counts": [],
            "stopword_counts": [],
            "entity_counts": [],
            "normalized_changes": 0,
            "total_tokens_evaluated": 0,
        }

        with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
            progress = tqdm(total=target_count, desc="Processing articles", unit="articles")

            for line in f_in:
                line_str = line.strip()
                if not line_str:
                    continue

                record = json.loads(line_str)
                rec_id = record.get("id", "")
                raw_article = record.get("article", "")
                highlights = record.get("highlights", "")

                processed_record = self.process_article(
                    raw_article=raw_article,
                    record_id=rec_id,
                    highlights=highlights,
                )

                f_out.write(json.dumps(processed_record, ensure_ascii=False) + "\n")
                processed_count += 1

                # Accumulate statistics for before/after analysis
                num_sents = len(processed_record["processed_sentences"])
                num_tokens = sum(len(s["tokens"]) for s in processed_record["processed_sentences"])
                num_content = sum(len(s["content_tokens"]) for s in processed_record["processed_sentences"])
                num_stopwords = sum(
                    sum(1 for t in s["token_details"] if t["is_stop"])
                    for s in processed_record["processed_sentences"]
                )
                num_entities = sum(len(s["entities"]) for s in processed_record["processed_sentences"])
                num_norm_changed = sum(
                    sum(1 for t in s["token_details"] if t["text"] != t["normalized"])
                    for s in processed_record["processed_sentences"]
                )

                stats_tracker["sentence_counts"].append(num_sents)
                stats_tracker["token_counts"].append(num_tokens)
                stats_tracker["content_token_counts"].append(num_content)
                stats_tracker["stopword_counts"].append(num_stopwords)
                stats_tracker["entity_counts"].append(num_entities)
                stats_tracker["normalized_changes"] += num_norm_changed
                stats_tracker["total_tokens_evaluated"] += num_tokens

                progress.update(1)
                if limit and processed_count >= limit:
                    break

            progress.close()

        stats_tracker["total_articles"] = processed_count
        summary_stats = self._compute_summary_stats(stats_tracker)
        return processed_count, summary_stats

    def _compute_summary_stats(self, tracker: Dict[str, Any]) -> Dict[str, Any]:
        """Compute mean, median, min, max for batch preprocessing distributions."""
        n_articles = tracker["total_articles"]
        if n_articles == 0:
            return {}

        total_toks = tracker["total_tokens_evaluated"]
        pct_norm_changed = round((tracker["normalized_changes"] / total_toks * 100), 2) if total_toks > 0 else 0.0

        def calc(vals: List[int]) -> Dict[str, float]:
            arr = np.array(vals, dtype=float)
            return {
                "mean": float(round(float(np.mean(arr)), 2)),
                "median": float(round(float(np.median(arr)), 2)),
                "min": int(np.min(arr)),
                "max": int(np.max(arr)),
            }

        return {
            "total_articles": n_articles,
            "sentences_per_article": calc(tracker["sentence_counts"]),
            "tokens_per_article": calc(tracker["token_counts"]),
            "content_tokens_per_article": calc(tracker["content_token_counts"]),
            "stopwords_per_article": calc(tracker["stopword_counts"]),
            "entities_per_article": calc(tracker["entity_counts"]),
            "total_tokens_evaluated": total_toks,
            "tokens_changed_by_normalization": tracker["normalized_changes"],
            "percentage_tokens_changed_by_normalization": pct_norm_changed,
        }


def generate_preprocessing_report(
    stats: Dict[str, Any],
    input_file: Path,
    output_file: Path,
    report_md_path: Path,
) -> None:
    """Generate Markdown before/after preprocessing analysis report."""
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Phase 2 Preprocessing Analysis Report

## 1. Execution Overview
- **Linguistic Engine**: `spaCy (en_core_web_sm)`
- **Input Corpus**: `{input_file.resolve()}`
- **Output Corpus**: `{output_file.resolve()}`
- **Articles Processed**: {stats['total_articles']}
- **Total Tokens Evaluated**: {stats['total_tokens_evaluated']:,}

## 2. Preprocessing Metrics Summary
| Metric | Mean | Median | Min | Max |
|:---|:---:|:---:|:---:|:---:|
| **Sentences per Article** | {stats['sentences_per_article']['mean']} | {stats['sentences_per_article']['median']} | {stats['sentences_per_article']['min']} | {stats['sentences_per_article']['max']} |
| **Total Tokens per Article** | {stats['tokens_per_article']['mean']} | {stats['tokens_per_article']['median']} | {stats['tokens_per_article']['min']} | {stats['tokens_per_article']['max']} |
| **Content Tokens per Article** | {stats['content_tokens_per_article']['mean']} | {stats['content_tokens_per_article']['median']} | {stats['content_tokens_per_article']['min']} | {stats['content_tokens_per_article']['max']} |
| **Stop Words per Article** | {stats['stopwords_per_article']['mean']} | {stats['stopwords_per_article']['median']} | {stats['stopwords_per_article']['min']} | {stats['stopwords_per_article']['max']} |
| **Named Entities per Article** | {stats['entities_per_article']['mean']} | {stats['entities_per_article']['median']} | {stats['entities_per_article']['min']} | {stats['entities_per_article']['max']} |

## 3. Normalization and Lexical Reduction
- **Tokens Modified by Lowercasing/Normalization**: {stats['tokens_changed_by_normalization']:,} ({stats['percentage_tokens_changed_by_normalization']}%)
- **Stop-Word Ratio**: Approximately {round((stats['stopwords_per_article']['mean'] / stats['tokens_per_article']['mean']) * 100, 1)}% of article tokens are grammatical function words (stop words).
- **Information Density**: Content tokens constitute approximately {round((stats['content_tokens_per_article']['mean'] / stats['tokens_per_article']['mean']) * 100, 1)}% of total words, providing dense semantic features for future TF-IDF and TextRank ranking.

## 4. Dual Representation Guarantee
- Every processed record strictly preserves the verbatim `original_article` and `original_sentences` alongside the linguistically enriched `processed_sentences`.
- Future extractive summarization algorithms in Phase 3, 4, and 5 can score sentences using the processed representation (lemmas, content words, POS, entities) while directly returning unmodified original sentences in the final summary.
"""
    with report_md_path.open("w", encoding="utf-8") as f_out:
        f_out.write(content)


def display_real_examples(processed_jsonl_path: Path, count: int = 5) -> None:
    """Display at least 5 real examples from the CNN/DailyMail processed dataset."""
    print("\n" + "=" * 70)
    print(f"REAL CNN/DAILYMAIL PREPROCESSED EXAMPLES ({count} SAMPLES)")
    print("=" * 70)

    loaded = 0
    with processed_jsonl_path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)
            loaded += 1

            orig_art = rec["original_article"]
            art_snippet = orig_art[:200] + "..." if len(orig_art) > 200 else orig_art
            clean_art = rec["cleaned_article"]
            clean_snippet = clean_art[:200] + "..." if len(clean_art) > 200 else clean_art

            sents = rec["processed_sentences"]
            first_sent = sents[0] if sents else {}

            print(f"\n======================================================================")
            print(f"EXAMPLE {loaded} — ARTICLE ID: {rec['id']}")
            print(f"======================================================================")
            print(f"1. RAW TEXT SNIPPET:\n   \"{art_snippet}\"\n")
            print(f"2. CLEANED TEXT SNIPPET:\n   \"{clean_snippet}\"\n")
            print(f"3. SENTENCE SEGMENTATION:")
            print(f"   Total Sentences Identified: {len(sents)}")
            if sents:
                print(f"   First Sentence Text: \"{first_sent.get('original_text', '')}\"\n")

            if first_sent:
                print(f"4. TOKENIZATION (First Sentence):")
                print(f"   {first_sent.get('tokens', [])[:12]} ... (total: {len(first_sent.get('tokens', []))})\n")

                print(f"5. STOP-WORD IDENTIFICATION:")
                stop_tokens = [t["text"] for t in first_sent.get("token_details", []) if t["is_stop"]]
                print(f"   Stop-words: {stop_tokens[:8]} ...")
                print(f"   Content tokens: {first_sent.get('content_tokens', [])[:8]} ...\n")

                print(f"6. LEMMATIZATION (First Sentence):")
                print(f"   {first_sent.get('lemmas', [])[:12]} ...\n")

                print(f"7. POS TAGS (Coarse Universal Dependencies):")
                print(f"   {first_sent.get('pos', [])[:12]} ...\n")

                print(f"8. NAMED ENTITIES (First Sentence):")
                ents = first_sent.get("entities", [])
                if ents:
                    for ent in ents[:5]:
                        print(f"   - {ent['text']} -> [{ent['label']}] (offset: {ent['start_char']}..{ent['end_char']})")
                else:
                    print("   - None in first sentence (article-level entities available)")

            print("-" * 70)
            if loaded >= count:
                break


def main() -> None:
    """CLI entrypoint for Phase 2 Preprocessing Pipeline."""
    parser = argparse.ArgumentParser(
        description="Run classical NLP preprocessing pipeline on CNN/DailyMail dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/raw/cnn_dailymail_test_1000.jsonl"),
        help="Input raw JSONL dataset path (default: dataset/raw/cnn_dailymail_test_1000.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/processed/cnn_dailymail_test_1000_processed.jsonl"),
        help="Output processed JSONL dataset path (default: dataset/processed/cnn_dailymail_test_1000_processed.jsonl).",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=Path("dataset/processed/preprocessing_stats.json"),
        help="Path to save preprocessing summary statistics JSON.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("docs/preprocessing_analysis.md"),
        help="Path to save preprocessing analysis Markdown report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of records to process (for quick testing).",
    )
    parser.add_argument(
        "--samples-to-show",
        type=int,
        default=5,
        help="Number of real sample records to display in terminal (default: 5).",
    )
    args = parser.parse_args()

    try:
        pipeline = PreprocessingPipeline(model_name="en_core_web_sm")
        processed_count, summary_stats = pipeline.process_file(
            input_path=args.input,
            output_path=args.output,
            limit=args.limit,
        )

        # Save statistics JSON
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        with args.stats_json.open("w", encoding="utf-8") as f_out:
            json.dump(summary_stats, f_out, indent=2)

        # Generate Markdown analysis report
        generate_preprocessing_report(
            stats=summary_stats,
            input_file=args.input,
            output_file=args.output,
            report_md_path=args.report_md,
        )

        print("\n============================================================")
        print("PREPROCESSING PIPELINE EXECUTION SUMMARY")
        print("============================================================")
        print(f"Total Articles Processed     : {summary_stats['total_articles']}")
        print(f"Mean Sentences per Article   : {summary_stats['sentences_per_article']['mean']}")
        print(f"Mean Tokens per Article      : {summary_stats['tokens_per_article']['mean']}")
        print(f"Mean Content Tokens/Article  : {summary_stats['content_tokens_per_article']['mean']}")
        print(f"Mean Stop Words per Article  : {summary_stats['stopwords_per_article']['mean']}")
        print(f"Mean Entities per Article    : {summary_stats['entities_per_article']['mean']}")
        print(f"Normalization Token Changes  : {summary_stats['percentage_tokens_changed_by_normalization']}%")
        print(f"Structured Stats File        : {args.stats_json.resolve()}")
        print(f"Markdown Analysis Report     : {args.report_md.resolve()}")
        print("============================================================\n")

        # Display real examples
        display_real_examples(args.output, count=args.samples_to_show)

    except Exception as err:
        print(f"\n[PIPELINE ERROR] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
