"""
Frequency-Based Extractive Summarization Module
Part of Phase 3: Frequency Summarization Baseline

This module implements a classical, frequency-based extractive summarizer
from first principles:
1. Extracts meaningful content tokens from preprocessed sentence representations.
2. Computes empirical word frequencies across the article.
3. Normalizes word frequencies: NF(w) = freq(w) / max(freq).
4. Calculates sentence scores as the average normalized frequency of its content words:
   Score(S) = sum(NF(w) for w in S) / len(content_words(S)).
5. Ranks sentences with deterministic tie-breaking (earlier sentences win ties).
6. Selects top-K highest scoring unique sentences.
7. Restores original chronological sentence order for coherent summary generation.
8. Outputs the exact, unmodified original sentences.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


class FrequencySummarizer:
    """
    Classical frequency-based extractive text summarizer.
    Calculates word salience using normalized term frequency and scores sentences
    based on the average importance of their constituent content words.
    """

    def __init__(self, pipeline: Optional[Any] = None):
        """
        Initialize the FrequencySummarizer.

        Args:
            pipeline: Optional PreprocessingPipeline instance for summarizing raw strings.
        """
        self._pipeline = pipeline

    @property
    def pipeline(self) -> Any:
        """Lazy load PreprocessingPipeline if raw text processing is needed."""
        if self._pipeline is None:
            from preprocessing.pipeline import PreprocessingPipeline
            self._pipeline = PreprocessingPipeline(model_name="en_core_web_sm")
        return self._pipeline

    def calculate_word_frequencies(self, content_words: List[str]) -> Dict[str, int]:
        """
        Calculate raw word frequency distribution across an article's content words.

        Args:
            content_words: Flat list of content tokens/lemmas from the entire document.

        Returns:
            Dictionary mapping lowercased content words to their integer occurrence counts.
        """
        if not content_words:
            return {}
        # Clean and count occurrences of alphanumeric content tokens
        cleaned_words = [w.strip().lower() for w in content_words if w and w.strip()]
        return dict(Counter(cleaned_words))

    def normalize_frequencies(self, word_frequencies: Dict[str, int]) -> Dict[str, float]:
        """
        Normalize word frequencies by dividing each term's frequency by the maximum
        frequency observed in the document:
            NF(w) = freq(w) / max(freq)

        The most frequent term receives a weight of 1.0; other terms receive values
        in the range (0.0, 1.0].

        Args:
            word_frequencies: Dictionary of raw word counts.

        Returns:
            Dictionary mapping words to normalized frequencies (0.0 to 1.0).
        """
        if not word_frequencies:
            return {}

        max_freq = max(word_frequencies.values())
        if max_freq <= 0:
            return {w: 0.0 for w in word_frequencies}

        return {w: round(count / max_freq, 6) for w, count in word_frequencies.items()}

    def score_sentence(
        self,
        sentence_content_tokens: List[str],
        normalized_frequencies: Dict[str, float],
    ) -> float:
        """
        Calculate sentence score as the average normalized frequency of its content words:
            Score(S) = sum(NF(w) for w in content_words(S)) / len(content_words(S))

        If a sentence contains zero content words (e.g., punctuation or stop words only),
        its score is 0.0.

        Args:
            sentence_content_tokens: List of content tokens in the sentence.
            normalized_frequencies: Document-level normalized word frequency map.

        Returns:
            Float sentence score representing average content word salience.
        """
        if not sentence_content_tokens:
            return 0.0

        content_words = [w.strip().lower() for w in sentence_content_tokens if w and w.strip()]
        if not content_words:
            return 0.0

        score_sum = sum(normalized_frequencies.get(w, 0.0) for w in content_words)
        return round(score_sum / len(content_words), 6)

    def rank_sentences(
        self,
        processed_sentences: List[Dict[str, Any]],
        normalized_frequencies: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Score and rank sentences in descending order of importance.

        Tie-breaking rule:
        If two sentences achieve the exact same frequency score, the earlier sentence
        in the document (smaller sentence_index) is given precedence to ensure stability
        and deterministic reproducibility.

        Args:
            processed_sentences: List of sentence dictionaries from Phase 2 preprocessing.
            normalized_frequencies: Normalized term frequencies.

        Returns:
            List of candidate sentence dicts sorted by (-score, sentence_index).
        """
        candidates: List[Dict[str, Any]] = []

        for sent in processed_sentences:
            s_idx = sent.get("sentence_index", 0)
            orig_text = sent.get("original_text", "")
            content_tokens = sent.get("content_tokens", [])

            s_score = self.score_sentence(content_tokens, normalized_frequencies)

            candidates.append(
                {
                    "sentence_index": s_idx,
                    "score": s_score,
                    "original_text": orig_text,
                    "content_words": content_tokens,
                }
            )

        # Sort descending by score; on tie, sort ascending by sentence_index
        candidates.sort(key=lambda item: (-item["score"], item["sentence_index"]))
        return candidates

    def summarize_processed_record(
        self,
        record: Dict[str, Any],
        num_sentences: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate an extractive summary for a Phase 2 processed article record.

        Steps:
        1. Extract all content tokens from the article to establish vocabulary frequencies.
        2. Compute raw and normalized word frequencies.
        3. Score each sentence using normalized word weights.
        4. Select top-K highest scoring sentences.
        5. Restore original chronological sentence ordering.
        6. Return original verbatim sentences in the final summary.

        Args:
            record: Processed JSON record containing 'processed_sentences'.
            num_sentences: Number of sentences to extract (default: 3).

        Returns:
            Structured summary result dictionary.
        """
        processed_sents = record.get("processed_sentences", [])
        if not processed_sents:
            return {
                "article_id": record.get("id", ""),
                "method": "frequency",
                "num_sentences_requested": num_sentences,
                "num_sentences_selected": 0,
                "summary": "",
                "selected_sentences": [],
                "highlights": record.get("highlights", ""),
                "top_words": [],
            }

        # Step 1: Collect all document-level content tokens
        all_content_tokens: List[str] = []
        for s in processed_sents:
            all_content_tokens.extend(s.get("content_tokens", []))

        # Step 2: Calculate raw and normalized frequencies
        word_freqs = self.calculate_word_frequencies(all_content_tokens)
        norm_freqs = self.normalize_frequencies(word_freqs)

        # Step 3: Score and rank sentences
        ranked_candidates = self.rank_sentences(processed_sents, norm_freqs)

        # Step 4: Top-K sentence selection (clamped to available sentence count)
        k = max(1, min(num_sentences, len(ranked_candidates)))
        top_candidates = ranked_candidates[:k]

        # Step 5: Restore original sentence order
        top_candidates.sort(key=lambda item: item["sentence_index"])

        # Step 6: Assemble summary using original unmodified text
        summary_sentences = [item["original_text"] for item in top_candidates]
        summary_text = " ".join(summary_sentences)

        # Build top words list for explainability
        sorted_top_words = [
            {"word": w, "frequency": word_freqs[w], "normalized_frequency": norm_freqs[w]}
            for w, _ in sorted(word_freqs.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        ]

        return {
            "article_id": record.get("id", ""),
            "method": "frequency",
            "num_sentences_requested": num_sentences,
            "num_sentences_selected": len(top_candidates),
            "summary": summary_text,
            "selected_sentences": [
                {
                    "sentence_index": item["sentence_index"],
                    "score": round(item["score"], 4),
                    "original_text": item["original_text"],
                    "content_words": item["content_words"],
                }
                for item in top_candidates
            ],
            "highlights": record.get("highlights", ""),
            "top_words": sorted_top_words,
        }

    def summarize_text(
        self,
        raw_text: str,
        num_sentences: int = 3,
        article_id: str = "custom_article",
    ) -> Dict[str, Any]:
        """
        Summarize a raw article string by running Phase 2 preprocessing on the fly.

        Args:
            raw_text: Unprocessed article string.
            num_sentences: Number of summary sentences to extract.
            article_id: Identifier tag for the article.

        Returns:
            Structured summary result dictionary.
        """
        processed_record = self.pipeline.process_article(
            raw_article=raw_text,
            record_id=article_id,
        )
        return self.summarize_processed_record(processed_record, num_sentences=num_sentences)

    def batch_summarize_file(
        self,
        input_jsonl: Path,
        output_jsonl: Path,
        num_sentences: int = 3,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Run the frequency summarizer over a processed JSONL file with streaming.

        Args:
            input_jsonl: Path to processed JSONL dataset (e.g. cnn_dailymail_test_1000_processed.jsonl).
            output_jsonl: Path to save generated summaries JSONL.
            num_sentences: Summary length in sentences.

        Returns:
            Tuple of (total_articles_processed, baseline_metrics_dict).
        """
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Input file not found at: {input_jsonl.resolve()}")

        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

        # Count total records for progress bar
        total_records = 0
        with input_jsonl.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                if line.strip():
                    total_records += 1

        print(f"Running Frequency Summarizer on {total_records} records...")
        print(f"Input:  {input_jsonl.resolve()}")
        print(f"Output: {output_jsonl.resolve()}")
        print(f"Requested summary length: {num_sentences} sentences\n")

        processed_count = 0
        orig_sent_counts: List[int] = []
        summary_word_counts: List[int] = []
        selected_sent_counts: List[int] = []

        with input_jsonl.open("r", encoding="utf-8") as f_in, output_jsonl.open("w", encoding="utf-8") as f_out:
            pbar = tqdm(total=total_records, desc="Summarizing articles", unit="articles")

            for line in f_in:
                line_str = line.strip()
                if not line_str:
                    continue

                record = json.loads(line_str)
                res = self.summarize_processed_record(record, num_sentences=num_sentences)

                # Save structured summary record
                output_record = {
                    "id": res["article_id"],
                    "method": "frequency",
                    "num_sentences_requested": res["num_sentences_requested"],
                    "num_sentences_selected": res["num_sentences_selected"],
                    "selected_sentences": res["selected_sentences"],
                    "summary": res["summary"],
                    "highlights": res["highlights"],
                    "top_words": res["top_words"][:5],
                }
                f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                processed_count += 1

                orig_sent_counts.append(len(record.get("processed_sentences", [])))
                selected_sent_counts.append(res["num_sentences_selected"])
                summary_word_counts.append(len(res["summary"].split()))

                pbar.update(1)

            pbar.close()

        import numpy as np

        metrics = {
            "total_articles_processed": processed_count,
            "num_sentences_requested": num_sentences,
            "original_sentences": {
                "mean": float(round(float(np.mean(orig_sent_counts)), 2)),
                "median": float(round(float(np.median(orig_sent_counts)), 2)),
                "min": int(np.min(orig_sent_counts)),
                "max": int(np.max(orig_sent_counts)),
            },
            "selected_sentences": {
                "mean": float(round(float(np.mean(selected_sent_counts)), 2)),
                "median": float(round(float(np.median(selected_sent_counts)), 2)),
                "min": int(np.min(selected_sent_counts)),
                "max": int(np.max(selected_sent_counts)),
            },
            "summary_word_count": {
                "mean": float(round(float(np.mean(summary_word_counts)), 2)),
                "median": float(round(float(np.median(summary_word_counts)), 2)),
                "min": int(np.min(summary_word_counts)),
                "max": int(np.max(summary_word_counts)),
            },
        }

        return processed_count, metrics


def generate_baseline_report(
    metrics: Dict[str, Any],
    input_file: Path,
    output_file: Path,
    report_md_path: Path,
    sample_summaries: List[Dict[str, Any]],
) -> None:
    """Generate Markdown report for the Phase 3 Frequency baseline experiment."""
    report_md_path.parent.mkdir(parents=True, exist_ok=True)

    samples_md = ""
    for idx, s in enumerate(sample_summaries, 1):
        samples_md += f"""
### Example {idx} (ID: `{s['article_id']}`)
- **Original Sentences**: {s['original_sentence_count']}
- **Requested Sentences**: {s['num_sentences_requested']} (Selected: {s['num_sentences_selected']})
- **Selected Indices & Scores**: {[(x['sentence_index'], x['score']) for x in s['selected_sentences']]}

**Generated Frequency Summary**:
> {s['summary']}

**Ground-Truth Reference Highlights**:
> {s['highlights']}

---
"""

    content = f"""# Phase 3: Frequency-Based Summarization Baseline Report

## 1. Executive Summary
This report presents the empirical results of the **Frequency-Based Extractive Summarizer**, serving as the foundational baseline for the summarization system. Sentences are ranked and selected strictly using normalized term frequency from Phase 2 content tokens, with original sentence wording and chronological order strictly preserved.

- **Algorithm**: Classical Word-Frequency Scoring
- **Input Corpus**: `{input_file.resolve()}`
- **Output Corpus**: `{output_file.resolve()}`
- **Articles Processed**: {metrics['total_articles_processed']}
- **Target Summary Length**: {metrics['num_sentences_requested']} sentences

---

## 2. Mathematical Formulation

### Word Frequency
$$\\text{{freq}}(w) = \\text{{total occurrences of content word }} w \\text{{ in article}}$$

### Normalized Frequency
$$\\text{{NF}}(w) = \\frac{{\\text{{freq}}(w)}}{{\\max_{{w'}} \\text{{freq}}(w')}}$$
where $\\max_{{w'}} \\text{{freq}}(w')$ is the maximum frequency of any content word in the article.

### Sentence Salience Score
$$\\text{{Score}}(S) = \\frac{{\\sum_{{w \\in C(S)}} \\text{{NF}}(w)}}{{|C(S)|}}$$
where $C(S)$ represents the list of meaningful content words in sentence $S$. If $|C(S)| = 0$, $\\text{{Score}}(S) = 0.0$.

### Selection & Reconstruction
1. Sentences are ranked descending by $\\text{{Score}}(S)$.
2. Ties are broken deterministically in favor of the earlier sentence index.
3. The top $K$ sentences are selected and then reordered by their original sentence index ascending:
$$\\text{{Order}}(S_i) < \\text{{Order}}(S_j) \\iff \\text{{Index}}(S_i) < \\text{{Index}}(S_j)$$

---

## 3. Quantitative Summary Statistics ($N = 1,000$ Articles)

| Metric | Mean | Median | Min | Max |
|:---|:---:|:---:|:---:|:---:|
| **Original Sentences per Article** | {metrics['original_sentences']['mean']} | {metrics['original_sentences']['median']} | {metrics['original_sentences']['min']} | {metrics['original_sentences']['max']} |
| **Selected Sentences per Summary** | {metrics['selected_sentences']['mean']} | {metrics['selected_sentences']['median']} | {metrics['selected_sentences']['min']} | {metrics['selected_sentences']['max']} |
| **Summary Word Count** | {metrics['summary_word_count']['mean']} | {metrics['summary_word_count']['median']} | {metrics['summary_word_count']['min']} | {metrics['summary_word_count']['max']} |

*Note: Phase 1 showed reference highlights average approximately 34.47 words (2.63 sentences). The 3-sentence frequency summaries average {metrics['summary_word_count']['mean']} words, providing a strong baseline for future ROUGE evaluation in Phase 6.*

---

## 4. Real CNN/DailyMail Summary Examples
{samples_md}

## 5. Architectural Properties & Known Baseline Limitations
1. **Extractive Fidelity**: The summary exclusively contains verbatim source sentences. Lemmatized or stop-word-stripped representations were used solely for mathematical scoring.
2. **Chronological Coherence**: Re-sorting selected top-K sentences by their original appearance order preserves discourse flow.
3. **Inability to Model Term Specificity**: Raw frequency does not downweight common terms that appear across all news articles (e.g. *said*, *told*, *year*). This will be resolved in Phase 4 using **TF-IDF**.
4. **Lack of Sentence Inter-Relationships**: Sentence scoring considers terms independently rather than measuring inter-sentence cross-similarity. This will be addressed in Phase 5 using **TextRank**.
"""
    with report_md_path.open("w", encoding="utf-8") as f_out:
        f_out.write(content)


def main() -> None:
    """CLI entrypoint for Frequency-based summarization."""
    parser = argparse.ArgumentParser(
        description="Frequency-based extractive summarization for news articles."
    )
    # Use Case A: Direct article text or file
    parser.add_argument("--article", type=str, default=None, help="Raw article text string.")
    parser.add_argument("--article-file", type=Path, default=None, help="Path to text file containing raw article.")

    # Use Case B: Processed dataset lookup
    parser.add_argument("--dataset", type=Path, default=None, help="Path to processed JSONL dataset.")
    parser.add_argument("--id", type=str, default=None, help="Article ID to summarize from dataset.")

    # Batch experiment options
    parser.add_argument("--batch", action="store_true", help="Run batch summarization over input dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/processed/cnn_dailymail_test_1000_processed.jsonl"),
        help="Input processed JSONL dataset for batch mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/processed/frequency_summaries_1000.jsonl"),
        help="Output JSONL dataset path for batch mode.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("docs/frequency_baseline_analysis.md"),
        help="Path to generate baseline Markdown report.",
    )
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=3,
        help="Number of sentences to extract for the summary (default: 3).",
    )
    args = parser.parse_args()

    summarizer = FrequencySummarizer()

    # Mode 1: Single article string
    if args.article:
        res = summarizer.summarize_text(args.article, num_sentences=args.num_sentences)
        print_single_summary_result(res)
        return

    # Mode 2: Single article file
    if args.article_file:
        if not args.article_file.exists():
            print(f"[ERROR] Article file not found: {args.article_file}", file=sys.stderr)
            sys.exit(1)
        raw_text = args.article_file.read_text(encoding="utf-8")
        res = summarizer.summarize_text(raw_text, num_sentences=args.num_sentences, article_id=args.article_file.stem)
        print_single_summary_result(res)
        return

    # Mode 3: Query specific ID from processed dataset
    if args.dataset and args.id:
        target_record = None
        with args.dataset.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                rec = json.loads(line)
                if rec.get("id") == args.id:
                    target_record = rec
                    break
        if not target_record:
            print(f"[ERROR] Article ID '{args.id}' not found in {args.dataset}", file=sys.stderr)
            sys.exit(1)

        res = summarizer.summarize_processed_record(target_record, num_sentences=args.num_sentences)
        print_single_summary_result(res)
        return

    # Mode 4: Batch summarization experiment (default if --batch or no specific article provided)
    if args.batch or (not args.article and not args.article_file and not args.id):
        count, metrics = summarizer.batch_summarize_file(
            input_jsonl=args.input,
            output_jsonl=args.output,
            num_sentences=args.num_sentences,
        )

        # Collect 5 real sample summaries for report
        samples: List[Dict[str, Any]] = []
        with args.output.open("r", encoding="utf-8") as f_out, args.input.open("r", encoding="utf-8") as f_in:
            for _ in range(5):
                out_line = f_out.readline()
                in_line = f_in.readline()
                if not out_line or not in_line:
                    break
                out_rec = json.loads(out_line)
                in_rec = json.loads(in_line)
                samples.append(
                    {
                        "article_id": out_rec["id"],
                        "original_sentence_count": len(in_rec.get("processed_sentences", [])),
                        "num_sentences_requested": out_rec["num_sentences_requested"],
                        "num_sentences_selected": out_rec["num_sentences_selected"],
                        "selected_sentences": out_rec["selected_sentences"],
                        "summary": out_rec["summary"],
                        "highlights": out_rec["highlights"],
                    }
                )

        generate_baseline_report(
            metrics=metrics,
            input_file=args.input,
            output_file=args.output,
            report_md_path=args.report_md,
            sample_summaries=samples,
        )

        print("\n============================================================")
        print("FREQUENCY-BASED SUMMARIZATION BASELINE SUMMARY")
        print("============================================================")
        print(f"Total Articles Processed     : {metrics['total_articles_processed']}")
        print(f"Requested Summary Length     : {metrics['num_sentences_requested']} sentences")
        print(f"Mean Original Sentences      : {metrics['original_sentences']['mean']}")
        print(f"Mean Selected Sentences      : {metrics['selected_sentences']['mean']}")
        print(f"Mean Summary Word Count      : {metrics['summary_word_count']['mean']}")
        print(f"Min / Max Summary Word Count : {metrics['summary_word_count']['min']} / {metrics['summary_word_count']['max']}")
        print(f"Output Dataset Saved To      : {args.output.resolve()}")
        print(f"Baseline Report Saved To     : {args.report_md.resolve()}")
        print("============================================================\n")


def print_single_summary_result(res: Dict[str, Any]) -> None:
    """Print readable single-article summary result to terminal."""
    print("\n" + "=" * 60)
    print(f"FREQUENCY EXTRACTIVE SUMMARY (ID: {res['article_id']})")
    print("=" * 60)
    print(f"Method                 : {res['method']}")
    print(f"Sentences Requested    : {res['num_sentences_requested']}")
    print(f"Sentences Selected     : {res['num_sentences_selected']}")
    print("-" * 60)
    print("TOP CONTENT WORDS & NORMALIZED SALIENCE:")
    for tw in res.get("top_words", [])[:8]:
        print(f"  - {tw['word']:<18} freq: {tw['frequency']:<4} NF: {tw['normalized_frequency']:.4f}")
    print("-" * 60)
    print("SELECTED ORIGINAL SENTENCES (ORIGINAL CHRONOLOGICAL ORDER):")
    for s in res.get("selected_sentences", []):
        print(f"  [{s['sentence_index']}] (Score: {s['score']:.4f}): {s['original_text']}")
    print("-" * 60)
    print(f"FINAL EXTRACTIVE SUMMARY:\n{res['summary']}")
    if res.get("highlights"):
        print("-" * 60)
        print(f"REFERENCE HIGHLIGHTS:\n{res['highlights']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
