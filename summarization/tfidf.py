"""
TF-IDF-Based Extractive Summarization Module
Part of Phase 4: TF-IDF Summarization Baseline

This module implements a transparent, classical TF-IDF extractive summarizer
from first principles:
1. Calculates Term Frequency (TF) per sentence:
   TF(t, d) = count(t in d) / total number of content terms in d
2. Calculates Document Frequency (DF) across sentences of an article:
   DF(t) = number of sentences containing term t (at most once per sentence)
3. Calculates Inverse Document Frequency (IDF) within the article:
   IDF(t) = log(N / DF(t))
4. Calculates TF-IDF weight per term in each sentence:
   TFIDF(t, d) = TF(t, d) * IDF(t)
5. Scores each sentence using average TF-IDF importance:
   SentenceScore(S) = sum(TFIDF values of content terms in S) / number of content terms in S
6. Ranks sentences with deterministic tie-breaking (earlier sentences win ties).
7. Selects top-K highest scoring sentences.
8. Restores original chronological sentence order for coherent summary generation.
9. Returns exact, verbatim original sentences.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


class TFIDFSummarizer:
    """
    Classical TF-IDF extractive text summarizer.
    Calculates intra-article TF-IDF salience for content terms and scores
    sentences based on the average TF-IDF importance of their content words.
    """

    def __init__(self, pipeline: Optional[Any] = None, smooth_idf: bool = False):
        """
        Initialize the TFIDFSummarizer.

        Args:
            pipeline: Optional PreprocessingPipeline instance for summarizing raw strings.
            smooth_idf: If True, uses smoothed IDF: log(1 + N/DF). Default False: log(N/DF).
        """
        self._pipeline = pipeline
        self.smooth_idf = smooth_idf

    @property
    def pipeline(self) -> Any:
        """Lazy load PreprocessingPipeline if raw text processing is needed."""
        if self._pipeline is None:
            from preprocessing.pipeline import PreprocessingPipeline
            self._pipeline = PreprocessingPipeline(model_name="en_core_web_sm")
        return self._pipeline

    def calculate_tf(self, sentence_content_tokens: List[str]) -> Dict[str, float]:
        """
        Calculate Term Frequency (TF) for content words in a single sentence:
            TF(t, d) = count(t in d) / total number of content terms in d

        Args:
            sentence_content_tokens: List of content tokens in the sentence.

        Returns:
            Dictionary mapping lowercased content terms to their normalized TF values.
        """
        if not sentence_content_tokens:
            return {}

        cleaned = [w.strip().lower() for w in sentence_content_tokens if w and w.strip()]
        total_content = len(cleaned)
        if total_content == 0:
            return {}

        counts = Counter(cleaned)
        return {term: count / total_content for term, count in counts.items()}

    def calculate_df(self, sentences_content_tokens: List[List[str]]) -> Dict[str, int]:
        """
        Calculate Document Frequency (DF) across sentences within an article:
            DF(t) = number of sentences containing term t

        Each term is counted at most once per sentence.

        Args:
            sentences_content_tokens: List of content token lists, one list per sentence.

        Returns:
            Dictionary mapping lowercased content terms to their sentence occurrence count.
        """
        df: Counter = Counter()
        for sent_tokens in sentences_content_tokens:
            if not sent_tokens:
                continue
            unique_terms = set(w.strip().lower() for w in sent_tokens if w and w.strip())
            df.update(unique_terms)
        return dict(df)

    def calculate_idf(
        self,
        df: Dict[str, int],
        num_sentences: int,
        smooth: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Calculate Inverse Document Frequency (IDF) for all terms in an article:
            Standard (smooth=False): IDF(t) = log(N / DF(t))
            Smoothed (smooth=True):  IDF(t) = log(1 + (N / DF(t)))

        where N is the total number of sentences in the article.

        Args:
            df: Dictionary mapping terms to their sentence document frequencies.
            num_sentences: Total number of sentences N in the article.
            smooth: Optional override for smoothing behavior. Defaults to self.smooth_idf.

        Returns:
            Dictionary mapping terms to their non-negative IDF values.
        """
        if num_sentences <= 0 or not df:
            return {t: 0.0 for t in df}

        use_smooth = self.smooth_idf if smooth is None else smooth
        idf: Dict[str, float] = {}

        for term, doc_freq in df.items():
            if doc_freq <= 0:
                idf[term] = 0.0
                continue

            if use_smooth:
                val = math.log(1.0 + (num_sentences / doc_freq))
            else:
                val = math.log(num_sentences / doc_freq)

            # Prevent floating point precision negative values when df == N
            idf[term] = max(0.0, val)

        return idf

    def calculate_tfidf(
        self,
        tf: Dict[str, float],
        idf: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate TF-IDF weight for terms in a sentence:
            TFIDF(t, d) = TF(t, d) * IDF(t)

        Args:
            tf: Dictionary of term frequencies in the sentence.
            idf: Dictionary of inverse document frequencies in the article.

        Returns:
            Dictionary mapping terms to their TF-IDF values.
        """
        return {term: tf_val * idf.get(term, 0.0) for term, tf_val in tf.items()}

    def score_sentence(
        self,
        sentence_content_tokens: List[str],
        idf: Dict[str, float],
    ) -> float:
        """
        Calculate sentence importance score using length-normalized average TF-IDF:
            SentenceScore(S) = sum(TFIDF(t, S) for t in C(S)) / |C(S)|

        where C(S) is the list of content terms in sentence S.
        If a sentence contains zero content terms, score = 0.0.

        Args:
            sentence_content_tokens: List of content tokens in the sentence.
            idf: Document-level IDF mapping.

        Returns:
            Float sentence score representing average TF-IDF importance.
        """
        if not sentence_content_tokens:
            return 0.0

        cleaned = [w.strip().lower() for w in sentence_content_tokens if w and w.strip()]
        total_content = len(cleaned)
        if total_content == 0:
            return 0.0

        tf = self.calculate_tf(cleaned)
        tfidf = self.calculate_tfidf(tf, idf)

        # sum of TFIDF values of unique content terms in S divided by number of content terms
        score = sum(tfidf.values()) / total_content
        return round(score, 6)

    def score_sentences(
        self,
        processed_sentences: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, float]]:
        """
        Compute DF, IDF, and score all sentences for a given article record.

        Args:
            processed_sentences: List of processed sentence dictionaries from Phase 2.

        Returns:
            Tuple of (scored_sentence_candidates, df_dict, idf_dict).
        """
        num_sentences = len(processed_sentences)
        sentences_tokens: List[List[str]] = [
            sent.get("content_tokens", []) for sent in processed_sentences
        ]

        df = self.calculate_df(sentences_tokens)
        idf = self.calculate_idf(df, num_sentences)

        candidates: List[Dict[str, Any]] = []
        for sent in processed_sentences:
            s_idx = sent.get("sentence_index", 0)
            orig_text = sent.get("original_text", "")
            content_tokens = sent.get("content_tokens", [])

            s_score = self.score_sentence(content_tokens, idf)

            candidates.append(
                {
                    "sentence_index": s_idx,
                    "score": s_score,
                    "original_text": orig_text,
                    "text": orig_text,
                    "content_words": content_tokens,
                }
            )

        return candidates, df, idf

    def rank_sentences(
        self,
        processed_sentences: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, float]]:
        """
        Score and rank sentences in descending order of importance.

        Tie-breaking rule:
        1. Higher TF-IDF score wins.
        2. Sentences with content terms win over zero-content sentences.
        3. If scores tie, the earlier sentence (lower sentence_index) wins.

        Args:
            processed_sentences: List of sentence dictionaries from Phase 2 preprocessing.

        Returns:
            Tuple of (ranked_candidates, df_dict, idf_dict).
        """
        candidates, df, idf = self.score_sentences(processed_sentences)

        # Sort descending by score; sentences with content words preferred; earlier index wins ties
        candidates.sort(
            key=lambda item: (
                -item["score"],
                0 if len(item["content_words"]) > 0 else 1,
                item["sentence_index"],
            )
        )
        return candidates, df, idf

    def summarize_processed_record(
        self,
        record: Dict[str, Any],
        num_sentences: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate an extractive TF-IDF summary for a Phase 2 processed article record.

        Steps:
        1. Extract content tokens from each sentence.
        2. Calculate Document Frequency (DF) across sentences.
        3. Calculate Inverse Document Frequency (IDF) within the article.
        4. Calculate TF and TF-IDF for each sentence and score sentences.
        5. Rank sentences with deterministic tie-breaking.
        6. Select top-K highest scoring sentences.
        7. Restore original chronological sentence ordering.
        8. Return original verbatim sentences in the final summary.

        Args:
            record: Processed JSON record containing 'processed_sentences'.
            num_sentences: Number of sentences to extract (default: 3).

        Returns:
            Structured summary result dictionary.
        """
        processed_sents = record.get("processed_sentences", [])
        article_id = record.get("id", record.get("article_id", ""))

        if not processed_sents:
            return {
                "id": article_id,
                "article_id": article_id,
                "method": "tfidf",
                "num_sentences_requested": num_sentences,
                "requested_sentence_count": num_sentences,
                "num_sentences_selected": 0,
                "selected_sentence_count": 0,
                "summary": "",
                "selected_sentences": [],
                "highlights": record.get("highlights", ""),
                "top_terms": [],
                "top_words": [],
            }

        # Step 1-4: Score and rank sentences
        ranked_candidates, df, idf = self.rank_sentences(processed_sents)

        # Step 5: Top-K sentence selection (clamped to available sentence count)
        k = max(1, min(num_sentences, len(ranked_candidates)))
        top_candidates = ranked_candidates[:k]

        # Step 6: Restore original chronological order
        top_candidates.sort(key=lambda item: item["sentence_index"])

        # Step 7: Assemble summary using original unmodified text
        summary_sentences = [item["original_text"] for item in top_candidates]
        summary_text = " ".join(summary_sentences)

        # Calculate article-level TF-IDF for top terms explainability
        all_content_tokens: List[str] = []
        for s in processed_sents:
            all_content_tokens.extend(
                [w.strip().lower() for w in s.get("content_tokens", []) if w and w.strip()]
            )

        total_doc_tokens = len(all_content_tokens)
        doc_tf = Counter(all_content_tokens)

        top_terms: List[Dict[str, Any]] = []
        if total_doc_tokens > 0:
            doc_tfidf = {
                term: (count / total_doc_tokens) * idf.get(term, 0.0)
                for term, count in doc_tf.items()
            }
            sorted_terms = sorted(doc_tfidf.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            top_terms = [
                {
                    "term": term,
                    "tfidf": round(tfidf_val, 6),
                    "tf": round(doc_tf[term] / total_doc_tokens, 6),
                    "df": df.get(term, 0),
                    "idf": round(idf.get(term, 0.0), 6),
                }
                for term, tfidf_val in sorted_terms
            ]

        return {
            "id": article_id,
            "article_id": article_id,
            "method": "tfidf",
            "num_sentences_requested": num_sentences,
            "requested_sentence_count": num_sentences,
            "num_sentences_selected": len(top_candidates),
            "selected_sentence_count": len(top_candidates),
            "summary": summary_text,
            "selected_sentences": [
                {
                    "sentence_index": item["sentence_index"],
                    "score": round(item["score"], 4),
                    "original_text": item["original_text"],
                    "text": item["original_text"],
                    "content_words": item["content_words"],
                }
                for item in top_candidates
            ],
            "highlights": record.get("highlights", ""),
            "top_terms": top_terms,
            "top_words": top_terms,
        }

    def summarize_text(
        self,
        raw_text: str,
        num_sentences: int = 3,
        article_id: str = "custom_article",
    ) -> Dict[str, Any]:
        """
        Summarize raw, unprocessed text on-the-fly using Phase 2 PreprocessingPipeline.

        Args:
            raw_text: Raw string article text.
            num_sentences: Number of sentences to extract (default: 3).
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
        input_path: str,
        output_path: str,
        num_sentences: int = 3,
    ) -> Dict[str, Any]:
        """
        Process a preprocessed JSONL file and write TF-IDF summaries to output JSONL.

        Args:
            input_path: Path to Phase 2 processed JSONL file.
            output_path: Destination path for generated summaries JSONL.
            num_sentences: Target sentence count per summary (default: 3).

        Returns:
            Dictionary containing batch execution metrics.
        """
        in_file = Path(input_path)
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if not in_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Count total records for progress bar
        with open(in_file, "r", encoding="utf-8") as f:
            total_records = sum(1 for line in f if line.strip())

        start_time = time.time()
        processed_count = 0
        total_selected_sentences = 0
        summary_word_counts: List[int] = []

        with open(in_file, "r", encoding="utf-8") as in_f, open(
            out_file, "w", encoding="utf-8"
        ) as out_f:
            for line in tqdm(
                in_f, total=total_records, desc="TF-IDF Summarizing", unit="article"
            ):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                summary_result = self.summarize_processed_record(
                    record, num_sentences=num_sentences
                )

                out_f.write(json.dumps(summary_result, ensure_ascii=False) + "\n")

                processed_count += 1
                n_sel = summary_result["num_sentences_selected"]
                total_selected_sentences += n_sel
                words_in_summary = len(summary_result["summary"].split())
                summary_word_counts.append(words_in_summary)

        elapsed = time.time() - start_time
        avg_words = sum(summary_word_counts) / max(1, len(summary_word_counts))
        avg_sentences = total_selected_sentences / max(1, processed_count)
        throughput = processed_count / max(0.001, elapsed)

        metrics = {
            "total_articles": processed_count,
            "target_num_sentences": num_sentences,
            "total_sentences_selected": total_selected_sentences,
            "mean_sentences_selected": round(avg_sentences, 2),
            "mean_summary_word_count": round(avg_words, 2),
            "min_summary_word_count": min(summary_word_counts) if summary_word_counts else 0,
            "max_summary_word_count": max(summary_word_counts) if summary_word_counts else 0,
            "elapsed_seconds": round(elapsed, 2),
            "throughput_articles_per_second": round(throughput, 2),
            "output_file": str(out_file),
        }

        return metrics


def main() -> None:
    """CLI entry point for TF-IDF Extractive Summarization."""
    parser = argparse.ArgumentParser(
        description="Phase 4: TF-IDF Extractive News Article Summarizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m summarization.tfidf --article "Sentence one. Sentence two. Sentence three." --num-sentences 2
  python -m summarization.tfidf --article-file my.txt --num-sentences 3
  python -m summarization.tfidf --dataset dataset/processed/cnn_dailymail_test_1000_processed.jsonl --id f001ec5c4704938247d27a44948eebb37ae98d01 --num-sentences 3
  python -m summarization.tfidf --batch --input dataset/processed/cnn_dailymail_test_1000_processed.jsonl --output dataset/processed/tfidf_summaries_1000.jsonl --num-sentences 3
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--article", type=str, help="Raw article text string to summarize")
    group.add_argument("--article-file", type=str, help="Path to text file containing article")
    group.add_argument(
        "--dataset",
        type=str,
        help="Path to preprocessed JSONL dataset (requires --id)",
    )
    group.add_argument(
        "--batch",
        action="store_true",
        help="Batch summarize an entire preprocessed JSONL dataset",
    )

    parser.add_argument("--id", type=str, help="Article ID when using --dataset")
    parser.add_argument("--input", type=str, help="Input preprocessed JSONL path for --batch")
    parser.add_argument("--output", type=str, help="Output destination JSONL path for --batch")
    parser.add_argument(
        "--num-sentences",
        "--sentences",
        dest="num_sentences",
        type=int,
        default=3,
        help="Number of sentences in summary (default: 3)",
    )
    parser.add_argument(
        "--smooth-idf",
        action="store_true",
        help="Use smoothed IDF formulation log(1 + N/DF)",
    )

    args = parser.parse_args()
    summarizer = TFIDFSummarizer(smooth_idf=args.smooth_idf)

    if args.article:
        res = summarizer.summarize_text(args.article, num_sentences=args.num_sentences)
        print("\n=== TF-IDF EXTRACTIVE SUMMARY ===")
        print(f"Requested Sentences: {res['num_sentences_requested']}")
        print(f"Selected Sentences:  {res['num_sentences_selected']}")
        print(f"\nGenerated Summary:\n{res['summary']}")
        print("\nSelected Sentences Detail:")
        for s in res["selected_sentences"]:
            print(f"  [{s['sentence_index']}] (score: {s['score']:.4f}): {s['text']}")
        if res.get("top_terms"):
            print("\nTop Distinctive Terms (TF-IDF):")
            for t in res["top_terms"][:5]:
                print(f"  - {t['term']} (tfidf: {t['tfidf']:.4f}, df: {t['df']}, idf: {t['idf']:.4f})")

    elif args.article_file:
        file_path = Path(args.article_file)
        if not file_path.exists():
            print(f"Error: File not found: {args.article_file}", file=sys.stderr)
            sys.exit(1)
        raw_text = file_path.read_text(encoding="utf-8")
        res = summarizer.summarize_text(raw_text, num_sentences=args.num_sentences)
        print("\n=== TF-IDF EXTRACTIVE SUMMARY ===")
        print(f"File: {args.article_file}")
        print(f"Requested Sentences: {res['num_sentences_requested']}")
        print(f"Selected Sentences:  {res['num_sentences_selected']}")
        print(f"\nGenerated Summary:\n{res['summary']}")
        print("\nSelected Sentences Detail:")
        for s in res["selected_sentences"]:
            print(f"  [{s['sentence_index']}] (score: {s['score']:.4f}): {s['text']}")

    elif args.dataset:
        if not args.id:
            print("Error: --id is required when --dataset is specified.", file=sys.stderr)
            sys.exit(1)
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset not found: {args.dataset}", file=sys.stderr)
            sys.exit(1)

        found_record = None
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("id") == args.id:
                        found_record = rec
                        break

        if not found_record:
            print(f"Error: Record with id '{args.id}' not found.", file=sys.stderr)
            sys.exit(1)

        res = summarizer.summarize_processed_record(found_record, num_sentences=args.num_sentences)
        print("\n=== TF-IDF EXTRACTIVE SUMMARY ===")
        print(f"Article ID:          {res['article_id']}")
        print(f"Requested Sentences: {res['num_sentences_requested']}")
        print(f"Selected Sentences:  {res['num_sentences_selected']}")
        print(f"\nGenerated Summary:\n{res['summary']}")
        print("\nReference Highlights:")
        print(res.get("highlights", "(None)"))
        print("\nSelected Sentences Detail:")
        for s in res["selected_sentences"]:
            print(f"  [{s['sentence_index']}] (score: {s['score']:.4f}): {s['text']}")
        if res.get("top_terms"):
            print("\nTop Distinctive Terms (TF-IDF):")
            for t in res["top_terms"][:5]:
                print(f"  - {t['term']} (tfidf: {t['tfidf']:.4f}, df: {t['df']}, idf: {t['idf']:.4f})")

    elif args.batch:
        if not args.input or not args.output:
            print("Error: --batch requires both --input and --output arguments.", file=sys.stderr)
            sys.exit(1)
        metrics = summarizer.batch_summarize_file(
            input_path=args.input,
            output_path=args.output,
            num_sentences=args.num_sentences,
        )
        print("\n=== TF-IDF BATCH SUMMARY COMPLETE ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
