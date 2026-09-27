"""
TextRank-Based Extractive Summarization Module
Part of Phase 5: TextRank Extractive Summarization

This module implements classical TextRank extractive summarization from first principles:
1. Constructs intra-article TF-IDF sparse vectors for each sentence using content tokens.
2. Computes symmetric pairwise Cosine Similarity between all sentence pairs.
3. Builds an undirected sentence graph represented as a weighted adjacency matrix:
   - W[i][j] = cosine_similarity(S_i, S_j) for i != j
   - W[i][i] = 0.0 (no self-loops)
4. Solves PageRank iteratively on the sentence graph:
   - PR(S_i) = (1 - d)/N + d * sum_{j != i} (W[j][i] / sum_k W[j][k] * PR(S_j))
   - Handles dangling nodes (sum_k W[j][k] == 0) by distributing their mass uniformly.
   - Iterates until convergence (sum |PR_new - PR_old| < tolerance) or max_iterations.
5. Ranks sentences in descending order of PageRank score with deterministic tie-breaking.
6. Selects top-K sentences and restores original chronological article order.
7. Emits exact, verbatim original sentence text.
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


class TextRankSummarizer:
    """
    Classical TextRank graph-based extractive text summarizer.
    Builds a sentence graph connected by TF-IDF cosine similarity edges and
    computes sentence centrality via iterative PageRank.
    """

    def __init__(
        self,
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        pipeline: Optional[Any] = None,
    ):
        """
        Initialize the TextRankSummarizer.

        Args:
            damping_factor: PageRank damping factor (default: 0.85).
            max_iterations: Maximum power iterations for PageRank (default: 100).
            tolerance: Convergence L1 norm difference threshold (default: 1e-6).
            pipeline: Optional PreprocessingPipeline instance for raw text processing.
        """
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self._pipeline = pipeline

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
            TF(t, S) = count(t in S) / total number of content terms in S

        Args:
            sentence_content_tokens: List of content tokens in the sentence.

        Returns:
            Dictionary mapping terms to their normalized TF values.
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
            sentences_content_tokens: List of content token lists, one per sentence.

        Returns:
            Dictionary mapping terms to their sentence occurrence count.
        """
        df: Counter = Counter()
        for sent_tokens in sentences_content_tokens:
            if not sent_tokens:
                continue
            unique_terms = set(w.strip().lower() for w in sent_tokens if w and w.strip())
            df.update(unique_terms)
        return dict(df)

    def calculate_idf(self, df: Dict[str, int], num_sentences: int) -> Dict[str, float]:
        """
        Calculate Inverse Document Frequency (IDF) for all terms in an article:
            IDF(t) = ln(N / DF(t))

        Args:
            df: Term to document frequency mapping.
            num_sentences: Total number of sentences N in the article.

        Returns:
            Dictionary mapping terms to their non-negative IDF values.
        """
        if num_sentences <= 0 or not df:
            return {t: 0.0 for t in df}

        idf: Dict[str, float] = {}
        for term, doc_freq in df.items():
            if doc_freq <= 0:
                idf[term] = 0.0
            else:
                idf[term] = max(0.0, math.log(num_sentences / doc_freq))
        return idf

    def build_sentence_vectors(
        self,
        sentences_content_tokens: List[List[str]],
    ) -> Tuple[List[Dict[str, float]], Dict[str, int], Dict[str, float]]:
        """
        Construct sparse TF-IDF vectors for all sentences in the article.

        Args:
            sentences_content_tokens: List of content token lists for each sentence.

        Returns:
            Tuple of (sentence_vectors, df, idf) where each sentence vector is a
            sparse Dict[str, float] mapping terms to TF-IDF weights.
        """
        num_sentences = len(sentences_content_tokens)
        df = self.calculate_df(sentences_content_tokens)
        idf = self.calculate_idf(df, num_sentences)

        vectors: List[Dict[str, float]] = []
        for sent_tokens in sentences_content_tokens:
            tf = self.calculate_tf(sent_tokens)
            tfidf_vec = {t: tf_val * idf.get(t, 0.0) for t, tf_val in tf.items()}
            vectors.append(tfidf_vec)

        return vectors, df, idf

    def cosine_similarity(
        self,
        vec_a: Dict[str, float],
        vec_b: Dict[str, float],
    ) -> float:
        """
        Calculate cosine similarity between two sparse term vectors:
            cosine_similarity(A, B) = (A . B) / (||A|| * ||B||)

        If either vector has zero magnitude, similarity is 0.0.

        Args:
            vec_a: Sparse TF-IDF dictionary for sentence A.
            vec_b: Sparse TF-IDF dictionary for sentence B.

        Returns:
            Float cosine similarity in range [0.0, 1.0].
        """
        if not vec_a or not vec_b:
            return 0.0

        # Dot product over shared terms (intersection)
        common_terms = set(vec_a.keys()) & set(vec_b.keys())
        if not common_terms:
            return 0.0

        dot_product = sum(vec_a[t] * vec_b[t] for t in common_terms)
        norm_a = math.sqrt(sum(val * val for val in vec_a.values()))
        norm_b = math.sqrt(sum(val * val for val in vec_b.values()))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        sim = dot_product / (norm_a * norm_b)
        # Numerical safeguard for precision
        return max(0.0, min(1.0, float(sim)))

    def build_similarity_matrix(
        self,
        sentence_vectors: List[Dict[str, float]],
    ) -> List[List[float]]:
        """
        Construct the weighted adjacency matrix W for the sentence graph:
            W[i][j] = cosine_similarity(S_i, S_j) for i != j
            W[i][i] = 0.0 (no self-loops)

        The resulting matrix is symmetric and non-negative.

        Args:
            sentence_vectors: List of sparse TF-IDF vectors for each sentence.

        Returns:
            N x N list of lists representing graph edge weights.
        """
        n = len(sentence_vectors)
        matrix: List[List[float]] = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                sim = self.cosine_similarity(sentence_vectors[i], sentence_vectors[j])
                matrix[i][j] = sim
                matrix[j][i] = sim

        return matrix

    def initialize_pagerank(self, num_sentences: int) -> List[float]:
        """
        Initialize uniform PageRank distribution:
            PR(S_i) = 1 / N

        Args:
            num_sentences: Total number of sentence nodes N.

        Returns:
            List of initial PageRank values summing to 1.0.
        """
        if num_sentences <= 0:
            return []
        initial_val = 1.0 / num_sentences
        return [initial_val] * num_sentences

    def calculate_pagerank(
        self,
        similarity_matrix: List[List[float]],
        damping_factor: Optional[float] = None,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> Tuple[List[float], int, bool, float]:
        """
        Compute PageRank scores iteratively from first principles:
            PR(S_i) = (1 - d)/N + d * sum_{j != i} [ (W[j][i] / sum_k W[j][k]) * PR(S_j) ]

        Handles dangling nodes (nodes with sum_k W[j][k] == 0) by distributing
        their PageRank mass uniformly across all nodes:
            dangling_mass / N

        Args:
            similarity_matrix: N x N weighted adjacency matrix.
            damping_factor: Optional override for damping factor d.
            max_iterations: Optional override for iteration cap.
            tolerance: Optional override for convergence threshold.

        Returns:
            Tuple of (pagerank_scores, iterations_run, converged_flag, final_difference).
        """
        n = len(similarity_matrix)
        if n == 0:
            return [], 0, True, 0.0
        if n == 1:
            return [1.0], 1, True, 0.0

        d = self.damping_factor if damping_factor is None else damping_factor
        max_iter = self.max_iterations if max_iterations is None else max_iterations
        tol = self.tolerance if tolerance is None else tolerance

        # Precompute row sums (outgoing edge weight sum from node j)
        row_sums = [sum(similarity_matrix[j]) for j in range(n)]
        dangling_nodes = [j for j in range(n) if row_sums[j] == 0.0]

        # Initialize PR: PR(S_i) = 1 / N
        scores = self.initialize_pagerank(n)

        teleport = (1.0 - d) / n
        converged = False
        iteration = 0
        final_diff = 0.0

        for it in range(1, max_iter + 1):
            iteration = it
            new_scores = [0.0] * n

            # Calculate total mass from dangling nodes to distribute uniformly
            dangling_mass = sum(scores[j] for j in dangling_nodes)
            dangling_contrib = (d * dangling_mass) / n

            # Compute incoming edge contributions
            for i in range(n):
                incoming_sum = 0.0
                for j in range(n):
                    if row_sums[j] > 0.0:
                        w_ji = similarity_matrix[j][i]
                        if w_ji > 0.0:
                            incoming_sum += (w_ji / row_sums[j]) * scores[j]

                new_scores[i] = teleport + (d * incoming_sum) + dangling_contrib

            # Check L1 convergence: sum |new_score - old_score|
            final_diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            scores = new_scores

            if final_diff < tol:
                converged = True
                break

        return scores, iteration, converged, final_diff

    def rank_sentences(
        self,
        processed_sentences: List[Dict[str, Any]],
        damping_factor: Optional[float] = None,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[List[float]], Dict[str, Any]]:
        """
        Construct graph, solve PageRank, and rank sentences in descending order of importance.

        Tie-breaking rule:
        1. Higher PageRank score wins.
        2. If scores tie, earlier sentence in document (smaller sentence_index) wins.

        Args:
            processed_sentences: List of Phase 2 processed sentence dicts.
            damping_factor: Optional override for d.
            max_iterations: Optional override for max iterations.
            tolerance: Optional override for tolerance.

        Returns:
            Tuple of (ranked_candidates, similarity_matrix, pagerank_metadata).
        """
        n = len(processed_sentences)
        if n == 0:
            return [], [], {"iterations": 0, "converged": True, "final_difference": 0.0}

        sentences_tokens = [s.get("content_tokens", []) for s in processed_sentences]
        sentence_vectors, df, idf = self.build_sentence_vectors(sentences_tokens)
        sim_matrix = self.build_similarity_matrix(sentence_vectors)

        scores, iters, converged, diff = self.calculate_pagerank(
            sim_matrix,
            damping_factor=damping_factor,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

        candidates: List[Dict[str, Any]] = []
        for idx, sent in enumerate(processed_sentences):
            s_idx = sent.get("sentence_index", idx)
            orig_text = sent.get("original_text", "")
            content_tokens = sent.get("content_tokens", [])
            score = scores[idx] if idx < len(scores) else 0.0

            candidates.append(
                {
                    "sentence_index": s_idx,
                    "textrank_score": score,
                    "score": score,
                    "original_text": orig_text,
                    "text": orig_text,
                    "content_words": content_tokens,
                }
            )

        # Deterministic ranking: score descending, sentence_index ascending on tie
        candidates.sort(key=lambda item: (-item["score"], item["sentence_index"]))

        meta = {
            "number_of_sentences": n,
            "damping_factor": self.damping_factor if damping_factor is None else damping_factor,
            "max_iterations": self.max_iterations if max_iterations is None else max_iterations,
            "tolerance": self.tolerance if tolerance is None else tolerance,
            "iterations": iters,
            "converged": converged,
            "final_difference": round(diff, 8),
        }

        return candidates, sim_matrix, meta

    def select_top_sentences(
        self,
        ranked_candidates: List[Dict[str, Any]],
        num_sentences: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Select top-K highest scoring sentences and restore original chronological order.

        Args:
            ranked_candidates: Candidates sorted by (-score, sentence_index).
            num_sentences: Number of sentences to select (K).

        Returns:
            List of selected sentence dicts sorted by sentence_index ascending.
        """
        if not ranked_candidates:
            return []

        k = max(1, min(num_sentences, len(ranked_candidates)))
        selected = list(ranked_candidates[:k])
        # Restore chronological order
        selected.sort(key=lambda item: item["sentence_index"])
        return selected

    def summarize_processed_record(
        self,
        record: Dict[str, Any],
        num_sentences: int = 3,
        damping_factor: Optional[float] = None,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate an extractive TextRank summary for a Phase 2 processed article record.

        Args:
            record: Processed JSON record containing 'processed_sentences'.
            num_sentences: Target summary sentence count (default: 3).
            damping_factor: Optional PageRank damping factor.
            max_iterations: Optional PageRank iteration limit.
            tolerance: Optional PageRank convergence tolerance.

        Returns:
            Structured summary result dictionary.
        """
        processed_sents = record.get("processed_sentences", [])
        article_id = record.get("id", record.get("article_id", ""))

        d = self.damping_factor if damping_factor is None else damping_factor
        m_iter = self.max_iterations if max_iterations is None else max_iterations
        tol = self.tolerance if tolerance is None else tolerance

        if not processed_sents:
            return {
                "id": article_id,
                "article_id": article_id,
                "method": "textrank",
                "num_sentences_requested": num_sentences,
                "requested_sentence_count": num_sentences,
                "num_sentences_selected": 0,
                "selected_sentence_count": 0,
                "summary": "",
                "selected_sentences": [],
                "ranking": [],
                "highlights": record.get("highlights", ""),
                "configuration": {
                    "num_sentences": num_sentences,
                    "damping_factor": d,
                    "max_iterations": m_iter,
                    "tolerance": tol,
                },
                "metadata": {
                    "number_of_sentences": 0,
                    "iterations": 0,
                    "converged": True,
                    "final_difference": 0.0,
                },
            }

        ranked_candidates, sim_matrix, meta = self.rank_sentences(
            processed_sents,
            damping_factor=d,
            max_iterations=m_iter,
            tolerance=tol,
        )

        selected_candidates = self.select_top_sentences(ranked_candidates, num_sentences=num_sentences)

        summary_text = " ".join(item["original_text"] for item in selected_candidates)

        ranking_output = [
            {
                "sentence_index": item["sentence_index"],
                "textrank_score": round(item["score"], 6),
            }
            for item in ranked_candidates
        ]

        return {
            "id": article_id,
            "article_id": article_id,
            "method": "textrank",
            "num_sentences_requested": num_sentences,
            "requested_sentence_count": num_sentences,
            "num_sentences_selected": len(selected_candidates),
            "selected_sentence_count": len(selected_candidates),
            "summary": summary_text,
            "selected_sentences": [
                {
                    "sentence_index": item["sentence_index"],
                    "textrank_score": round(item["score"], 6),
                    "score": round(item["score"], 6),
                    "original_text": item["original_text"],
                    "text": item["original_text"],
                    "content_words": item["content_words"],
                }
                for item in selected_candidates
            ],
            "ranking": ranking_output,
            "highlights": record.get("highlights", ""),
            "configuration": {
                "num_sentences": num_sentences,
                "damping_factor": d,
                "max_iterations": m_iter,
                "tolerance": tol,
            },
            "metadata": meta,
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
            raw_text: Raw article text string.
            num_sentences: Target sentence count.
            article_id: Identifier tag for article.

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
        Process a preprocessed JSONL file and write TextRank summaries to output JSONL.

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

        with open(in_file, "r", encoding="utf-8") as f:
            total_records = sum(1 for line in f if line.strip())

        start_time = time.time()
        processed_count = 0
        total_selected_sentences = 0
        summary_word_counts: List[int] = []
        iteration_counts: List[int] = []

        with open(in_file, "r", encoding="utf-8") as in_f, open(
            out_file, "w", encoding="utf-8"
        ) as out_f:
            for line in tqdm(
                in_f, total=total_records, desc="TextRank Summarizing", unit="article"
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
                iteration_counts.append(summary_result["metadata"]["iterations"])

        elapsed = time.time() - start_time
        avg_words = sum(summary_word_counts) / max(1, len(summary_word_counts))
        avg_sentences = total_selected_sentences / max(1, processed_count)
        avg_iters = sum(iteration_counts) / max(1, len(iteration_counts))
        throughput = processed_count / max(0.001, elapsed)

        metrics = {
            "total_articles": processed_count,
            "target_num_sentences": num_sentences,
            "total_sentences_selected": total_selected_sentences,
            "mean_sentences_selected": round(avg_sentences, 2),
            "mean_summary_word_count": round(avg_words, 2),
            "min_summary_word_count": min(summary_word_counts) if summary_word_counts else 0,
            "max_summary_word_count": max(summary_word_counts) if summary_word_counts else 0,
            "mean_pagerank_iterations": round(avg_iters, 2),
            "elapsed_seconds": round(elapsed, 2),
            "throughput_articles_per_second": round(throughput, 2),
            "output_file": str(out_file),
        }

        return metrics


def main() -> None:
    """CLI entry point for TextRank Extractive Summarization."""
    parser = argparse.ArgumentParser(
        description="Phase 5: TextRank Extractive News Article Summarizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m summarization.textrank --article "Sentence one. Sentence two. Sentence three." --num-sentences 2
  python -m summarization.textrank --article-file my.txt --num-sentences 3
  python -m summarization.textrank --dataset dataset/processed/cnn_dailymail_test_1000_processed.jsonl --id f001ec5c4704938247d27a44948eebb37ae98d01 --num-sentences 3
  python -m summarization.textrank --dataset dataset/processed/cnn_dailymail_test_1000_processed.jsonl --num-sentences 3
  python -m summarization.textrank --batch --input dataset/processed/cnn_dailymail_test_1000_processed.jsonl --output dataset/processed/textrank_summaries_1000.jsonl --num-sentences 3
        """,
    )

    # Arguments
    parser.add_argument("--article", type=str, help="Raw article text string to summarize")
    parser.add_argument("--article-file", type=str, help="Path to text file containing article")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to preprocessed JSONL dataset (runs single article with --id, or batch without --id)",
    )
    parser.add_argument("--id", type=str, help="Article ID when using --dataset for a single article")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch summarize an entire preprocessed JSONL dataset (uses --input and --output)",
    )
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
        "--damping-factor",
        type=float,
        default=0.85,
        help="PageRank damping factor (default: 0.85)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum PageRank iterations (default: 100)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="PageRank convergence tolerance (default: 1e-6)",
    )

    args = parser.parse_args()

    # Validate mode
    if not (args.article or args.article_file or args.dataset or args.batch):
        parser.error("Must specify one of --article, --article-file, --dataset, or --batch.")

    summarizer = TextRankSummarizer(
        damping_factor=args.damping_factor,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
    )

    if args.article:
        res = summarizer.summarize_text(args.article, num_sentences=args.num_sentences)
        print("\n=== TEXTRANK EXTRACTIVE SUMMARY ===")
        print(f"Requested Sentences: {res['num_sentences_requested']}")
        print(f"Selected Sentences:  {res['num_sentences_selected']}")
        print(f"PageRank Iterations: {res['metadata']['iterations']} (converged: {res['metadata']['converged']})")
        print(f"\nGenerated Summary:\n{res['summary']}")
        print("\nSelected Sentences Detail:")
        for s in res["selected_sentences"]:
            print(f"  [{s['sentence_index']}] (TextRank: {s['textrank_score']:.6f}): {s['text']}")

    elif args.article_file:
        file_path = Path(args.article_file)
        if not file_path.exists():
            print(f"Error: File not found: {args.article_file}", file=sys.stderr)
            sys.exit(1)
        raw_text = file_path.read_text(encoding="utf-8")
        res = summarizer.summarize_text(raw_text, num_sentences=args.num_sentences)
        print("\n=== TEXTRANK EXTRACTIVE SUMMARY ===")
        print(f"File: {args.article_file}")
        print(f"Requested Sentences: {res['num_sentences_requested']}")
        print(f"Selected Sentences:  {res['num_sentences_selected']}")
        print(f"PageRank Iterations: {res['metadata']['iterations']} (converged: {res['metadata']['converged']})")
        print(f"\nGenerated Summary:\n{res['summary']}")
        print("\nSelected Sentences Detail:")
        for s in res["selected_sentences"]:
            print(f"  [{s['sentence_index']}] (TextRank: {s['textrank_score']:.6f}): {s['text']}")

    elif args.dataset and args.id:
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
        print("\n=== TEXTRANK EXTRACTIVE SUMMARY ===")
        print(f"Article ID:          {res['article_id']}")
        print(f"Requested Sentences: {res['num_sentences_requested']}")
        print(f"Selected Sentences:  {res['num_sentences_selected']}")
        print(f"PageRank Iterations: {res['metadata']['iterations']} (converged: {res['metadata']['converged']})")
        print(f"\nGenerated Summary:\n{res['summary']}")
        print("\nReference Highlights:")
        print(res.get("highlights", "(None)"))
        print("\nSelected Sentences Detail:")
        for s in res["selected_sentences"]:
            print(f"  [{s['sentence_index']}] (TextRank: {s['textrank_score']:.6f}): {s['text']}")

    elif args.dataset and not args.id:
        # Full 1000 article baseline mode directly via --dataset
        out_path = "dataset/processed/textrank_summaries_1000.jsonl"
        print(f"Processing full dataset '{args.dataset}' -> '{out_path}'")
        metrics = summarizer.batch_summarize_file(
            input_path=args.dataset,
            output_path=out_path,
            num_sentences=args.num_sentences,
        )
        print("\n=== TEXTRANK BATCH SUMMARY COMPLETE ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    elif args.batch:
        if not args.input or not args.output:
            print("Error: --batch requires both --input and --output arguments.", file=sys.stderr)
            sys.exit(1)
        metrics = summarizer.batch_summarize_file(
            input_path=args.input,
            output_path=args.output,
            num_sentences=args.num_sentences,
        )
        print("\n=== TEXTRANK BATCH SUMMARY COMPLETE ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
