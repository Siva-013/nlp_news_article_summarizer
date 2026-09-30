"""
Central Evaluator Module
Part of Phase 6: Quantitative Evaluation and Algorithm Comparison

Orchestrates multi-algorithm ROUGE evaluation across the CNN/DailyMail dataset:
- Matches generated summaries from Frequency, TF-IDF, and TextRank by article ID.
- Computes ROUGE-1, ROUGE-2, and ROUGE-L against reference highlights.
- Calculates macro-averages and full statistical distributions (mean, median, std, min, max).
- Identifies per-article winners and computes win-tie distributions.
- Analyzes summary word length distributions.
- Emits structured output JSON, article-level JSONL, and visual figures.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from evaluation.comparison import (
    calculate_descriptive_stats,
    compute_per_article_winners,
    generate_plots,
)
from evaluation.rouge import evaluate_summary_pair


class Evaluator:
    """
    Evaluator for comparing extractive summarization models against reference highlights.
    """

    def __init__(
        self,
        dataset_path: str = "dataset/processed/cnn_dailymail_test_1000_processed.jsonl",
        frequency_path: str = "dataset/processed/frequency_summaries_1000.jsonl",
        tfidf_path: str = "dataset/processed/tfidf_summaries_1000.jsonl",
        textrank_path: str = "dataset/processed/textrank_summaries_1000.jsonl",
    ):
        self.dataset_path = Path(dataset_path)
        self.frequency_path = Path(frequency_path)
        self.tfidf_path = Path(tfidf_path)
        self.textrank_path = Path(textrank_path)

    @staticmethod
    def load_jsonl_by_id(file_path: Path, text_key: str = "summary") -> Dict[str, Dict[str, Any]]:
        """
        Load a JSONL file into an in-memory dictionary keyed by record ID.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        data_by_id: Dict[str, Dict[str, Any]] = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rec_id = record.get("id") or record.get("article_id")
                if not rec_id:
                    continue
                data_by_id[rec_id] = record
                
        return data_by_id

    def run_evaluation(
        self,
        output_results_path: str = "dataset/processed/evaluation_results.json",
        article_results_path: str = "dataset/processed/article_level_evaluation.jsonl",
        plot_dir: str = "docs/figures",
    ) -> Dict[str, Any]:
        """
        Execute full comparative evaluation across all three algorithms.
        """
        start_time = time.time()
        
        # 1. Load ground truth reference highlights
        print("Loading dataset references...")
        dataset_records = self.load_jsonl_by_id(self.dataset_path, text_key="highlights")
        
        # 2. Load summaries for each algorithm
        print("Loading Frequency summaries...")
        freq_records = self.load_jsonl_by_id(self.frequency_path)
        
        print("Loading TF-IDF summaries...")
        tfidf_records = self.load_jsonl_by_id(self.tfidf_path)
        
        print("Loading TextRank summaries...")
        textrank_records = self.load_jsonl_by_id(self.textrank_path)
        
        # 3. Find common article IDs
        common_ids = [
            rec_id for rec_id in dataset_records
            if rec_id in freq_records and rec_id in tfidf_records and rec_id in textrank_records
        ]
        
        if not common_ids:
            raise ValueError("No common article IDs found across dataset and summary files!")
            
        print(f"Evaluating {len(common_ids)} common articles across Frequency, TF-IDF, and TextRank...")
        
        # Prepare collectors
        article_evaluations: List[Dict[str, Any]] = []
        raw_scores: Dict[str, Dict[str, Dict[str, List[float]]]] = {
            algo: {
                metric: {"precision": [], "recall": [], "f1": []}
                for metric in ("rouge1", "rouge2", "rougeL")
            }
            for algo in ("frequency", "tfidf", "textrank")
        }
        
        word_counts: Dict[str, List[int]] = {
            "frequency": [],
            "tfidf": [],
            "textrank": [],
            "reference": [],
        }
        
        # 4. Evaluate each article
        out_art_path = Path(article_results_path)
        out_art_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_art_path, "w", encoding="utf-8") as art_out:
            for rec_id in tqdm(common_ids, desc="Computing ROUGE", unit="article"):
                ref_text = dataset_records[rec_id].get("highlights", "")
                word_counts["reference"].append(len(ref_text.split()))
                
                article_entry: Dict[str, Any] = {
                    "id": rec_id,
                    "reference_highlights": ref_text,
                    "reference_word_count": len(ref_text.split()),
                }
                
                algo_records = {
                    "frequency": freq_records[rec_id],
                    "tfidf": tfidf_records[rec_id],
                    "textrank": textrank_records[rec_id],
                }
                
                for algo, rec in algo_records.items():
                    summary_text = rec.get("summary", "")
                    w_count = len(summary_text.split())
                    word_counts[algo].append(w_count)
                    
                    scores = evaluate_summary_pair(summary_text, ref_text)
                    
                    article_entry[algo] = {
                        "summary": summary_text,
                        "word_count": w_count,
                        "rouge1": scores["rouge1"],
                        "rouge2": scores["rouge2"],
                        "rougeL": scores["rougeL"],
                    }
                    
                    for metric in ("rouge1", "rouge2", "rougeL"):
                        raw_scores[algo][metric]["precision"].append(scores[metric]["precision"])
                        raw_scores[algo][metric]["recall"].append(scores[metric]["recall"])
                        raw_scores[algo][metric]["f1"].append(scores[metric]["f1"])
                        
                article_evaluations.append(article_entry)
                art_out.write(json.dumps(article_entry, ensure_ascii=False) + "\n")
                
        # 5. Calculate descriptive statistics (Macro-averages)
        overall_results: Dict[str, Any] = {}
        for algo in ("frequency", "tfidf", "textrank"):
            overall_results[algo] = {}
            for metric in ("rouge1", "rouge2", "rougeL"):
                overall_results[algo][metric] = {
                    "precision": calculate_descriptive_stats(raw_scores[algo][metric]["precision"]),
                    "recall": calculate_descriptive_stats(raw_scores[algo][metric]["recall"]),
                    "f1": calculate_descriptive_stats(raw_scores[algo][metric]["f1"]),
                }
                
        # 6. Calculate summary length statistics
        length_stats: Dict[str, Dict[str, float]] = {
            k: calculate_descriptive_stats([float(x) for x in v])
            for k, v in word_counts.items()
        }
        
        # 7. Calculate per-article win statistics
        win_stats = compute_per_article_winners(article_evaluations)
        
        # 8. Determine winning algorithm per metric based strictly on mean F1
        best_algorithms: Dict[str, str] = {}
        for metric in ("rouge1", "rouge2", "rougeL"):
            f1_means = {
                algo: overall_results[algo][metric]["f1"]["mean"]
                for algo in ("frequency", "tfidf", "textrank")
            }
            best_algo = max(f1_means, key=f1_means.get)
            best_algorithms[metric] = best_algo
            
        elapsed_sec = round(time.time() - start_time, 2)
        
        # Build comprehensive results payload
        results_payload: Dict[str, Any] = {
            "metadata": {
                "num_articles_evaluated": len(common_ids),
                "dataset_file": str(self.dataset_path),
                "evaluation_elapsed_seconds": elapsed_sec,
                "algorithms_compared": ["frequency", "tfidf", "textrank"],
                "best_algorithms_by_f1": best_algorithms,
            },
            "overall_macro_results": overall_results,
            "per_article_win_statistics": win_stats,
            "summary_length_statistics": length_stats,
        }
        
        # Save JSON output
        out_res_path = Path(output_results_path)
        out_res_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_res_path, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2, ensure_ascii=False)
            
        # 9. Generate visualization plots
        print("Generating visualization charts...")
        generated_plots = generate_plots(overall_results, length_stats, output_dir=plot_dir)
        results_payload["metadata"]["plots_generated"] = generated_plots
        
        return results_payload


def print_terminal_summary(results: Dict[str, Any]) -> None:
    """Print the clean formatted terminal summary table required by Section 20."""
    meta = results["metadata"]
    macro = results["overall_macro_results"]
    best = meta["best_algorithms_by_f1"]
    
    n_arts = meta["num_articles_evaluated"]
    f_r1 = macro["frequency"]["rouge1"]["f1"]["mean"]
    f_r2 = macro["frequency"]["rouge2"]["f1"]["mean"]
    f_rl = macro["frequency"]["rougeL"]["f1"]["mean"]
    
    t_r1 = macro["tfidf"]["rouge1"]["f1"]["mean"]
    t_r2 = macro["tfidf"]["rouge2"]["f1"]["mean"]
    t_rl = macro["tfidf"]["rougeL"]["f1"]["mean"]
    
    tr_r1 = macro["textrank"]["rouge1"]["f1"]["mean"]
    tr_r2 = macro["textrank"]["rouge2"]["f1"]["mean"]
    tr_rl = macro["textrank"]["rougeL"]["f1"]["mean"]
    
    print("\n" + "=" * 40)
    print("PHASE 6 EVALUATION COMPLETE")
    print("=" * 40)
    print(f"\nArticles Evaluated: {n_arts}\n")
    print(f"{'':<14} {'R1-F1':<8} {'R2-F1':<8} {'RL-F1':<8}")
    print(f"{'Frequency':<14} {f_r1:<8.4f} {f_r2:<8.4f} {f_rl:<8.4f}")
    print(f"{'TF-IDF':<14} {t_r1:<8.4f} {t_r2:<8.4f} {t_rl:<8.4f}")
    print(f"{'TextRank':<14} {tr_r1:<8.4f} {tr_r2:<8.4f} {tr_rl:<8.4f}")
    print(f"\nBest ROUGE-1: {best['rouge1'].capitalize()}")
    print(f"Best ROUGE-2: {best['rouge2'].capitalize()}")
    print(f"Best ROUGE-L: {best['rougeL'].capitalize()}")
    print("\nDetailed results:")
    print("dataset/processed/evaluation_results.json")
    print("\nReport:")
    print("docs/evaluation_analysis.md")
    print("=" * 40 + "\n")


def main() -> None:
    """CLI entry point for Phase 6 evaluation."""
    parser = argparse.ArgumentParser(
        description="Phase 6: Quantitative Evaluation and Algorithm Comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/processed/cnn_dailymail_test_1000_processed.jsonl",
        help="Path to preprocessed dataset JSONL containing ground truth highlights",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="dataset/processed/frequency_summaries_1000.jsonl",
        help="Path to Phase 3 Frequency summaries JSONL",
    )
    parser.add_argument(
        "--tfidf",
        type=str,
        default="dataset/processed/tfidf_summaries_1000.jsonl",
        help="Path to Phase 4 TF-IDF summaries JSONL",
    )
    parser.add_argument(
        "--textrank",
        type=str,
        default="dataset/processed/textrank_summaries_1000.jsonl",
        help="Path to Phase 5 TextRank summaries JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/processed/evaluation_results.json",
        help="Destination path for overall evaluation metrics JSON",
    )
    parser.add_argument(
        "--article-output",
        type=str,
        default="dataset/processed/article_level_evaluation.jsonl",
        help="Destination path for per-article evaluation results JSONL",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="docs/figures",
        help="Directory to save generated comparison plots",
    )
    
    args = parser.parse_args()
    
    evaluator = Evaluator(
        dataset_path=args.dataset,
        frequency_path=args.frequency,
        tfidf_path=args.tfidf,
        textrank_path=args.textrank,
    )
    
    results = evaluator.run_evaluation(
        output_results_path=args.output,
        article_results_path=args.article_output,
        plot_dir=args.plot_dir,
    )
    
    print_terminal_summary(results)


if __name__ == "__main__":
    main()
