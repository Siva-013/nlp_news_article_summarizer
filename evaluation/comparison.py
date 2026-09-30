"""
Statistical Comparison and Visualization Module
Part of Phase 6: Quantitative Evaluation and Algorithm Comparison

Computes:
- Aggregate statistics (mean, median, std, min, max) for ROUGE scores
- Per-article win-loss-tie counts across algorithms
- Summary word length distributions
- Visualization plots via Matplotlib
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def calculate_descriptive_stats(values: List[float]) -> Dict[str, float]:
    """
    Calculate mean, median, sample standard deviation, min, and max for a list of numbers.
    
    Args:
        values: List of float values.
        
    Returns:
        Dictionary with statistical summaries.
    """
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    
    n = len(values)
    mean_val = sum(values) / n
    sorted_vals = sorted(values)
    
    # Median calculation
    mid = n // 2
    if n % 2 == 1:
        median_val = sorted_vals[mid]
    else:
        median_val = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        
    # Sample standard deviation
    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
        std_val = math.sqrt(variance)
    else:
        std_val = 0.0
        
    return {
        "mean": round(mean_val, 6),
        "median": round(median_val, 6),
        "std": round(std_val, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def compute_per_article_winners(
    article_evaluations: List[Dict[str, Any]],
    algorithms: Tuple[str, ...] = ("frequency", "tfidf", "textrank"),
    metrics: Tuple[str, ...] = ("rouge1", "rouge2", "rougeL"),
) -> Dict[str, Dict[str, Any]]:
    """
    Determine which algorithm achieved the highest F1 score on each article.
    
    Args:
        article_evaluations: List of per-article evaluation dictionaries.
        algorithms: Tuple of algorithm keys.
        metrics: Tuple of ROUGE metric keys.
        
    Returns:
        Dictionary mapping metric names to win counts and tie counts.
    """
    win_stats: Dict[str, Dict[str, Any]] = {}
    
    for metric in metrics:
        counts: Dict[str, int] = {algo: 0 for algo in algorithms}
        counts["tie"] = 0
        
        for article in article_evaluations:
            scores = {algo: article[algo][metric]["f1"] for algo in algorithms if algo in article}
            if not scores:
                continue
            max_score = max(scores.values())
            # Find all algorithms that achieved the maximum score (within floating epsilon)
            best_algos = [algo for algo, score in scores.items() if abs(score - max_score) < 1e-6]
            
            if len(best_algos) == 1:
                counts[best_algos[0]] += 1
            else:
                counts["tie"] += 1
                
        win_stats[metric] = counts
        
    return win_stats


def generate_plots(
    overall_results: Dict[str, Any],
    length_stats: Dict[str, Dict[str, float]],
    output_dir: str = "docs/figures",
) -> List[str]:
    """
    Generate evaluation comparison plots using Matplotlib and save to output_dir.
    
    Args:
        overall_results: Macro-average evaluation results dictionary.
        length_stats: Summary word count statistics dictionary.
        output_dir: Directory where figures should be saved.
        
    Returns:
        List of generated image file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    algos = ["frequency", "tfidf", "textrank"]
    labels = ["Frequency", "TF-IDF", "TextRank"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    
    # -------------------------------------------------------------
    # 1. ROUGE F1 Comparison Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.25
    x = np.arange(3)  # ROUGE-1, ROUGE-2, ROUGE-L
    
    for i, algo in enumerate(algos):
        f1_vals = [
            overall_results[algo]["rouge1"]["f1"]["mean"],
            overall_results[algo]["rouge2"]["f1"]["mean"],
            overall_results[algo]["rougeL"]["f1"]["mean"],
        ]
        ax.bar(x + (i - 1) * bar_width, f1_vals, width=bar_width, label=labels[i], color=colors[i], alpha=0.9)
        for idx, val in enumerate(f1_vals):
            ax.text(x[idx] + (i - 1) * bar_width, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
            
    ax.set_ylabel("Macro-Average F1 Score", fontsize=11)
    ax.set_title("ROUGE F1-Score Comparison Across Algorithms (1,000 Articles)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["ROUGE-1", "ROUGE-2", "ROUGE-L"], fontsize=10)
    ax.set_ylim(0, max(0.4, ax.get_ylim()[1] * 1.15))
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plot1_path = out_path / "rouge_comparison.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    generated_files.append(str(plot1_path))
    
    # -------------------------------------------------------------
    # 2. Precision vs. Recall Tradeoff Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^"]
    
    for i, algo in enumerate(algos):
        p_vals = [
            overall_results[algo]["rouge1"]["precision"]["mean"],
            overall_results[algo]["rouge2"]["precision"]["mean"],
            overall_results[algo]["rougeL"]["precision"]["mean"],
        ]
        r_vals = [
            overall_results[algo]["rouge1"]["recall"]["mean"],
            overall_results[algo]["rouge2"]["recall"]["mean"],
            overall_results[algo]["rougeL"]["recall"]["mean"],
        ]
        ax.scatter(r_vals, p_vals, color=colors[i], s=120, label=labels[i], marker=markers[i], edgecolor="black", zorder=4)
        for m_idx, metric_label in enumerate(["R-1", "R-2", "R-L"]):
            ax.annotate(
                f"{labels[i]} ({metric_label})",
                (r_vals[m_idx], p_vals[m_idx]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )
            
    ax.set_xlabel("Mean Recall", fontsize=11)
    ax.set_ylabel("Mean Precision", fontsize=11)
    ax.set_title("ROUGE Precision vs. Recall Tradeoff", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plot2_path = out_path / "rouge_precision_recall.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    generated_files.append(str(plot2_path))
    
    # -------------------------------------------------------------
    # 3. Summary Length Distribution Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    all_keys = ["frequency", "tfidf", "textrank", "reference"]
    all_labels = ["Frequency", "TF-IDF", "TextRank", "Reference"]
    all_colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
    
    means = [length_stats[k]["mean"] for k in all_keys if k in length_stats]
    stds = [length_stats[k]["std"] for k in all_keys if k in length_stats]
    medians = [length_stats[k]["median"] for k in all_keys if k in length_stats]
    
    x_pos = np.arange(len(all_labels))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=all_colors, alpha=0.85, edgecolor="black")
    
    for idx, (m_val, med_val) in enumerate(zip(means, medians)):
        ax.text(idx, m_val / 2, f"Mean: {m_val:.1f}\nMed: {med_val:.1f}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        
    ax.set_ylabel("Word Count", fontsize=11)
    ax.set_title("Summary Length Comparison (Words per Summary)", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_labels, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plot3_path = out_path / "summary_length_comparison.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    generated_files.append(str(plot3_path))
    
    return generated_files
