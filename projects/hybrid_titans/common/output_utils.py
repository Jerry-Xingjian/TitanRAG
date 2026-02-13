"""
Output and persistence utilities for evaluation results.
"""

import os
import json
from datetime import datetime


def print_summary_bar(results_dict, show_counts=True):
    """Print a bar-chart summary of retriever results.

    Args:
        results_dict: Dict of {retriever_name: accuracy_float_or_stats_dict}
        show_counts: If True, show correct/total counts alongside accuracy
    """
    for name, stats in results_dict.items():
        if isinstance(stats, dict) and stats.get("total", 0) > 0:
            accuracy = stats["correct"] / stats["total"] * 100
            bar = "█" * int(accuracy / 10) + "░" * (10 - int(accuracy / 10))
            if show_counts:
                print(f"  {name:12s}: {bar} {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
            else:
                print(f"  {name:12s}: {bar} {accuracy:.1f}%")
        elif isinstance(stats, (int, float)):
            accuracy = stats
            bar = "█" * int(accuracy / 10) + "░" * (10 - int(accuracy / 10))
            print(f"  {name:12s}: {bar} {accuracy:.1f}%")


def save_results(dataset_name, config, summary, details=None):
    """Save evaluation results to evaluations/ directory as JSON.

    Args:
        dataset_name: e.g. 'hotpotqa', 'squad', 'multidoc', 'essay_climate'
        config: dict of run configuration (epochs, topk, titles, etc.)
        summary: dict of {retriever_name: accuracy_or_stats}
        details: optional list of per-question result dicts
    """
    eval_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), 'evaluations')
    os.makedirs(eval_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{dataset_name}_{timestamp}.json"

    # Show relative path for portability
    try:
        rel_path = os.path.relpath(os.path.join(eval_dir, filename))
    except ValueError:
        rel_path = os.path.join(eval_dir, filename)

    # Normalize summary to {name: {accuracy, correct, total}}
    norm_summary = {}
    for name, stats in summary.items():
        if isinstance(stats, dict) and "total" in stats:
            total = stats["total"]
            correct = stats["correct"]
            norm_summary[name] = {
                "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
                "correct": correct, "total": total
            }
        elif isinstance(stats, (int, float)):
            norm_summary[name] = {"accuracy": round(stats, 1)}

    data = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "summary": norm_summary,
    }
    if details:
        data["details"] = details

    filepath = os.path.join(eval_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {rel_path}")
