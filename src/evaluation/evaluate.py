"""
Model Evaluation & Comparison Utilities
=======================================
"""
import os, sys, json
import numpy as np
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.helpers import compute_all_metrics, DATA_DIR, logger


def load_comparison_results() -> Dict:
    """Load model comparison results from disk."""
    path = os.path.join(DATA_DIR, "model_comparison.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def get_best_model_name() -> str:
    """Return the name of the best performing model by RMSE."""
    results = load_comparison_results()
    if not results:
        return "lstm"  # Default fallback
    return min(results, key=lambda k: results[k]["RMSE"])


def format_metrics_table(results: Dict) -> str:
    """Pretty-print model comparison as a table string."""
    header = f"{'Model':<15} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}"
    lines = [header, "─" * len(header)]
    for name, metrics in results.items():
        lines.append(f"{name:<15} {metrics['RMSE']:>10.4f} {metrics['MAE']:>10.4f} {metrics['MAPE']:>9.2f}%")
    return "\n".join(lines)
