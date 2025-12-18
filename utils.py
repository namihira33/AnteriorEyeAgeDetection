"""
Utilities for saving and loading results
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def save_results_json(
    results: Dict[str, Any],
    filepath: str,
    indent: int = 2
):
    """
    Save results to JSON file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=indent, cls=NumpyEncoder)
    
    print(f"Results saved to {filepath}")


def load_results_json(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_cv_summary_csv(
    all_model_results: Dict[str, Dict[str, Any]],
    filepath: str
):
    """
    Save cross-validation summary for all models to CSV.
    
    Creates a summary table with model name and metrics.
    """
    rows = []
    
    for model_name, results in all_model_results.items():
        row = {
            "model": model_name,
            "mean_val_mae": results["mean_val_mae"],
            "std_val_mae": results["std_val_mae"],
            "mean_val_rmse": results["mean_val_rmse"],
            "std_val_rmse": results["std_val_rmse"]
        }
        
        # Add per-fold results
        for i, fold_result in enumerate(results["folds"]):
            row[f"fold{i+1}_mae"] = fold_result["final_val_mae"]
            row[f"fold{i+1}_rmse"] = fold_result["final_val_rmse"]
            row[f"fold{i+1}_lr"] = fold_result["best_lr"]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values("mean_val_mae")
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    
    print(f"CV summary saved to {filepath}")
    return df


def save_test_results_csv(
    all_model_results: Dict[str, Dict[str, Any]],
    filepath: str
):
    """
    Save test set results for all models to CSV.
    """
    rows = []
    
    for model_name, results in all_model_results.items():
        metrics = results["metrics"]
        row = {
            "model": model_name,
            "mae": metrics["mae"]["value"],
            "mae_ci_lower": metrics["mae"]["ci_lower"],
            "mae_ci_upper": metrics["mae"]["ci_upper"],
            "rmse": metrics["rmse"]["value"],
            "rmse_ci_lower": metrics["rmse"]["ci_lower"],
            "rmse_ci_upper": metrics["rmse"]["ci_upper"],
            "pearson_r": metrics["pearson_r"]["value"],
            "pearson_r_ci_lower": metrics["pearson_r"]["ci_lower"],
            "pearson_r_ci_upper": metrics["pearson_r"]["ci_upper"]
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values("mae")
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    
    print(f"Test results saved to {filepath}")
    return df


def save_predictions_csv(
    predictions: np.ndarray,
    targets: np.ndarray,
    image_paths: List[str],
    filepath: str
):
    """
    Save individual predictions to CSV.
    """
    df = pd.DataFrame({
        "image_path": image_paths,
        "true_age": targets,
        "predicted_age": predictions,
        "error": predictions - targets,
        "abs_error": np.abs(predictions - targets)
    })
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    
    print(f"Predictions saved to {filepath}")
    return df


def create_experiment_log(
    config,
    start_time: datetime,
    end_time: datetime
) -> Dict[str, Any]:
    """
    Create experiment log with configuration and timing info.
    """
    log = {
        "experiment_info": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds()
        },
        "config": {
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "n_folds": config.n_folds,
            "max_epochs": config.max_epochs,
            "early_stopping_patience": config.early_stopping_patience,
            "optimizer": config.optimizer,
            "weight_decay": config.weight_decay,
            "lr_candidates": config.lr_candidates,
            "models": config.models,
            "seed": config.seed
        }
    }
    
    return log


def save_experiment_log(
    log: Dict[str, Any],
    filepath: str
):
    """
    Save experiment log to JSON.
    """
    save_results_json(log, filepath)


def print_summary_table(df: pd.DataFrame, title: str = "Results Summary"):
    """
    Print formatted summary table.
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")


def generate_latex_table(
    test_results: Dict[str, Dict[str, Any]],
    cv_results: Dict[str, Dict[str, Any]] = None
) -> str:
    """
    Generate LaTeX table for publication.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Model Comparison Results}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Model & MAE (95\% CI) & RMSE (95\% CI) & Pearson $r$ (95\% CI) \\",
        r"\hline"
    ]
    
    for model_name, results in test_results.items():
        metrics = results["metrics"]
        
        mae_str = f"{metrics['mae']['value']:.3f} ({metrics['mae']['ci_lower']:.3f}-{metrics['mae']['ci_upper']:.3f})"
        rmse_str = f"{metrics['rmse']['value']:.3f} ({metrics['rmse']['ci_lower']:.3f}-{metrics['rmse']['ci_upper']:.3f})"
        r_str = f"{metrics['pearson_r']['value']:.3f} ({metrics['pearson_r']['ci_lower']:.3f}-{metrics['pearson_r']['ci_upper']:.3f})"
        
        lines.append(f"{model_name} & {mae_str} & {rmse_str} & {r_str} \\\\")
    
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    return "\n".join(lines)
