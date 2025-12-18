"""
Evaluation Metrics for Age Estimation
- MAE, RMSE, Pearson r
- Bootstrap Confidence Intervals
- Detailed Statistics and Age-Stratified Analysis
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, List, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def calculate_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(predictions - targets))


def calculate_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def calculate_pearson_r(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Pearson correlation coefficient.
    
    Returns:
        r: Correlation coefficient
        p_value: Two-tailed p-value
    """
    r, p_value = stats.pearsonr(predictions, targets)
    return r, p_value


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Returns:
        Dictionary with MAE, RMSE, Pearson r, and p-value
    """
    mae = calculate_mae(predictions, targets)
    rmse = calculate_rmse(predictions, targets)
    r, p_value = calculate_pearson_r(predictions, targets)
    
    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": r,
        "p_value": p_value
    }


def bootstrap_confidence_interval(
    predictions: np.ndarray,
    targets: np.ndarray,
    metric_fn,
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        metric_fn: Function to calculate metric (takes predictions, targets)
        n_iterations: Number of bootstrap iterations
        ci: Confidence interval level (e.g., 0.95 for 95% CI)
        seed: Random seed
        
    Returns:
        point_estimate: Original metric value
        ci_lower: Lower bound of CI
        ci_upper: Upper bound of CI
    """
    np.random.seed(seed)
    n_samples = len(predictions)
    
    # Point estimate
    point_estimate = metric_fn(predictions, targets)
    
    # Bootstrap
    bootstrap_values = []
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_preds = predictions[indices]
        boot_targets = targets[indices]
        bootstrap_values.append(metric_fn(boot_preds, boot_targets))
    
    bootstrap_values = np.array(bootstrap_values)
    
    # Calculate CI
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_values, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)
    
    return point_estimate, ci_lower, ci_upper


def bootstrap_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Calculate bootstrap CIs for all metrics.
    
    Returns:
        Dictionary with point estimates and CIs for MAE, RMSE, and Pearson r
    """
    results = {}
    
    # MAE
    mae, mae_lower, mae_upper = bootstrap_confidence_interval(
        predictions, targets, calculate_mae, n_iterations, ci, seed
    )
    results["mae"] = {
        "value": mae,
        "ci_lower": mae_lower,
        "ci_upper": mae_upper
    }
    
    # RMSE
    rmse, rmse_lower, rmse_upper = bootstrap_confidence_interval(
        predictions, targets, calculate_rmse, n_iterations, ci, seed
    )
    results["rmse"] = {
        "value": rmse,
        "ci_lower": rmse_lower,
        "ci_upper": rmse_upper
    }
    
    # Pearson r (need wrapper function)
    def pearson_r_only(preds, targs):
        r, _ = calculate_pearson_r(preds, targs)
        return r
    
    r, r_lower, r_upper = bootstrap_confidence_interval(
        predictions, targets, pearson_r_only, n_iterations, ci, seed
    )
    results["pearson_r"] = {
        "value": r,
        "ci_lower": r_lower,
        "ci_upper": r_upper
    }
    
    return results


def evaluate_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
    n_bootstrap: int = 1000,
    ci: float = 0.95
) -> Dict[str, Any]:
    """
    Evaluate model on test set with bootstrap CIs.
    
    Returns:
        Dictionary with predictions, targets, and metrics with CIs
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, ages in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(ages.numpy())
    
    predictions = np.array(all_preds)
    targets = np.array(all_targets)
    
    # Calculate metrics with bootstrap CIs
    metrics = bootstrap_all_metrics(
        predictions, targets,
        n_iterations=n_bootstrap,
        ci=ci
    )
    
    return {
        "predictions": predictions.tolist(),
        "targets": targets.tolist(),
        "metrics": metrics
    }


def ensemble_predictions(
    models: List[nn.Module],
    test_loader: DataLoader,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get ensemble predictions from multiple models (e.g., from different folds).
    
    Returns:
        mean_predictions: Average predictions across models
        targets: Ground truth
    """
    all_model_preds = []
    targets = None
    
    for model in models:
        model.eval()
        model_preds = []
        model_targets = []
        
        with torch.no_grad():
            for images, ages in test_loader:
                images = images.to(device)
                outputs = model(images)
                model_preds.extend(outputs.cpu().numpy())
                model_targets.extend(ages.numpy())
        
        all_model_preds.append(model_preds)
        targets = np.array(model_targets)
    
    # Average predictions
    all_model_preds = np.array(all_model_preds)
    mean_predictions = np.mean(all_model_preds, axis=0)
    
    return mean_predictions, targets


def evaluate_ensemble(
    model_paths: List[str],
    model_class,
    model_name: str,
    test_loader: DataLoader,
    device: str = "cuda",
    n_bootstrap: int = 1000,
    ci: float = 0.95
) -> Dict[str, Any]:
    """
    Evaluate ensemble of models from different folds.
    """
    # Load models
    models = []
    for path in model_paths:
        model = model_class(model_name, pretrained=False).to(device)
        model.load_state_dict(torch.load(path))
        models.append(model)
    
    # Get ensemble predictions
    predictions, targets = ensemble_predictions(models, test_loader, device)
    
    # Calculate metrics
    metrics = bootstrap_all_metrics(
        predictions, targets,
        n_iterations=n_bootstrap,
        ci=ci
    )
    
    return {
        "predictions": predictions.tolist(),
        "targets": targets.tolist(),
        "metrics": metrics,
        "n_models": len(models)
    }


def format_metric_with_ci(metric_dict: Dict[str, float], decimals: int = 3) -> str:
    """
    Format metric with CI for display.
    
    Example: "3.920 (95% CI: 3.332 - 4.637)"
    """
    value = metric_dict["value"]
    lower = metric_dict["ci_lower"]
    upper = metric_dict["ci_upper"]
    
    return f"{value:.{decimals}f} (95% CI: {lower:.{decimals}f} - {upper:.{decimals}f})"


def print_evaluation_results(results: Dict[str, Any]):
    """
    Print formatted evaluation results.
    """
    print("\n" + "="*60)
    print("Test Set Evaluation Results")
    print("="*60)
    
    metrics = results["metrics"]
    
    print(f"\nMAE:       {format_metric_with_ci(metrics['mae'])}")
    print(f"RMSE:      {format_metric_with_ci(metrics['rmse'])}")
    print(f"Pearson r: {format_metric_with_ci(metrics['pearson_r'])}")
    
    print("\n" + "="*60)


def calculate_detailed_statistics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate detailed statistics for predictions.
    """
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    return {
        "n_samples": len(predictions),
        "mae": np.mean(abs_errors),
        "median_ae": np.median(abs_errors),
        "rmse": np.sqrt(np.mean(errors**2)),
        "mean_error": np.mean(errors),  # Bias
        "std_error": np.std(errors),
        "min_error": np.min(errors),
        "max_error": np.max(errors),
        "q25_abs_error": np.percentile(abs_errors, 25),
        "q75_abs_error": np.percentile(abs_errors, 75),
        "within_5_years": np.mean(abs_errors <= 5) * 100,
        "within_10_years": np.mean(abs_errors <= 10) * 100,
    }


def calculate_age_stratified_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    age_bins: List[int] = [0, 30, 40, 50, 60, 70, 100],
    n_bootstrap: int = 1000
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate metrics stratified by age group with bootstrap CIs.
    """
    import pandas as pd
    
    bin_labels = [f"{age_bins[i]}-{age_bins[i+1]}" for i in range(len(age_bins)-1)]
    age_groups = pd.cut(targets, bins=age_bins, labels=bin_labels)
    
    results = {}
    
    for group in bin_labels:
        mask = age_groups == group
        if mask.sum() == 0:
            continue
        
        group_preds = predictions[mask]
        group_targets = targets[mask]
        group_errors = np.abs(group_preds - group_targets)
        
        # Bootstrap CI for MAE
        bootstrap_maes = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(group_errors), size=len(group_errors), replace=True)
            bootstrap_maes.append(np.mean(group_errors[idx]))
        
        results[group] = {
            "n": int(mask.sum()),
            "mae": float(np.mean(group_errors)),
            "mae_ci_lower": float(np.percentile(bootstrap_maes, 2.5)),
            "mae_ci_upper": float(np.percentile(bootstrap_maes, 97.5)),
            "rmse": float(np.sqrt(np.mean((group_preds - group_targets)**2))),
            "mean_bias": float(np.mean(group_preds - group_targets))
        }
    
    return results


def compare_predictions_paired(
    predictions1: np.ndarray,
    predictions2: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """
    Compare two sets of predictions using paired statistical tests.
    
    Returns:
        Dictionary with statistical test results
    """
    errors1 = np.abs(predictions1 - targets)
    errors2 = np.abs(predictions2 - targets)
    
    diff = errors1 - errors2
    
    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(errors1, errors2)
    
    # Wilcoxon signed-rank test
    w_stat, w_pval = stats.wilcoxon(errors1, errors2)
    
    # Bootstrap CI for difference
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(diff), size=len(diff), replace=True)
        bootstrap_diffs.append(np.mean(diff[idx]))
    
    return {
        "mae_1": float(np.mean(errors1)),
        "mae_2": float(np.mean(errors2)),
        "mean_difference": float(np.mean(diff)),
        "diff_ci_lower": float(np.percentile(bootstrap_diffs, 2.5)),
        "diff_ci_upper": float(np.percentile(bootstrap_diffs, 97.5)),
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pval),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pval),
        "cohens_d": float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0
    }


def generate_ci_comparison_table(
    model_results: Dict[str, Dict[str, Any]]
) -> str:
    """
    Generate a formatted table comparing CIs across models.
    
    Returns:
        LaTeX formatted table string
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Model Comparison with 95\% Confidence Intervals}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & MAE (95\% CI) & RMSE (95\% CI) & Pearson $r$ (95\% CI) \\",
        r"\midrule"
    ]
    
    for model_name, results in model_results.items():
        metrics = results['metrics']
        
        mae_str = f"{metrics['mae']['value']:.3f} ({metrics['mae']['ci_lower']:.3f}--{metrics['mae']['ci_upper']:.3f})"
        rmse_str = f"{metrics['rmse']['value']:.3f} ({metrics['rmse']['ci_lower']:.3f}--{metrics['rmse']['ci_upper']:.3f})"
        r_str = f"{metrics['pearson_r']['value']:.3f} ({metrics['pearson_r']['ci_lower']:.3f}--{metrics['pearson_r']['ci_upper']:.3f})"
        
        lines.append(f"{model_name} & {mae_str} & {rmse_str} & {r_str} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    return "\n".join(lines)
