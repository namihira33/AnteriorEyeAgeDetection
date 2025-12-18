"""
Analysis Script for Age Estimation Models
- Model comparison with Grad-CAM visualization
- Confidence interval analysis
- Statistical tests between models
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import Config
from dataset import (
    load_data_from_csv,
    get_transforms,
    AgeEstimationDataset,
    create_test_dataloader
)
from models import AgeEstimationModel, MODEL_DISPLAY_NAMES
from evaluation import (
    calculate_metrics,
    bootstrap_all_metrics,
    format_metric_with_ci
)
from visualization import (
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    get_target_layer,
    visualize_cam,
    generate_model_comparison,
    batch_generate_cams,
    analyze_attention_by_age_group,
    plot_error_distribution
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_path: str, model_name: str, device: str = 'cuda') -> nn.Module:
    """Load a trained model from checkpoint."""
    model = AgeEstimationModel(model_name, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_ensemble_models(
    model_dir: str,
    model_name: str,
    n_folds: int = 5,
    device: str = 'cuda'
) -> List[nn.Module]:
    """Load all fold models for ensemble."""
    models = []
    for fold in range(1, n_folds + 1):
        model_path = Path(model_dir) / f"{model_name}_fold{fold}.pth"
        if model_path.exists():
            model = load_model(str(model_path), model_name, device)
            models.append(model)
        else:
            print(f"Warning: Model not found: {model_path}")
    return models


def get_ensemble_predictions(
    models: List[nn.Module],
    dataloader: DataLoader,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray]:
    """Get averaged predictions from ensemble."""
    all_preds = []
    targets = None
    
    for model in models:
        model.eval()
        preds = []
        targs = []
        
        with torch.no_grad():
            for images, ages in dataloader:
                images = images.to(device)
                outputs = model(images)
                preds.extend(outputs.cpu().numpy())
                targs.extend(ages.numpy())
        
        all_preds.append(preds)
        targets = np.array(targs)
    
    # Average predictions
    mean_preds = np.mean(all_preds, axis=0)
    return mean_preds, targets


def compare_models_statistical(
    model_results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str
) -> pd.DataFrame:
    """
    Perform statistical comparison between models.
    Uses paired t-test and Wilcoxon signed-rank test.
    """
    results = []
    model_names = list(model_results.keys())
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            errors1 = np.abs(model_results[model1]['predictions'] - model_results[model1]['targets'])
            errors2 = np.abs(model_results[model2]['predictions'] - model_results[model2]['targets'])
            
            # Paired t-test
            t_stat, t_pval = stats.ttest_rel(errors1, errors2)
            
            # Wilcoxon signed-rank test
            w_stat, w_pval = stats.wilcoxon(errors1, errors2)
            
            # Effect size (Cohen's d)
            diff = errors1 - errors2
            cohens_d = np.mean(diff) / np.std(diff)
            
            results.append({
                'Model 1': model1,
                'Model 2': model2,
                'MAE 1': np.mean(errors1),
                'MAE 2': np.mean(errors2),
                'Difference': np.mean(errors1) - np.mean(errors2),
                't-statistic': t_stat,
                't p-value': t_pval,
                'Wilcoxon statistic': w_stat,
                'Wilcoxon p-value': w_pval,
                "Cohen's d": cohens_d
            })
    
    df = pd.DataFrame(results)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / 'statistical_comparison.csv', index=False)
    
    print("\nStatistical Comparison:")
    print(df.to_string(index=False))
    
    return df


def analyze_error_by_age(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    output_dir: str,
    age_bins: List[int] = [0, 30, 40, 50, 60, 70, 100]
):
    """
    Analyze prediction error by age group with confidence intervals.
    """
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    # Bin by age
    bin_labels = [f"{age_bins[i]}-{age_bins[i+1]}" for i in range(len(age_bins)-1)]
    age_groups = pd.cut(targets, bins=age_bins, labels=bin_labels)
    
    # Calculate statistics per group
    results = []
    for group in bin_labels:
        mask = age_groups == group
        if mask.sum() == 0:
            continue
        
        group_errors = abs_errors[mask]
        
        # Bootstrap CI for MAE
        bootstrap_maes = []
        for _ in range(1000):
            sample = np.random.choice(group_errors, size=len(group_errors), replace=True)
            bootstrap_maes.append(np.mean(sample))
        
        ci_lower = np.percentile(bootstrap_maes, 2.5)
        ci_upper = np.percentile(bootstrap_maes, 97.5)
        
        results.append({
            'Age Group': group,
            'N': mask.sum(),
            'MAE': np.mean(group_errors),
            'MAE CI Lower': ci_lower,
            'MAE CI Upper': ci_upper,
            'Std': np.std(group_errors),
            'Mean Error': np.mean(errors[mask]),
            'Median Error': np.median(errors[mask])
        })
    
    df = pd.DataFrame(results)
    
    # Save
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / 'error_by_age.csv', index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE by age group with CI
    x = range(len(df))
    axes[0].bar(x, df['MAE'], yerr=[df['MAE'] - df['MAE CI Lower'], 
                                      df['MAE CI Upper'] - df['MAE']], 
                capsize=5, alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df['Age Group'], rotation=45)
    axes[0].set_xlabel('Age Group (years)')
    axes[0].set_ylabel('MAE (years)')
    axes[0].set_title(f'{model_name} - MAE by Age Group (95% CI)')
    
    # Mean error (bias) by age group
    axes[1].bar(x, df['Mean Error'], alpha=0.7, color='orange')
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df['Age Group'], rotation=45)
    axes[1].set_xlabel('Age Group (years)')
    axes[1].set_ylabel('Mean Error (years)')
    axes[1].set_title(f'{model_name} - Prediction Bias by Age Group')
    
    plt.tight_layout()
    plt.savefig(output_path / 'error_by_age.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nError by Age Group for {model_name}:")
    print(df.to_string(index=False))
    
    return df


def generate_publication_figures(
    model_results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str
):
    """
    Generate publication-quality figures for all models.
    """
    output_path = Path(output_dir) / 'figures'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Scatter plots for all models
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, results) in enumerate(model_results.items()):
        preds = results['predictions']
        targets = results['targets']
        
        # Scatter plot
        axes[i].scatter(targets, preds, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        # Regression line
        z = np.polyfit(targets, preds, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        axes[i].plot(x_line, p(x_line), 'b-', lw=1, alpha=0.7)
        
        # Metrics
        mae = np.mean(np.abs(preds - targets))
        r, _ = stats.pearsonr(preds, targets)
        
        axes[i].set_xlabel('True Age (years)', fontsize=12)
        axes[i].set_ylabel('Predicted Age (years)', fontsize=12)
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        axes[i].set_title(f'{display_name}\nMAE={mae:.2f}, r={r:.3f}', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path / 'scatter_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'scatter_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 2: Model comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = []
    maes = []
    ci_lowers = []
    ci_uppers = []
    
    for model_name, results in model_results.items():
        preds = results['predictions']
        targets = results['targets']
        
        # Bootstrap CI
        abs_errors = np.abs(preds - targets)
        bootstrap_maes = []
        for _ in range(1000):
            sample = np.random.choice(abs_errors, size=len(abs_errors), replace=True)
            bootstrap_maes.append(np.mean(sample))
        
        model_names.append(MODEL_DISPLAY_NAMES.get(model_name, model_name))
        maes.append(np.mean(abs_errors))
        ci_lowers.append(np.percentile(bootstrap_maes, 2.5))
        ci_uppers.append(np.percentile(bootstrap_maes, 97.5))
    
    x = range(len(model_names))
    yerr = [np.array(maes) - np.array(ci_lowers), np.array(ci_uppers) - np.array(maes)]
    
    bars = ax.bar(x, maes, yerr=yerr, capsize=5, alpha=0.7, color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('MAE (years)', fontsize=12)
    ax.set_title('Model Comparison - Test Set MAE with 95% CI', fontsize=14)
    
    # Add value labels
    for bar, mae, lower, upper in zip(bars, maes, ci_lowers, ci_uppers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mae:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'model_comparison_bar.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'model_comparison_bar.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Publication figures saved to {output_path}")


def generate_detailed_report(
    model_results: Dict[str, Dict[str, Any]],
    output_dir: str
) -> str:
    """
    Generate a detailed analysis report in Markdown format.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# Age Estimation Model Analysis Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Summary\n"
    ]
    
    # Summary table
    lines.append("| Model | MAE (95% CI) | RMSE | Pearson r |")
    lines.append("|-------|--------------|------|-----------|")
    
    for model_name, results in model_results.items():
        metrics = results['metrics']
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        
        mae_str = f"{metrics['mae']['value']:.3f} ({metrics['mae']['ci_lower']:.3f}-{metrics['mae']['ci_upper']:.3f})"
        rmse_str = f"{metrics['rmse']['value']:.3f}"
        r_str = f"{metrics['pearson_r']['value']:.3f}"
        
        lines.append(f"| {display_name} | {mae_str} | {rmse_str} | {r_str} |")
    
    # Best model
    best_model = min(model_results.keys(), 
                     key=lambda m: model_results[m]['metrics']['mae']['value'])
    best_mae = model_results[best_model]['metrics']['mae']['value']
    
    lines.extend([
        f"\n## Best Model\n",
        f"**{MODEL_DISPLAY_NAMES.get(best_model, best_model)}** achieved the lowest MAE of **{best_mae:.3f} years**.",
        "\n## Detailed Results\n"
    ])
    
    for model_name, results in model_results.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        metrics = results['metrics']
        
        lines.extend([
            f"### {display_name}\n",
            f"- MAE: {format_metric_with_ci(metrics['mae'])}",
            f"- RMSE: {format_metric_with_ci(metrics['rmse'])}",
            f"- Pearson r: {format_metric_with_ci(metrics['pearson_r'])}",
            ""
        ])
    
    report = "\n".join(lines)
    
    # Save
    with open(output_path / 'analysis_report.md', 'w') as f:
        f.write(report)
    
    print(f"Report saved to {output_path / 'analysis_report.md'}")
    
    return report


def run_cam_analysis(
    models_dir: str,
    model_names: List[str],
    test_csv: str,
    image_dir: str,
    output_dir: str,
    n_samples: int = 20,
    device: str = 'cuda'
):
    """
    Run Grad-CAM analysis for all models.
    """
    config = Config()
    
    # Load test data
    test_paths, test_ages = load_data_from_csv(test_csv, image_dir=image_dir, ext=".jpg")
    
    transform = get_transforms(
        image_size=config.image_size,
        mean=config.mean,
        std=config.std,
        is_train=False
    )
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"CAM Analysis for {MODEL_DISPLAY_NAMES.get(model_name, model_name)}")
        print(f"{'='*60}")
        
        # Load best fold model (fold 1)
        model_path = Path(models_dir) / f"{model_name}_fold1.pth"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue
        
        model = load_model(str(model_path), model_name, device)
        
        # Create dataloader
        test_dataset = AgeEstimationDataset(test_paths[:n_samples], 
                                            test_ages[:n_samples], 
                                            transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # Generate CAMs
        cam_results = batch_generate_cams(
            model, model_name, test_loader, 
            output_dir, method='gradcam',
            max_samples=n_samples, device=device
        )
        
        # Analyze by age group
        full_dataset = AgeEstimationDataset(test_paths, test_ages, transform)
        full_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)
        
        try:
            analyze_attention_by_age_group(
                model, model_name, full_loader,
                output_dir, method='gradcam',
                max_per_group=10, device=device
            )
        except Exception as e:
            print(f"Warning: Age group analysis failed: {e}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Age Estimation Models")
    
    parser.add_argument("--models_dir", type=str, required=True,
                       help="Directory containing trained models")
    parser.add_argument("--test_csv", type=str, required=True,
                       help="Path to test CSV file")
    parser.add_argument("--image_dir", type=str, default="",
                       help="Directory containing test images")
    parser.add_argument("--output_dir", type=str, default="./analysis_results",
                       help="Output directory for analysis results")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Models to analyze")
    parser.add_argument("--n_folds", type=int, default=5,
                       help="Number of folds used in training")
    parser.add_argument("--cam_samples", type=int, default=20,
                       help="Number of samples for CAM analysis")
    parser.add_argument("--skip_cam", action="store_true",
                       help="Skip CAM visualization")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    config = Config()
    
    # Default models if not specified
    if args.models is None:
        args.models = [
            "resnet50",
            "efficientnet_b7",
            "vit_base_patch16_224",
            "vit_large_patch16_224",
            "swin_base_patch4_window7_224"
        ]
    
    # Load test data
    print("\nLoading test data...")
    test_paths, test_ages = load_data_from_csv(
        args.test_csv, 
        image_dir=args.image_dir,
        ext=".jpg"
    )
    print(f"Test samples: {len(test_paths)}")
    
    # Create test dataloader
    test_loader = create_test_dataloader(test_paths, test_ages, config)
    
    # Analyze each model
    all_results = {}
    
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Analyzing: {MODEL_DISPLAY_NAMES.get(model_name, model_name)}")
        print(f"{'='*60}")
        
        # Load ensemble models
        models = load_ensemble_models(
            args.models_dir, model_name, 
            args.n_folds, device
        )
        
        if len(models) == 0:
            print(f"No models found for {model_name}")
            continue
        
        # Get predictions
        predictions, targets = get_ensemble_predictions(models, test_loader, device)
        
        # Calculate metrics with bootstrap CI
        metrics = bootstrap_all_metrics(
            predictions, targets,
            n_iterations=1000,
            ci=0.95
        )
        
        all_results[model_name] = {
            'predictions': predictions,
            'targets': targets,
            'metrics': metrics
        }
        
        # Print results
        print(f"\nResults for {model_name}:")
        print(f"  MAE: {format_metric_with_ci(metrics['mae'])}")
        print(f"  RMSE: {format_metric_with_ci(metrics['rmse'])}")
        print(f"  Pearson r: {format_metric_with_ci(metrics['pearson_r'])}")
        
        # Error distribution plot
        plot_error_distribution(predictions, targets, model_name, args.output_dir)
        
        # Error by age analysis
        analyze_error_by_age(predictions, targets, model_name, args.output_dir)
        
        # Clean up
        for model in models:
            del model
        torch.cuda.empty_cache()
    
    if len(all_results) == 0:
        print("No models analyzed. Exiting.")
        return
    
    # Statistical comparison between models
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("Statistical Comparison Between Models")
        print("="*60)
        compare_models_statistical(all_results, args.output_dir)
    
    # Generate publication figures
    generate_publication_figures(all_results, args.output_dir)
    
    # Generate report
    generate_detailed_report(all_results, args.output_dir)
    
    # CAM analysis
    if not args.skip_cam:
        print("\n" + "="*60)
        print("Grad-CAM Analysis")
        print("="*60)
        run_cam_analysis(
            args.models_dir,
            args.models,
            args.test_csv,
            args.image_dir,
            args.output_dir,
            n_samples=args.cam_samples,
            device=device
        )
    
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
