"""
Phase 1: Model Comparison for Age Estimation from Anterior Segment Images

This script compares 5 models:
- ResNet50
- EfficientNet-B7
- ViT-Base
- ViT-Large
- Swin Transformer

Using 5-fold cross-validation with LR range test.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import Config
from dataset import (
    load_data_from_csv,
    get_kfold_splits,
    create_test_dataloader
)
from models import AgeEstimationModel, MODEL_DISPLAY_NAMES, print_model_summary
from trainer import cross_validate
from evaluation import (
    evaluate_on_test_set,
    evaluate_ensemble,
    print_evaluation_results
)
from utils import (
    save_results_json,
    save_cv_summary_csv,
    save_test_results_csv,
    save_predictions_csv,
    create_experiment_log,
    save_experiment_log,
    print_summary_table
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Model Comparison for Age Estimation"
    )
    
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Path to training CSV file"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/phase1",
        help="Output directory"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to compare (default: all 5 models)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum epochs per fold"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--skip_cv",
        action="store_true",
        help="Skip cross-validation (only evaluate on test set)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    start_time = datetime.now()
    
    # Create config
    config = Config(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        n_folds=args.n_folds,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers
    )
    
    if args.models:
        config.models = args.models
    
    # Setup
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Print model summary
    print_model_summary()
    
    # Load data
    print("\nLoading training data...")
    # 修正後
    train_paths, train_ages = load_data_from_csv(
        config.train_csv,
        image_dir="./images/normal20251210/training",
        ext=".jpg"
    )
    test_paths, test_ages = load_data_from_csv(
        config.test_csv,
        image_dir="./images/normal20251210/test",
        ext=".jpg"
    )
    print(f"  Total samples: {len(train_paths)}")
    print(f"  Age range: {min(train_ages):.1f} - {max(train_ages):.1f} years")
    print(f"  Mean age: {np.mean(train_ages):.1f} ± {np.std(train_ages):.1f} years")
    
    print("\nLoading test data...")
    print(f"  Total samples: {len(test_paths)}")
    print(f"  Age range: {min(test_ages):.1f} - {max(test_ages):.1f} years")
    print(f"  Mean age: {np.mean(test_ages):.1f} ± {np.std(test_ages):.1f} years")
    
    # Get CV splits
    fold_splits = get_kfold_splits(
        len(train_paths),
        n_folds=config.n_folds,
        seed=config.seed
    )
    
    # Results storage
    all_cv_results = {}
    all_test_results = {}
    
    # Cross-validation for each model
    for model_name in config.models:
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        print(f"\n{'#'*60}")
        print(f"# Processing: {display_name}")
        print(f"{'#'*60}")
        
        if not args.skip_cv:
            # Cross-validation
            cv_results = cross_validate(
                model_name=model_name,
                image_paths=train_paths,
                ages=train_ages,
                fold_splits=fold_splits,
                config=config,
                device=device
            )
            all_cv_results[model_name] = cv_results
            
            # Save CV results
            save_results_json(
                cv_results,
                Path(config.output_dir) / f"{model_name}_cv_results.json"
            )
        
        # Test set evaluation with ensemble
        print(f"\nEvaluating {display_name} on test set (ensemble)...")
        
        model_paths = [
            Path(config.output_dir) / f"{model_name}_fold{i+1}.pth"
            for i in range(config.n_folds)
        ]
        
        # Check if all fold models exist
        if all(p.exists() for p in model_paths):
            test_loader = create_test_dataloader(test_paths, test_ages, config)
            
            test_results = evaluate_ensemble(
                model_paths=[str(p) for p in model_paths],
                model_class=AgeEstimationModel,
                model_name=model_name,
                test_loader=test_loader,
                device=device,
                n_bootstrap=config.bootstrap_n_iterations,
                ci=config.bootstrap_ci
            )
            
            all_test_results[model_name] = test_results
            print_evaluation_results(test_results)
            
            # Save predictions
            save_predictions_csv(
                np.array(test_results["predictions"]),
                np.array(test_results["targets"]),
                test_paths,
                Path(config.output_dir) / f"{model_name}_test_predictions.csv"
            )
        else:
            print(f"  Warning: Not all fold models found for {model_name}")
            print(f"  Expected: {[str(p) for p in model_paths]}")
    
    # Save summary results
    if all_cv_results:
        cv_summary_df = save_cv_summary_csv(
            all_cv_results,
            Path(config.output_dir) / "cv_summary.csv"
        )
        print_summary_table(cv_summary_df, "Cross-Validation Summary")
    
    if all_test_results:
        test_summary_df = save_test_results_csv(
            all_test_results,
            Path(config.output_dir) / "test_results.csv"
        )
        print_summary_table(test_summary_df, "Test Set Results")
    
    # Find best model
    if all_test_results:
        best_model = min(
            all_test_results.keys(),
            key=lambda m: all_test_results[m]["metrics"]["mae"]["value"]
        )
        best_mae = all_test_results[best_model]["metrics"]["mae"]["value"]
        print(f"\n{'*'*60}")
        print(f"Best Model: {MODEL_DISPLAY_NAMES.get(best_model, best_model)}")
        print(f"Test MAE: {best_mae:.3f} years")
        print(f"{'*'*60}")
    
    # Save experiment log
    end_time = datetime.now()
    log = create_experiment_log(config, start_time, end_time)
    log["results"] = {
        "best_model": best_model if all_test_results else None,
        "cv_results_available": list(all_cv_results.keys()),
        "test_results_available": list(all_test_results.keys())
    }
    save_experiment_log(log, Path(config.output_dir) / "experiment_log.json")
    
    print(f"\nExperiment completed in {(end_time - start_time).total_seconds():.1f} seconds")
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
