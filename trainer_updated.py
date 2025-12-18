"""
Training Utilities for Age Estimation
- LR Range Test
- Early Stopping
- Training Loop
- Final Model Training on Full Data
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import copy
import json
from pathlib import Path


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.mode == "min":
            score = -score
            
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            
        return self.early_stop
    
    def load_best_model(self, model: nn.Module) -> nn.Module:
        """Load the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
    scheduler: Optional[Any] = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for images, ages in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        ages = ages.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, ages)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(ages.cpu().numpy())
    
    if scheduler is not None:
        scheduler.step()
    
    avg_loss = total_loss / len(dataloader.dataset)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    
    return {
        "loss": avg_loss,
        "mae": mae
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Validate the model.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, ages in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            ages = ages.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, ages)
            
            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(ages.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    
    return {
        "loss": avg_loss,
        "mae": mae,
        "rmse": rmse,
        "predictions": preds,
        "targets": targets
    }


def lr_range_test(
    model_class,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr_candidates: List[float],
    n_epochs: int = 3,
    weight_decay: float = 0.01,
    device: str = "cuda",
    pretrained: bool = True
) -> Tuple[float, Dict[str, List[float]]]:
    """
    Learning Rate Range Test.
    
    Tests each learning rate candidate and returns the best one
    based on validation MAE.
    
    Args:
        model_class: Model class to instantiate
        model_name: Name of the model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        lr_candidates: List of learning rates to test
        n_epochs: Number of epochs per LR test
        weight_decay: Weight decay for AdamW
        device: Device to use
        pretrained: Whether to use pretrained weights
        
    Returns:
        best_lr: Best learning rate
        results: Dictionary with MAE for each LR
    """
    results = {
        "lr": [],
        "val_mae": [],
        "val_loss": []
    }
    
    print(f"\nLR Range Test for {model_name}")
    print("-" * 40)
    
    for lr in lr_candidates:
        print(f"\nTesting LR = {lr}")
        
        # Create fresh model for each LR
        model = model_class(model_name, pretrained=pretrained).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        best_val_mae = float('inf')
        
        for epoch in range(n_epochs):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_metrics = validate(model, val_loader, criterion, device)
            
            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
            
            print(f"  Epoch {epoch+1}/{n_epochs} - "
                  f"Train MAE: {train_metrics['mae']:.3f}, "
                  f"Val MAE: {val_metrics['mae']:.3f}")
        
        results["lr"].append(lr)
        results["val_mae"].append(best_val_mae)
        results["val_loss"].append(val_metrics["loss"])
        
        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()
    
    # Find best LR
    best_idx = np.argmin(results["val_mae"])
    best_lr = results["lr"][best_idx]
    
    print(f"\nBest LR: {best_lr} (Val MAE: {results['val_mae'][best_idx]:.3f})")
    
    return best_lr, results


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    max_epochs: int,
    patience: int,
    weight_decay: float = 0.01,
    device: str = "cuda"
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a single fold with early stopping.
    
    Returns:
        model: Trained model (best state)
        history: Training history
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr/100)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, mode="min")
    
    history = {
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": []
    }
    
    for epoch in range(max_epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Log
        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        
        print(f"Epoch {epoch+1}/{max_epochs} - "
              f"Train MAE: {train_metrics['mae']:.3f}, "
              f"Val MAE: {val_metrics['mae']:.3f}, "
              f"Val RMSE: {val_metrics['rmse']:.3f}")
        
        # Early stopping
        if early_stopping(val_metrics["mae"], model):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model = early_stopping.load_best_model(model)
    
    return model, history


def train_full_data(
    model: nn.Module,
    train_loader: DataLoader,
    lr: float,
    n_epochs: int,
    weight_decay: float = 0.01,
    device: str = "cuda"
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train on full training data without validation (for final model).
    
    Uses fixed number of epochs determined from CV results.
    
    Returns:
        model: Trained model
        history: Training history
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr/100)
    criterion = nn.MSELoss()
    
    history = {
        "train_loss": [],
        "train_mae": []
    }
    
    print(f"\nTraining final model on full data for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        
        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae"])
        
        print(f"Epoch {epoch+1}/{n_epochs} - "
              f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train MAE: {train_metrics['mae']:.3f}")
    
    return model, history


def cross_validate(
    model_name: str,
    image_paths: List[str],
    ages: List[float],
    fold_splits: List[Tuple[np.ndarray, np.ndarray]],
    config,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Perform K-Fold Cross Validation.
    
    Returns:
        results: Dictionary with results for each fold and aggregated metrics
    """
    from models import AgeEstimationModel
    from dataset import create_fold_dataloaders
    
    results = {
        "model_name": model_name,
        "folds": [],
        "best_lrs": [],
        "val_maes": [],
        "val_rmses": [],
        "best_epochs": []  # 各foldの最適エポック数を記録
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{len(fold_splits)}")
        print(f"{'='*60}")
        
        # Create dataloaders for this fold
        train_loader, val_loader = create_fold_dataloaders(
            image_paths, ages, fold_idx, train_idx, val_idx, config
        )
        
        # LR Range Test
        best_lr, lr_results = lr_range_test(
            AgeEstimationModel,
            model_name,
            train_loader,
            val_loader,
            config.lr_candidates,
            n_epochs=config.lr_test_epochs,
            weight_decay=config.weight_decay,
            device=device,
            pretrained=config.pretrained
        )
        results["best_lrs"].append(best_lr)
        
        # Train with best LR
        print(f"\nTraining with LR = {best_lr}")
        model = AgeEstimationModel(model_name, pretrained=config.pretrained).to(device)
        
        model, history = train_fold(
            model,
            train_loader,
            val_loader,
            lr=best_lr,
            max_epochs=config.max_epochs,
            patience=config.early_stopping_patience,
            weight_decay=config.weight_decay,
            device=device
        )
        
        # 最適エポック数を記録（early stoppingで停止したエポック）
        best_epoch = len(history["val_mae"])
        results["best_epochs"].append(best_epoch)
        
        # Final validation
        criterion = nn.MSELoss()
        val_metrics = validate(model, val_loader, criterion, device)
        
        fold_results = {
            "fold": fold_idx + 1,
            "best_lr": best_lr,
            "best_epoch": best_epoch,
            "lr_test_results": lr_results,
            "history": history,
            "final_val_mae": val_metrics["mae"],
            "final_val_rmse": val_metrics["rmse"]
        }
        
        results["folds"].append(fold_results)
        results["val_maes"].append(val_metrics["mae"])
        results["val_rmses"].append(val_metrics["rmse"])
        
        # Save fold model
        torch.save(
            model.state_dict(),
            Path(config.output_dir) / f"{model_name}_fold{fold_idx+1}.pth"
        )
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Aggregate results
    results["mean_val_mae"] = np.mean(results["val_maes"])
    results["std_val_mae"] = np.std(results["val_maes"])
    results["mean_val_rmse"] = np.mean(results["val_rmses"])
    results["std_val_rmse"] = np.std(results["val_rmses"])
    
    # CVから決定されたハイパーパラメータ
    results["optimal_lr"] = np.median(results["best_lrs"])  # 中央値を使用
    results["optimal_epochs"] = int(np.median(results["best_epochs"]))  # 中央値を使用
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results for {model_name}")
    print(f"{'='*60}")
    print(f"Mean Val MAE: {results['mean_val_mae']:.3f} ± {results['std_val_mae']:.3f}")
    print(f"Mean Val RMSE: {results['mean_val_rmse']:.3f} ± {results['std_val_rmse']:.3f}")
    print(f"Optimal LR (median): {results['optimal_lr']}")
    print(f"Optimal Epochs (median): {results['optimal_epochs']}")
    
    return results


def train_final_model(
    model_name: str,
    image_paths: List[str],
    ages: List[float],
    cv_results: Dict[str, Any],
    config,
    device: str = "cuda"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train final model on full training data using hyperparameters from CV.
    
    Args:
        model_name: Name of the model
        image_paths: All training image paths
        ages: All training ages
        cv_results: Results from cross_validate() containing optimal hyperparameters
        config: Configuration object
        device: Device to use
        
    Returns:
        model: Trained final model
        results: Training results
    """
    from models import AgeEstimationModel
    from dataset import get_transforms, AgeEstimationDataset
    from torch.utils.data import DataLoader
    
    print(f"\n{'#'*60}")
    print(f"# Training Final Model on Full Data: {model_name}")
    print(f"{'#'*60}")
    
    # CVから得られた最適ハイパーパラメータを使用
    optimal_lr = cv_results["optimal_lr"]
    optimal_epochs = cv_results["optimal_epochs"]
    
    print(f"Using LR: {optimal_lr}")
    print(f"Using Epochs: {optimal_epochs}")
    
    # 全訓練データのDataLoaderを作成
    transform = get_transforms(
        image_size=config.image_size,
        mean=config.mean,
        std=config.std,
        is_train=True
    )
    
    full_dataset = AgeEstimationDataset(image_paths, ages, transform)
    full_loader = DataLoader(
        full_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # モデル作成
    model = AgeEstimationModel(model_name, pretrained=config.pretrained).to(device)
    
    # 全データで訓練
    model, history = train_full_data(
        model,
        full_loader,
        lr=optimal_lr,
        n_epochs=optimal_epochs,
        weight_decay=config.weight_decay,
        device=device
    )
    
    # 最終モデルを保存
    final_model_path = Path(config.output_dir) / f"{model_name}_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    results = {
        "model_name": model_name,
        "optimal_lr": optimal_lr,
        "optimal_epochs": optimal_epochs,
        "history": history,
        "final_train_mae": history["train_mae"][-1],
        "model_path": str(final_model_path)
    }
    
    return model, results


def cross_validate_and_train_final(
    model_name: str,
    image_paths: List[str],
    ages: List[float],
    fold_splits: List[Tuple[np.ndarray, np.ndarray]],
    config,
    device: str = "cuda",
    train_final: bool = True
) -> Dict[str, Any]:
    """
    Perform CV and optionally train final model on full data.
    
    This is the recommended workflow:
    1. Run CV to determine optimal hyperparameters
    2. Train final model on all training data
    
    Args:
        model_name: Name of the model
        image_paths: Training image paths
        ages: Training ages
        fold_splits: CV fold splits
        config: Configuration
        device: Device
        train_final: Whether to train final model on full data
        
    Returns:
        results: Combined CV and final model results
    """
    # Step 1: Cross-validation
    cv_results = cross_validate(
        model_name, image_paths, ages, fold_splits, config, device
    )
    
    results = {
        "cv_results": cv_results,
        "final_model_results": None
    }
    
    # Step 2: Train final model on full data
    if train_final:
        model, final_results = train_final_model(
            model_name, image_paths, ages, cv_results, config, device
        )
        results["final_model_results"] = final_results
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    return results
