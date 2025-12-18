"""
Dataset and DataLoader for Age Estimation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold


class AgeEstimationDataset(Dataset):
    """
    Dataset for age estimation from anterior segment images.
    
    Args:
        image_paths: List of image file paths
        ages: List of ages (labels)
        transform: Optional torchvision transforms
    """
    
    def __init__(
        self,
        image_paths: List[str],
        ages: List[float],
        transform: Optional[Callable] = None
    ):
        self.image_paths = image_paths
        self.ages = ages
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get age label
        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        
        return image, age


def get_transforms(
    image_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    is_train: bool = True
) -> transforms.Compose:
    """
    Get image transforms.
    
    Note: No data augmentation as per the project specification.
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    
    return transforms.Compose(transform_list)


import os

def load_data_from_csv(
    csv_path: str,
    image_dir: str = "",
    ext: str = ".jpg"
) -> Tuple[List[str], List[float]]:
    df = pd.read_csv(csv_path)
    
    path_col = None
    age_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'path' in col_lower or 'image' in col_lower or 'file' in col_lower:
            path_col = col
        if 'age' in col_lower or 'label' in col_lower or 'year' in col_lower:
            age_col = col
    
    if path_col is None:
        path_col = df.columns[0]
    if age_col is None:
        age_col = df.columns[1]
    
    image_names = df[path_col].tolist()
    ages = df[age_col].astype(float).tolist()
    
    # フルパスを作成し、存在するファイルのみ残す
    image_paths = []
    valid_ages = []
    skipped = 0
    
    for name, age in zip(image_names, ages):
        if image_dir:
            path = f"{image_dir}/{name}{ext}"
        else:
            path = f"{name}{ext}"
        
        if os.path.exists(path):
            image_paths.append(path)
            valid_ages.append(age)
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"  Skipped {skipped} files (not found)")
    
    return image_paths, valid_ages


def create_fold_dataloaders(
    image_paths: List[str],
    ages: List[float],
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    config
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for a specific fold.
    """
    # Split data
    train_paths = [image_paths[i] for i in train_indices]
    train_ages = [ages[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    val_ages = [ages[i] for i in val_indices]
    
    # Create transforms
    train_transform = get_transforms(
        image_size=config.image_size,
        mean=config.mean,
        std=config.std,
        is_train=True
    )
    val_transform = get_transforms(
        image_size=config.image_size,
        mean=config.mean,
        std=config.std,
        is_train=False
    )
    
    # Create datasets
    train_dataset = AgeEstimationDataset(train_paths, train_ages, train_transform)
    val_dataset = AgeEstimationDataset(val_paths, val_ages, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_test_dataloader(
    image_paths: List[str],
    ages: List[float],
    config
) -> DataLoader:
    """
    Create test dataloader.
    """
    test_transform = get_transforms(
        image_size=config.image_size,
        mean=config.mean,
        std=config.std,
        is_train=False
    )
    
    test_dataset = AgeEstimationDataset(image_paths, ages, test_transform)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return test_loader


def get_kfold_splits(
    n_samples: int,
    n_folds: int = 5,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Get K-Fold cross-validation splits.
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(n_samples)
    
    splits = []
    for train_idx, val_idx in kfold.split(indices):
        splits.append((train_idx, val_idx))
    
    return splits
