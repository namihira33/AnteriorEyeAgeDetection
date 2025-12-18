"""
Configuration for Age Estimation from Anterior Segment Images
Based on: Guo et al., Br J Ophthalmol 2024
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class Config:
    # Paths (to be set by user)
    train_csv: str = ""  # CSV with columns: image_path, age
    test_csv: str = ""   # CSV with columns: image_path, age
    output_dir: str = "./results"
    
    # Image settings
    image_size: int = 224
    
    # Model settings
    models: List[str] = field(default_factory=lambda: [
        "resnet50",
        "efficientnet_b7",
        "vit_base_patch16_224",
        "vit_large_patch16_224",
        "swin_base_patch4_window7_224"
    ])
    pretrained: bool = True
    
    # Training settings
    batch_size: int = 16
    num_workers: int = 4
    n_folds: int = 5
    max_epochs: int = 100
    early_stopping_patience: int = 5
    
    # Optimizer settings
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    
    # LR Range Test settings
    lr_candidates: List[float] = field(default_factory=lambda: [
        1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3
    ])
    lr_test_epochs: int = 3
    
    # Normalization (ImageNet)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Evaluation settings
    bootstrap_n_iterations: int = 1000
    bootstrap_ci: float = 0.95
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
