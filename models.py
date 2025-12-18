"""
Model Definitions for Age Estimation
Using timm library with ImageNet pretrained weights
"""

import timm
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class AgeEstimationModel(nn.Module):
    """
    Wrapper class for age estimation models.
    Modifies the final layer for regression (single output).
    """
    
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        num_outputs: int = 1
    ):
        super().__init__()
        
        self.model_name = model_name
        
        # Create backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_outputs
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Age predictions of shape (B, 1)
        """
        out = self.backbone(x)
        return out.squeeze(-1)  # (B,) for regression
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature embeddings before the final layer.
        Useful for visualization and analysis.
        """
        return self.backbone.forward_features(x)


# Model name mapping for cleaner display
MODEL_DISPLAY_NAMES = {
    "resnet50": "ResNet-50",
    "efficientnet_b7": "EfficientNet-B7",
    "vit_base_patch16_224": "ViT-Base",
    "vit_large_patch16_224": "ViT-Large",
    "swin_base_patch4_window7_224": "Swin-Base"
}


def get_model(
    model_name: str,
    pretrained: bool = True,
    device: str = "cuda"
) -> AgeEstimationModel:
    """
    Get age estimation model by name.
    
    Args:
        model_name: One of the supported model names
        pretrained: Whether to use ImageNet pretrained weights
        device: Device to load model on
        
    Returns:
        AgeEstimationModel instance
    """
    model = AgeEstimationModel(
        model_name=model_name,
        pretrained=pretrained,
        num_outputs=1
    )
    
    model = model.to(device)
    
    return model


def get_all_models(
    pretrained: bool = True,
    device: str = "cuda"
) -> Dict[str, AgeEstimationModel]:
    """
    Get all five models for comparison.
    """
    model_names = [
        "resnet50",
        "efficientnet_b7",
        "vit_base_patch16_224",
        "vit_large_patch16_224",
        "swin_base_patch4_window7_224"
    ]
    
    models = {}
    for name in model_names:
        print(f"Loading {MODEL_DISPLAY_NAMES.get(name, name)}...")
        models[name] = get_model(name, pretrained, device)
    
    return models


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable
    }


def get_model_info(model_name: str, pretrained: bool = True) -> Dict[str, Any]:
    """
    Get model information without loading to GPU.
    """
    model = AgeEstimationModel(model_name, pretrained=False)
    params = count_parameters(model)
    
    return {
        "name": model_name,
        "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
        "parameters": params
    }


def print_model_summary():
    """
    Print summary of all models.
    """
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    
    model_names = [
        "resnet50",
        "efficientnet_b7",
        "vit_base_patch16_224",
        "vit_large_patch16_224",
        "swin_base_patch4_window7_224"
    ]
    
    for name in model_names:
        info = get_model_info(name)
        print(f"\n{info['display_name']}:")
        print(f"  Total Parameters: {info['parameters']['total']:,}")
        print(f"  Trainable Parameters: {info['parameters']['trainable']:,}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print_model_summary()
