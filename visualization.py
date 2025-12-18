"""
Visualization Module for Age Estimation Models
- Grad-CAM, Grad-CAM++, Score-CAM
- Attention Map Visualization for ViT/Swin
- Comparison across models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import cv2
from tqdm import tqdm


class GradCAM:
    """
    Grad-CAM implementation for CNN and Transformer models.
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks 
    via Gradient-based Localization", ICCV 2017
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        use_cuda: bool = True
    ):
        self.model = model
        self.target_layer = target_layer
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Save forward activation."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Save backward gradient."""
        self.gradients = grad_output[0].detach()
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for classification (None for regression)
            
        Returns:
            cam: Grad-CAM heatmap (H, W)
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # For regression, use output directly
        if target_class is None:
            target = output
        else:
            target = output[:, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Get weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Resize to input size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation.
    Reference: Chattopadhyay et al., "Grad-CAM++: Improved Visual Explanations for 
    Deep Convolutional Networks", WACV 2018
    """
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target = output
        else:
            target = output[:, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Grad-CAM++ weights
        gradients = self.gradients
        activations = self.activations
        
        # Alpha calculation
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # Weights
        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)
        
        # CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Resize and normalize
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


class ScoreCAM(GradCAM):
    """
    Score-CAM implementation (gradient-free).
    Reference: Wang et al., "Score-CAM: Score-Weighted Visual Explanations for 
    Convolutional Neural Networks", CVPR 2020
    """
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        batch_size: int = 16
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)
        
        # Forward to get activations
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        activations = self.activations
        b, c, h, w = activations.shape
        
        # Upsample activations to input size
        upsampled = F.interpolate(
            activations, 
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize each channel
        upsampled = upsampled.reshape(c, -1)
        upsampled = upsampled - upsampled.min(dim=1, keepdim=True)[0]
        upsampled = upsampled / (upsampled.max(dim=1, keepdim=True)[0] + 1e-8)
        upsampled = upsampled.reshape(1, c, input_tensor.shape[2], input_tensor.shape[3])
        
        # Get scores for each masked input
        scores = []
        with torch.no_grad():
            for i in range(0, c, batch_size):
                batch_masks = upsampled[0, i:i+batch_size].unsqueeze(1)
                masked_inputs = input_tensor * batch_masks
                outputs = self.model(masked_inputs)
                scores.extend(outputs.cpu().numpy())
        
        scores = np.array(scores)
        scores = scores - scores.min()
        scores = scores / (scores.max() + 1e-8)
        
        # Weighted sum
        cam = np.zeros((h, w))
        for i, score in enumerate(scores):
            cam += score * activations[0, i].cpu().numpy()
        
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


class AttentionRollout:
    """
    Attention Rollout for Vision Transformers.
    Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020
    """
    
    def __init__(self, model: nn.Module, use_cuda: bool = True):
        self.model = model
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        discard_ratio: float = 0.9,
        head_fusion: str = "mean"
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)
        
        # Get attention weights from all layers
        attention_weights = []
        
        def hook_fn(module, input, output):
            # For ViT attention layers
            if hasattr(output, 'shape') and len(output.shape) == 3:
                attention_weights.append(output.detach())
        
        hooks = []
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() and 'drop' not in name.lower():
                hooks.append(module.register_forward_hook(hook_fn))
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if len(attention_weights) == 0:
            # Fallback: return uniform attention
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]))
        
        # Process attention weights
        result = torch.eye(attention_weights[0].shape[-1]).to(self.device)
        
        for attention in attention_weights:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(dim=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown head fusion: {head_fusion}")
            
            # Add identity for residual connection
            I = torch.eye(attention_heads_fused.shape[-1]).to(self.device)
            attention_heads_fused = (attention_heads_fused + I) / 2
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
            
            result = torch.matmul(attention_heads_fused, result)
        
        # Get CLS token attention
        mask = result[0, 0, 1:]  # Exclude CLS token
        
        # Reshape to 2D
        num_patches = int(np.sqrt(mask.shape[0]))
        mask = mask.reshape(num_patches, num_patches).cpu().numpy()
        
        # Resize to input size
        mask = cv2.resize(mask, (input_tensor.shape[3], input_tensor.shape[2]))
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        return mask


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Get the target layer for Grad-CAM based on model architecture.
    """
    if 'resnet' in model_name.lower():
        # Last conv layer before avgpool
        return model.backbone.layer4[-1]
    elif 'efficientnet' in model_name.lower():
        # Last conv block
        return model.backbone.conv_head
    elif 'vit' in model_name.lower():
        # Last transformer block
        return model.backbone.blocks[-1].norm1
    elif 'swin' in model_name.lower():
        # Last layer norm
        return model.backbone.layers[-1].blocks[-1].norm1
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


def visualize_cam(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Overlay CAM heatmap on original image.
    
    Args:
        image: Original image (H, W, 3) in range [0, 1]
        cam: CAM heatmap (H, W) in range [0, 1]
        alpha: Overlay transparency
        colormap: Matplotlib colormap name
        
    Returns:
        visualization: Overlayed image (H, W, 3) in range [0, 1]
    """
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(cam)[:, :, :3]
    
    # Overlay
    visualization = alpha * heatmap + (1 - alpha) * image
    visualization = np.clip(visualization, 0, 1)
    
    return visualization


def generate_cam_comparison(
    model: nn.Module,
    model_name: str,
    image_path: str,
    transform: Callable,
    output_dir: str,
    methods: List[str] = ['gradcam', 'gradcam++', 'scorecam'],
    device: str = 'cuda'
):
    """
    Generate CAM visualizations using multiple methods for a single image.
    """
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    original_np = np.array(original_image) / 255.0
    
    input_tensor = transform(original_image).unsqueeze(0)
    
    # Get target layer
    target_layer = get_target_layer(model, model_name)
    
    # Generate CAMs
    cams = {}
    
    if 'gradcam' in methods:
        gradcam = GradCAM(model, target_layer)
        cams['Grad-CAM'] = gradcam(input_tensor)
    
    if 'gradcam++' in methods:
        gradcampp = GradCAMPlusPlus(model, target_layer)
        cams['Grad-CAM++'] = gradcampp(input_tensor)
    
    if 'scorecam' in methods:
        scorecam = ScoreCAM(model, target_layer)
        cams['Score-CAM'] = scorecam(input_tensor)
    
    # Resize original image to match CAM size
    original_resized = cv2.resize(original_np, (224, 224))
    
    # Create visualization
    n_methods = len(cams)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 4))
    
    # Original image
    axes[0].imshow(original_resized)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # CAM visualizations
    for i, (method_name, cam) in enumerate(cams.items()):
        vis = visualize_cam(original_resized, cam)
        axes[i + 1].imshow(vis)
        axes[i + 1].set_title(method_name)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(image_path).stem
    save_path = output_path / f"{model_name}_{image_name}_cam_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved CAM comparison to {save_path}")
    
    return cams


def generate_model_comparison(
    models: Dict[str, nn.Module],
    image_path: str,
    transform: Callable,
    output_dir: str,
    method: str = 'gradcam',
    device: str = 'cuda'
):
    """
    Compare attention regions across different models for the same image.
    """
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    original_np = np.array(original_image) / 255.0
    original_resized = cv2.resize(original_np, (224, 224))
    
    input_tensor = transform(original_image).unsqueeze(0)
    
    # Generate CAMs for each model
    cams = {}
    predictions = {}
    
    for model_name, model in models.items():
        model.to(device)
        model.eval()
        
        # Get prediction
        with torch.no_grad():
            pred = model(input_tensor.to(device)).item()
        predictions[model_name] = pred
        
        # Get CAM
        try:
            target_layer = get_target_layer(model, model_name)
            
            if method == 'gradcam':
                cam_generator = GradCAM(model, target_layer)
            elif method == 'gradcam++':
                cam_generator = GradCAMPlusPlus(model, target_layer)
            else:
                cam_generator = ScoreCAM(model, target_layer)
            
            cams[model_name] = cam_generator(input_tensor)
        except Exception as e:
            print(f"Warning: Could not generate CAM for {model_name}: {e}")
            cams[model_name] = np.zeros((224, 224))
    
    # Create visualization
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(4 * (n_models + 1), 8))
    
    # Original image
    axes[0, 0].imshow(original_resized)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Model visualizations
    for i, (model_name, cam) in enumerate(cams.items()):
        # CAM heatmap
        axes[0, i + 1].imshow(cam, cmap='jet')
        axes[0, i + 1].set_title(f'{model_name}\nPred: {predictions[model_name]:.1f}')
        axes[0, i + 1].axis('off')
        
        # Overlay
        vis = visualize_cam(original_resized, cam)
        axes[1, i + 1].imshow(vis)
        axes[1, i + 1].set_title('Overlay')
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(image_path).stem
    save_path = output_path / f"model_comparison_{image_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved model comparison to {save_path}")
    
    return cams, predictions


def batch_generate_cams(
    model: nn.Module,
    model_name: str,
    dataloader: DataLoader,
    output_dir: str,
    method: str = 'gradcam',
    max_samples: int = 50,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Generate CAMs for a batch of images.
    
    Returns:
        results: Dictionary with CAMs, predictions, and statistics
    """
    model.to(device)
    model.eval()
    
    target_layer = get_target_layer(model, model_name)
    
    if method == 'gradcam':
        cam_generator = GradCAM(model, target_layer)
    elif method == 'gradcam++':
        cam_generator = GradCAMPlusPlus(model, target_layer)
    else:
        cam_generator = ScoreCAM(model, target_layer)
    
    output_path = Path(output_dir) / model_name / 'cams'
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_cams = []
    all_predictions = []
    all_targets = []
    
    sample_count = 0
    
    for batch_idx, (images, ages) in enumerate(tqdm(dataloader, desc="Generating CAMs")):
        for i in range(images.shape[0]):
            if sample_count >= max_samples:
                break
            
            input_tensor = images[i:i+1]
            
            # Get prediction
            with torch.no_grad():
                pred = model(input_tensor.to(device)).item()
            
            # Generate CAM
            cam = cam_generator(input_tensor)
            
            all_cams.append(cam)
            all_predictions.append(pred)
            all_targets.append(ages[i].item())
            
            sample_count += 1
        
        if sample_count >= max_samples:
            break
    
    # Calculate average CAM
    avg_cam = np.mean(all_cams, axis=0)
    
    # Save average CAM
    plt.figure(figsize=(6, 6))
    plt.imshow(avg_cam, cmap='jet')
    plt.colorbar()
    plt.title(f'{model_name} - Average Attention Map')
    plt.axis('off')
    plt.savefig(output_path / 'average_cam.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'cams': all_cams,
        'predictions': all_predictions,
        'targets': all_targets,
        'average_cam': avg_cam
    }


def analyze_attention_by_age_group(
    model: nn.Module,
    model_name: str,
    dataloader: DataLoader,
    output_dir: str,
    age_bins: List[int] = [20, 40, 60, 80],
    method: str = 'gradcam',
    max_per_group: int = 20,
    device: str = 'cuda'
):
    """
    Analyze attention patterns by age group.
    """
    model.to(device)
    model.eval()
    
    target_layer = get_target_layer(model, model_name)
    
    if method == 'gradcam':
        cam_generator = GradCAM(model, target_layer)
    else:
        cam_generator = GradCAMPlusPlus(model, target_layer)
    
    # Collect CAMs by age group
    age_group_cams = {f"{age_bins[i]}-{age_bins[i+1]}": [] 
                      for i in range(len(age_bins)-1)}
    
    for images, ages in tqdm(dataloader, desc="Collecting CAMs"):
        for i in range(images.shape[0]):
            age = ages[i].item()
            
            # Find age group
            group = None
            for j in range(len(age_bins)-1):
                if age_bins[j] <= age < age_bins[j+1]:
                    group = f"{age_bins[j]}-{age_bins[j+1]}"
                    break
            
            if group is None or len(age_group_cams[group]) >= max_per_group:
                continue
            
            # Generate CAM
            input_tensor = images[i:i+1]
            cam = cam_generator(input_tensor)
            age_group_cams[group].append(cam)
    
    # Calculate average CAM per age group
    output_path = Path(output_dir) / model_name / 'age_group_analysis'
    output_path.mkdir(parents=True, exist_ok=True)
    
    avg_cams = {}
    for group, cams in age_group_cams.items():
        if len(cams) > 0:
            avg_cams[group] = np.mean(cams, axis=0)
    
    # Visualization
    n_groups = len(avg_cams)
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4))
    
    if n_groups == 1:
        axes = [axes]
    
    for i, (group, avg_cam) in enumerate(avg_cams.items()):
        axes[i].imshow(avg_cam, cmap='jet')
        axes[i].set_title(f'Age {group}\n(n={len(age_group_cams[group])})')
        axes[i].axis('off')
    
    plt.suptitle(f'{model_name} - Attention by Age Group')
    plt.tight_layout()
    plt.savefig(output_path / 'age_group_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved age group analysis to {output_path}")
    
    return avg_cams


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    output_dir: str
):
    """
    Plot prediction error distribution.
    """
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Error histogram
    axes[0, 0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='r', linestyle='--', label='Zero error')
    axes[0, 0].set_xlabel('Prediction Error (years)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Error Distribution')
    axes[0, 0].legend()
    
    # Absolute error histogram
    axes[0, 1].hist(abs_errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(np.mean(abs_errors), color='r', linestyle='--', 
                       label=f'MAE={np.mean(abs_errors):.2f}')
    axes[0, 1].set_xlabel('Absolute Error (years)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Absolute Error Distribution')
    axes[0, 1].legend()
    
    # Scatter plot
    axes[1, 0].scatter(targets, predictions, alpha=0.5)
    axes[1, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 
                    'r--', label='Perfect prediction')
    axes[1, 0].set_xlabel('True Age (years)')
    axes[1, 0].set_ylabel('Predicted Age (years)')
    axes[1, 0].set_title('Prediction vs True Age')
    axes[1, 0].legend()
    
    # Error vs True Age
    axes[1, 1].scatter(targets, errors, alpha=0.5)
    axes[1, 1].axhline(0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('True Age (years)')
    axes[1, 1].set_ylabel('Prediction Error (years)')
    axes[1, 1].set_title('Error vs True Age')
    
    plt.suptitle(f'{model_name} - Prediction Analysis')
    plt.tight_layout()
    plt.savefig(output_path / 'error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error distribution to {output_path}")


if __name__ == "__main__":
    print("Visualization module loaded successfully")
