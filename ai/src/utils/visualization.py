"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import torch


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training curves (loss and metrics).
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_metrics: Dictionary of training metrics (optional)
        val_metrics: Dictionary of validation metrics (optional)
        save_path: Path to save the plot (optional)
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create subplots
    num_plots = 1
    if train_metrics or val_metrics:
        num_plots = 2
    
    fig, axes = plt.subplots(1, num_plots, figsize=(12, 4))
    if num_plots == 1:
        axes = [axes]
    
    # Plot losses
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics if provided
    if num_plots == 2 and (train_metrics or val_metrics):
        metrics_to_plot = ['accuracy', 'f1_score']  # Common metrics
        
        for metric in metrics_to_plot:
            if train_metrics and metric in train_metrics:
                axes[1].plot(epochs, train_metrics[metric], 
                            label=f'Train {metric}', linewidth=2)
            if val_metrics and metric in val_metrics:
                axes[1].plot(epochs, val_metrics[metric], 
                            label=f'Val {metric}', linewidth=2)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric')
        axes[1].set_title('Training and Validation Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_predictions(
    images: List[np.ndarray],
    predictions: List[Dict],
    class_names: List[str],
    num_samples: int = 8,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize predictions on images.
    
    Args:
        images: List of input images
        predictions: List of prediction dictionaries
        class_names: List of class names
        num_samples: Number of samples to visualize
        save_path: Path to save the plot (optional)
    """
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Display image
        img = images[i]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            if img.shape[0] == 3:  # CHW format
                img = img.transpose(1, 2, 0)
            # Denormalize if needed
            if img.min() < 0:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
                img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        # Add prediction text
        pred = predictions[i]
        pred_text = f"{pred['predicted_class']}\n"
        pred_text += f"Confidence: {pred['confidence']:.2%}"
        
        ax.text(0.02, 0.98, pred_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

