"""
Helper utility functions
"""

import torch
from typing import Optional
import torch.nn as nn


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device (CUDA if available, else CPU).
    
    Args:
        device: Preferred device ("cuda" or "cpu"), None for auto-detect
    
    Returns:
        torch.device object
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device("cpu")
    
    return torch.device(device)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_model_size(model: nn.Module) -> str:
    """
    Get model size in human-readable format.
    
    Args:
        model: PyTorch model
    
    Returns:
        Formatted model size string
    """
    param_count = count_parameters(model)
    # Rough estimate: 4 bytes per float32 parameter
    size_bytes = param_count * 4
    return format_size(size_bytes)

