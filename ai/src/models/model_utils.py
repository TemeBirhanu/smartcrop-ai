"""
Model utility functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def freeze_backbone(model: nn.Module, freeze: bool = True) -> nn.Module:
    """
    Freeze or unfreeze backbone layers.
    Only the classification head will be trainable.
    
    Args:
        model: PyTorch model
        freeze: If True, freeze backbone; if False, unfreeze
    
    Returns:
        Model with frozen/unfrozen backbone
    """
    for name, param in model.named_parameters():
        # Freeze backbone (everything except classifier head)
        if 'classifier' not in name and 'fc' not in name:
            param.requires_grad = not freeze
        else:
            param.requires_grad = freeze
    
    return model


def load_pretrained_weights(
    model: nn.Module,
    weights_path: Optional[str] = None,
    strict: bool = True
) -> nn.Module:
    """
    Load pretrained weights into model.
    
    Args:
        model: PyTorch model
        weights_path: Path to weights file (None = use default pretrained)
        strict: Whether to strictly match state dict keys
    
    Returns:
        Model with loaded weights
    """
    if weights_path:
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=strict)
    else:
        # Use default pretrained weights (handled by model definition)
        pass
    
    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count number of parameters in model.
    
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


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in megabytes.
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

