"""
Learning rate schedulers
"""

import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR
)
from typing import Optional


def get_scheduler(
    optimizer: optim.Optimizer,
    num_epochs: int,
    scheduler_type: str = "cosine",
    **kwargs
):
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        num_epochs: Total number of epochs
        scheduler_type: Type of scheduler ("cosine", "step", "plateau", "onecycle")
        **kwargs: Additional scheduler parameters
    
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        # Cosine annealing - good for fine-tuning
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    
    elif scheduler_type == "step":
        # Step decay
        step_size = kwargs.get('step_size', num_epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == "plateau":
        # Reduce on plateau
        return ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 3),
            verbose=True
        )
    
    elif scheduler_type == "onecycle":
        # One cycle policy
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 0.01),
            epochs=num_epochs,
            steps_per_epoch=kwargs.get('steps_per_epoch', 100)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

