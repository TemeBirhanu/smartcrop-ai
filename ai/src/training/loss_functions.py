"""
Loss functions for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Useful when some diseases are rare in your dataset.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning.
    Can combine classification loss with other losses if needed.
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        use_focal: bool = False
    ):
        """
        Initialize combined loss.
        
        Args:
            classification_weight: Weight for classification loss
            use_focal: Use Focal Loss instead of CrossEntropy
        """
        super().__init__()
        self.classification_weight = classification_weight
        
        if use_focal:
            self.classification_loss = FocalLoss()
        else:
            self.classification_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
        
        Returns:
            Combined loss value
        """
        loss = self.classification_weight * self.classification_loss(
            predictions, targets
        )
        return loss

