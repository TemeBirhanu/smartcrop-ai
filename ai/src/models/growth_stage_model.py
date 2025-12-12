"""
Growth stage classifier using ResNet50 (AgML)
Uses PRETRAINED weights from ImageNet/AgML - only fine-tunes classification head
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class GrowthStageClassifier(nn.Module):
    """
    ResNet50-based classifier for crop growth stage prediction.
    Uses ImageNet/AgML pretrained weights - only trains the classification head.
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,  # Load ImageNet pretrained weights
        dropout: float = 0.2
    ):
        """
        Initialize growth stage classifier.
        
        Args:
            num_classes: Number of growth stages (typically 4: 
                        Seedling, Vegetative, Flowering, Maturity)
            pretrained: Use ImageNet/AgML pretrained weights (default: True)
            dropout: Dropout rate
        """
        super().__init__()
        
        # Load pretrained ResNet50 (trained on ImageNet)
        # Can later load AgML weights if available
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace only the classification head
        # Backbone stays frozen, only this head will be trained
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Logits (B, num_classes)
        """
        return self.backbone(x)
    
    def get_stage_name(self, idx: int) -> str:
        """
        Get growth stage name from index.
        
        Args:
            idx: Class index
        
        Returns:
            Stage name
        """
        stages = ["Seedling", "Vegetative", "Flowering", "Maturity"]
        if 0 <= idx < len(stages):
            return stages[idx]
        return "Unknown"

