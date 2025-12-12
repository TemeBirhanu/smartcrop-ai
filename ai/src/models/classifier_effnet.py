"""
EfficientNet classifier for disease classification
Uses PRETRAINED weights from ImageNet - only fine-tunes classification head
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B3 classifier for crop disease detection.
    Uses pretrained ImageNet weights - only trains the last classification layers.
    Used for high-accuracy server-side inference.
    """
    
    def __init__(
        self,
        num_classes: int = 38,
        pretrained: bool = True,  # Load ImageNet pretrained weights
        dropout: float = 0.3
    ):
        """
        Initialize EfficientNet classifier.
        
        Args:
            num_classes: Number of disease classes
            pretrained: Use ImageNet pretrained weights (default: True)
            dropout: Dropout rate
        """
        super().__init__()
        
        # Load pretrained EfficientNet-B3 (trained on ImageNet)
        # This gives us powerful feature extraction without training from scratch
        self.backbone = models.efficientnet_b3(pretrained=pretrained)
        
        # Replace only the classification head
        # Backbone stays frozen, only this head will be trained
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(512, num_classes)
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
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head.
        Useful for Grad-CAM.
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor
        """
        # Get features from backbone
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

