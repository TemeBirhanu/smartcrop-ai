"""
MobileNetV3 classifier for on-device disease classification
Uses PRETRAINED weights from ImageNet - only fine-tunes classification head
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class MobileNetV3Classifier(nn.Module):
    """
    MobileNetV3-Large classifier for crop disease detection.
    Uses pretrained ImageNet weights - only trains the last classification layers.
    Lightweight model for mobile/edge device inference.
    Target size: ~5MB when converted to TFLite.
    """
    
    def __init__(
        self,
        num_classes: int = 38,
        pretrained: bool = True,  # Load ImageNet pretrained weights
        dropout: float = 0.2
    ):
        """
        Initialize MobileNetV3 classifier.
        
        Args:
            num_classes: Number of disease classes
            pretrained: Use ImageNet pretrained weights (default: True)
            dropout: Dropout rate
        """
        super().__init__()
        
        # Load pretrained MobileNetV3-Large (trained on ImageNet)
        # This gives us powerful feature extraction without training from scratch
        if pretrained:
            # Use weights parameter for newer torchvision
            try:
                from torchvision.models import MobileNet_V3_Large_Weights
                self.backbone = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            except (ImportError, AttributeError):
                # Fallback for older torchvision
                self.backbone = models.mobilenet_v3_large(pretrained=True)
        else:
            self.backbone = models.mobilenet_v3_large(pretrained=False)
        
        # Replace only the classification head
        # Backbone stays frozen, only this head will be trained
        # MobileNetV3-Large classifier structure: Sequential(
        #   Linear(960, 1280),  # First linear layer
        #   Hardswish(),
        #   Dropout(),
        #   Linear(1280, num_classes)  # Last linear layer
        # )
        # We need to get the input features from the FIRST Linear layer
        original_classifier = self.backbone.classifier
        
        # Get input features from the first Linear layer
        if isinstance(original_classifier, nn.Sequential):
            # Find the first Linear layer
            in_features = None
            for layer in original_classifier:
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
        else:
            # If not Sequential, try to get from first layer
            if hasattr(original_classifier, '__getitem__'):
                first_layer = original_classifier[0]
                if isinstance(first_layer, nn.Linear):
                    in_features = first_layer.in_features
                else:
                    in_features = 960  # MobileNetV3-Large default
            else:
                in_features = 960
        
        # Default to 960 if not found (MobileNetV3-Large standard input to classifier)
        if in_features is None:
            in_features = 960
        
        # Replace classifier with new one
        # Input: 960 features from backbone, Output: num_classes
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(128, num_classes)
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

