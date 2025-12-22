"""
Tests for model loading and initialization
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.classifier_mobilenet import MobileNetV3Classifier
from src.models.classifier_effnet import EfficientNetClassifier
from src.models.growth_stage_model import GrowthStageClassifier
from src.models.model_utils import (
    freeze_backbone,
    load_pretrained_weights,
    count_parameters,
    get_model_size_mb
)


class TestModelCreation:
    """Test model creation and initialization"""
    
    def test_mobilenet_creation(self):
        """Test MobileNetV3 model creation"""
        num_classes = 10
        model = MobileNetV3Classifier(
            num_classes=num_classes,
            pretrained=True
        )
        
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
            assert output.shape == (1, num_classes)
    
    def test_efficientnet_creation(self):
        """Test EfficientNet-B3 model creation"""
        num_classes = 10
        model = EfficientNetClassifier(
            num_classes=num_classes,
            pretrained=True
        )
        
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
            assert output.shape == (1, num_classes)
    
    def test_growth_stage_model_creation(self):
        """Test growth stage model creation"""
        num_stages = 5
        model = GrowthStageClassifier(
            num_stages=num_stages,
            pretrained=True
        )
        
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
            assert output.shape == (1, num_stages)


class TestModelUtils:
    """Test model utility functions"""
    
    def test_freeze_backbone(self):
        """Test freezing backbone layers"""
        model = MobileNetV3Classifier(num_classes=10, pretrained=False)
        
        # Count trainable params before freezing
        trainable_before = count_parameters(model, trainable_only=True)
        
        # Freeze backbone
        model = freeze_backbone(model, freeze=True)
        
        # Count trainable params after freezing
        trainable_after = count_parameters(model, trainable_only=True)
        
        # After freezing, only classifier should be trainable
        assert trainable_after < trainable_before
        assert trainable_after > 0  # Classifier should still be trainable
    
    def test_count_parameters(self):
        """Test parameter counting"""
        model = MobileNetV3Classifier(num_classes=10, pretrained=False)
        
        total = count_parameters(model, trainable_only=False)
        trainable = count_parameters(model, trainable_only=True)
        
        assert total > 0
        assert trainable > 0
        assert trainable <= total
    
    def test_get_model_size(self):
        """Test model size calculation"""
        model = MobileNetV3Classifier(num_classes=10, pretrained=False)
        
        size_mb = get_model_size_mb(model)
        assert size_mb > 0
        assert isinstance(size_mb, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

