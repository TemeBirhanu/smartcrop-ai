"""
Tests for inference pipeline
"""

import pytest
import torch
import numpy as np
import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.inference.predict_image import predict_disease
from src.inference.postprocess import calculate_severity, format_predictions
from src.models.classifier_mobilenet import MobileNetV3Classifier


class TestInference:
    """Test inference functions"""
    
    def create_dummy_model(self, num_classes=10):
        """Create a dummy model for testing"""
        model = MobileNetV3Classifier(
            num_classes=num_classes,
            pretrained=False
        )
        model.eval()
        return model
    
    def create_dummy_image(self, size=(224, 224)):
        """Create a dummy image"""
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        return img
    
    def test_predict_disease(self):
        """Test disease prediction"""
        model = self.create_dummy_model(num_classes=5)
        image = self.create_dummy_image()
        class_names = [f"class_{i}" for i in range(5)]
        
        device = "cpu"
        prediction = predict_disease(
            model=model,
            image=image,
            class_names=class_names,
            device=device,
            top_k=3
        )
        
        assert prediction is not None
        assert 'predicted_class' in prediction
        assert 'confidence' in prediction
        assert 'top_k' in prediction
        assert isinstance(prediction['confidence'], float)
        assert 0 <= prediction['confidence'] <= 1
    
    def test_format_predictions(self):
        """Test prediction formatting"""
        prediction = {
            'predicted_class': 'leaf_blight',
            'confidence': 0.855,
            'top_k': [
                {'class': 'leaf_blight', 'confidence': 0.855},
                {'class': 'healthy', 'confidence': 0.10},
                {'class': 'rust', 'confidence': 0.045}
            ]
        }
        
        formatted = format_predictions(prediction)
        
        assert formatted is not None
        assert 'disease' in formatted
        assert 'confidence' in formatted
        assert 'top_alternatives' in formatted
    
    def test_calculate_severity(self):
        """Test severity calculation"""
        # Mock detection results
        lesion_count = 5
        leaf_area = 10000.0  # pixels
        affected_area = 1500.0  # pixels
        
        severity = calculate_severity(
            lesion_count=lesion_count,
            leaf_area=leaf_area,
            affected_area=affected_area
        )
        
        assert severity is not None
        assert isinstance(severity, dict)
        assert 'severity_level' in severity
        assert 'affected_percentage' in severity
        assert 'lesion_count' in severity


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

