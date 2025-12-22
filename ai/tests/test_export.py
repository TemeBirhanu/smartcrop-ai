"""
Tests for model export functionality
"""

import pytest
import torch
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.classifier_mobilenet import MobileNetV3Classifier
from src.export.export_onnx import export_to_onnx


class TestExport:
    """Test model export functions"""
    
    def create_dummy_model(self, num_classes=10):
        """Create a dummy model for testing"""
        model = MobileNetV3Classifier(
            num_classes=num_classes,
            pretrained=False
        )
        model.eval()
        return model
    
    def test_onnx_export(self, tmp_path):
        """Test ONNX export"""
        model = self.create_dummy_model(num_classes=5)
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Export to ONNX
        output_path = str(tmp_path / "test_model.onnx")
        
        try:
            export_to_onnx(
                model=model,
                dummy_input=dummy_input,
                output_path=output_path,
                opset_version=11
            )
            
            # Check if file was created
            assert os.path.exists(output_path)
            
        except ImportError:
            # ONNX might not be installed in test environment
            pytest.skip("ONNX not available")
    
    def test_model_export_structure(self):
        """Test that export functions exist and are callable"""
        from src.export.export_utils import export_model
        
        # Just check that the function exists
        assert callable(export_model)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

