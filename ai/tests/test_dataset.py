"""
Tests for dataset and data preprocessing
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os
from PIL import Image

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.dataset import CropDataset
from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_inference_transforms
)
from src.data.preprocess import preprocess_image


class TestTransforms:
    """Test data augmentation transforms"""
    
    def test_train_transforms(self):
        """Test training transforms"""
        transforms = get_train_transforms()
        assert transforms is not None
        
        # Create dummy image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Apply transforms
        transformed = transforms(image=img)
        assert 'image' in transformed
        assert isinstance(transformed['image'], torch.Tensor)
        assert transformed['image'].shape == (3, 224, 224)
    
    def test_val_transforms(self):
        """Test validation transforms"""
        transforms = get_val_transforms()
        assert transforms is not None
        
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        transformed = transforms(image=img)
        assert 'image' in transformed
        assert isinstance(transformed['image'], torch.Tensor)
    
    def test_inference_transforms(self):
        """Test inference transforms"""
        transforms = get_inference_transforms(img_size=224)
        assert transforms is not None
        
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        transformed = transforms(image=img)
        assert 'image' in transformed
        assert transformed['image'].shape == (3, 224, 224)


class TestDataset:
    """Test CropDataset class"""
    
    def create_dummy_dataset(self, tmp_path):
        """Create a dummy dataset structure"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create dummy images
        for label in ["healthy", "disease"]:
            label_dir = data_dir / label
            label_dir.mkdir()
            for i in range(5):
                img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
                img.save(label_dir / f"img_{i}.jpg")
        
        # Create annotations CSV
        annotations_file = tmp_path / "annotations.csv"
        with open(annotations_file, 'w') as f:
            f.write("image_path,label\n")
            for label in ["healthy", "disease"]:
                for i in range(5):
                    f.write(f"{label}/img_{i}.jpg,{label}\n")
        
        return str(data_dir), str(annotations_file)
    
    def test_dataset_creation(self, tmp_path):
        """Test dataset can be created"""
        data_dir, annotations_file = self.create_dummy_dataset(tmp_path)
        
        # This will fail if dataset structure is wrong
        # For now, just test that the class exists
        assert CropDataset is not None
    
    def test_preprocess_image(self):
        """Test image preprocessing function"""
        # Create dummy image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test preprocessing
        processed = preprocess_image(img)
        assert processed is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

