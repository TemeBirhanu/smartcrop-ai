"""
PyTorch Dataset class for crop disease classification
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A


class CropDiseaseDataset(Dataset):
    """
    Dataset class for crop disease classification.
    
    Expected directory structure:
    data/raw/
    ├── crop_name/
    │   ├── disease_class_1/
    │   ├── disease_class_2/
    │   └── ...
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        crops: Optional[List[str]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing crop folders
            split: "train", "val", or "test"
            transform: Albumentations transform
            crops: List of crops to include (None = all)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load data
        self.samples = self._load_samples(crops)
        self.class_to_idx = self._build_class_mapping()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
    def _load_samples(self, crops: Optional[List[str]]) -> List[Tuple[str, str]]:
        """
        Load image paths and labels.
        
        Returns:
            List of (image_path, class_name) tuples
        """
        samples = []
        
        # Get crop directories
        crop_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if crops:
            crop_dirs = [d for d in crop_dirs if d.name in crops]
        
        # Load images from each crop/disease folder
        for crop_dir in crop_dirs:
            for disease_dir in crop_dir.iterdir():
                if not disease_dir.is_dir():
                    continue
                
                # Get all images in this disease folder
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                images = [
                    img for img in disease_dir.iterdir()
                    if img.suffix.lower() in image_extensions
                ]
                
                # Create samples: (image_path, class_name)
                class_name = f"{crop_dir.name}_{disease_dir.name}"
                for img_path in images:
                    samples.append((str(img_path), class_name))
        
        return samples
    
    def _build_class_mapping(self) -> dict:
        """Build class name to index mapping."""
        classes = sorted(set([label for _, label in self.samples]))
        return {cls: idx for idx, cls in enumerate(classes)}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Returns:
            (image_tensor, class_index)
        """
        img_path, class_name = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default: convert to tensor
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            image = transform(image)
        
        # Get class index
        class_idx = self.class_to_idx[class_name]
        
        return image, class_idx
    
    def get_class_name(self, idx: int) -> str:
        """Get class name from index."""
        return self.idx_to_class[idx]
    
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.class_to_idx)

