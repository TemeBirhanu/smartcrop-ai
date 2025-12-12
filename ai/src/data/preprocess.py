"""
Image preprocessing utilities
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Union
import torch
from torchvision import transforms


def preprocess_image(
    image: Union[np.ndarray, Image.Image, str],
    size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> torch.Tensor:
    """
    Preprocess image for model inference.
    
    Args:
        image: Input image (numpy array, PIL Image, or file path)
        size: Target size (height, width)
        normalize: Whether to normalize using ImageNet stats
    
    Returns:
        Preprocessed tensor ready for model input
    """
    # Load image if path provided
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize
    transform_list = [transforms.Resize(size)]
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize (ImageNet stats)
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    transform = transforms.Compose(transform_list)
    return transform(image).unsqueeze(0)  # Add batch dimension


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image array
    
    Returns:
        Normalized image
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    if image.max() > 1.0:
        image = image / 255.0
    
    return image


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image using OpenCV.
    
    Args:
        image: Input image
        size: Target size (width, height)
        interpolation: Interpolation method
    
    Returns:
        Resized image
    """
    return cv2.resize(image, size, interpolation=interpolation)


def clean_image(image: np.ndarray) -> np.ndarray:
    """
    Clean image (remove noise, enhance contrast).
    
    Args:
        image: Input image
    
    Returns:
        Cleaned image
    """
    # Convert to LAB color space for better processing
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return image

