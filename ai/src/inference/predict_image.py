"""
Image prediction functions
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import cv2

from ..models.classifier_effnet import EfficientNetClassifier
from ..models.classifier_mobilenet import MobileNetV3Classifier
from ..data.transforms import get_inference_transforms


def predict_disease(
    model: torch.nn.Module,
    image: np.ndarray,
    class_names: List[str],
    device: str = "cuda",
    top_k: int = 3
) -> Dict:
    """
    Predict disease from a single image.
    
    Args:
        model: Trained classifier model
        image: Input image (numpy array, BGR or RGB)
        class_names: List of class names
        device: Device to run inference on
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions:
        {
            'predicted_class': str,
            'confidence': float,
            'top_k': List[Dict] with 'class', 'confidence'
        }
    """
    model.eval()
    model = model.to(device)
    
    # Get transforms
    transform = get_inference_transforms()
    
    # Preprocess image - Albumentations expects numpy array
    if isinstance(image, np.ndarray):
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
    elif isinstance(image, Image.Image):
        image_rgb = np.array(image.convert('RGB'))
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Apply transforms (Albumentations expects dict with 'image' key)
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(class_names)))
    
    # Format results
    top_k_results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        top_k_results.append({
            'class': class_names[idx.item()],
            'confidence': prob.item()
        })
    
    predicted_class = top_k_results[0]['class']
    confidence = top_k_results[0]['confidence']
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top_k': top_k_results
    }


def predict_batch(
    model: torch.nn.Module,
    images: List[np.ndarray],
    class_names: List[str],
    device: str = "cuda",
    batch_size: int = 32
) -> List[Dict]:
    """
    Predict diseases from a batch of images.
    
    Args:
        model: Trained classifier model
        images: List of input images (numpy arrays)
        class_names: List of class names
        device: Device to run inference on
        batch_size: Batch size for inference
    
    Returns:
        List of prediction dictionaries
    """
    model.eval()
    model = model.to(device)
    
    # Get transforms
    transform = get_inference_transforms()
    
    results = []
    
    # Process in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        
        # Preprocess batch
        batch_tensors = []
        for img in batch_images:
            # Convert to RGB numpy array
            if isinstance(img, np.ndarray):
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = img
            elif isinstance(img, Image.Image):
                img_rgb = np.array(img.convert('RGB'))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            # Apply transforms (Albumentations expects dict with 'image' key)
            transformed = transform(image=img_rgb)
            batch_tensors.append(transformed['image'])
        
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Process results
        for prob in probabilities:
            top_probs, top_indices = torch.topk(prob, k=min(3, len(class_names)))
            
            top_k_results = []
            for p, idx in zip(top_probs, top_indices):
                top_k_results.append({
                    'class': class_names[idx.item()],
                    'confidence': p.item()
                })
            
            results.append({
                'predicted_class': top_k_results[0]['class'],
                'confidence': top_k_results[0]['confidence'],
                'top_k': top_k_results
            })
    
    return results

