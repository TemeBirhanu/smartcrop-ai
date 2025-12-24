"""
Grad-CAM heatmap generation for explainability
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
from PIL import Image

from ..models.classifier_effnet import EfficientNetClassifier
from ..models.classifier_mobilenet import MobileNetV3Classifier


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Generates heatmaps showing which parts of the image the model focuses on.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained model
            target_layer: Name of target layer (if None, uses last conv layer)
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        
        # Register hooks
        self.gradients = None
        self.activations = None
        
        self.hook_layers()
    
    def hook_layers(self):
        """Register forward and backward hooks."""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # Find target layer
        if self.target_layer:
            target = dict(self.model.named_modules())[self.target_layer]
        else:
            # Find last convolutional layer
            target = None
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target = module
        
        if target:
            target.register_forward_hook(forward_hook)
            target.register_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Class index to generate heatmap for (None = predicted class)
        
        Returns:
            Heatmap as numpy array (H, W)
        """
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Get device from activations
        device = activations.device
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Generate heatmap on the same device as activations
        heatmap = torch.zeros(activations.shape[1:], device=device)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
        
        # Apply ReLU
        heatmap = F.relu(heatmap)
        
        # Normalize (detach and move to CPU after computation)
        heatmap = heatmap.detach().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap


def generate_gradcam_heatmap(
    model: torch.nn.Module,
    image: np.ndarray,
    class_idx: Optional[int] = None,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM heatmap for an image.
    
    Args:
        model: Trained model
        image: Input image (numpy array)
        class_idx: Class index for heatmap (None = predicted class)
        device: Device to run on
    
    Returns:
        Tuple of (heatmap, overlay_image)
        - heatmap: Raw heatmap (H, W)
        - overlay_image: Image with heatmap overlay (H, W, 3)
    """
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer=None)
    
    # Preprocess image
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
    else:
        image_rgb = np.array(image)
    
    # Convert to tensor
    from ..data.transforms import get_inference_transforms
    transform = get_inference_transforms()
    
    # Ensure image is numpy array (RGB)
    if isinstance(image_rgb, Image.Image):
        image_rgb = np.array(image_rgb.convert('RGB'))
    elif not isinstance(image_rgb, np.ndarray):
        raise ValueError(f"Unsupported image type: {type(image_rgb)}")
    
    # Apply transforms (Albumentations expects dict with 'image' key)
    transformed = transform(image=image_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Get original image size
    orig_h, orig_w = image_rgb.shape[:2]
    
    # Generate heatmap
    heatmap = gradcam.generate(input_tensor, class_idx)
    
    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
    
    # Convert heatmap to RGB
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image
    overlay = cv2.addWeighted(
        image_rgb.astype(np.uint8),
        0.6,
        heatmap_colored,
        0.4,
        0
    )
    
    return heatmap_resized, overlay

