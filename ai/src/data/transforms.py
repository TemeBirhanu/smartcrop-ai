"""
Data augmentation transforms using Albumentations
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any


def get_train_transforms(config: Dict[str, Any] = None) -> A.Compose:
    """
    Get training data augmentation transforms.
    
    Args:
        config: Configuration dictionary with augmentation parameters
    
    Returns:
        Albumentations compose object
    """
    if config is None:
        config = {
            "horizontal_flip": 0.5,
            "vertical_flip": 0.3,
            "rotation": 15,
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1
            }
        }
    
    transforms_list = [
        A.HorizontalFlip(p=config.get("horizontal_flip", 0.5)),
        A.VerticalFlip(p=config.get("vertical_flip", 0.3)),
        A.Rotate(limit=config.get("rotation", 15), p=0.5),
    ]
    
    # Color jitter
    if "color_jitter" in config:
        cj = config["color_jitter"]
        transforms_list.append(
            A.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
                saturation=cj.get("saturation", 0.2),
                hue=cj.get("hue", 0.1),
                p=0.5
            )
        )
    
    # Normalization and tensor conversion
    transforms_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list)


def get_val_transforms() -> A.Compose:
    """
    Get validation/test transforms (minimal augmentation).
    
    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_inference_transforms() -> A.Compose:
    """
    Get inference transforms (same as validation - no augmentation).
    
    Returns:
        Albumentations compose object
    """
    return get_val_transforms()

