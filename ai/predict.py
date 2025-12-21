"""
Main inference script for SmartCrop AI
Usage: python predict.py --image path/to/image.jpg --model outputs/models/checkpoints/mobilenet_v3_best.pth
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import cv2
import numpy as np
from src.inference.predict_image import predict_disease
from src.inference.heatmap_cam import generate_gradcam_heatmap
from src.inference.postprocess import format_predictions
from src.models.classifier_mobilenet import MobileNetV3Classifier
from src.models.classifier_effnet import EfficientNetClassifier
from src.utils.helpers import get_device
from src.utils.file_utils import load_yaml, load_json
from src.utils.logger import setup_logger


def load_model(model_type: str, checkpoint_path: str, num_classes: int, device: str):
    """Load model from checkpoint."""
    if model_type == 'mobilenet_v3':
        model = MobileNetV3Classifier(num_classes=num_classes, pretrained=False)
    elif model_type == 'efficientnet_b3':
        model = EfficientNetClassifier(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model = model.to(device)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Run inference with SmartCrop AI')
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['mobilenet_v3', 'efficientnet_b3'],
        default='mobilenet_v3',
        help='Type of model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/model.yaml',
        help='Path to model config file'
    )
    parser.add_argument(
        '--class-map',
        type=str,
        default='data/metadata/class_map.json',
        help='Path to class map JSON file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--heatmap',
        action='store_true',
        help='Generate Grad-CAM heatmap'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output image with heatmap'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(level='INFO')
    
    logger.info("=" * 60)
    logger.info("SmartCrop AI - Inference Script")
    logger.info("=" * 60)
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load class map
    try:
        class_map = load_json(args.class_map)
        class_names = [k for k in sorted(class_map.keys(), key=lambda x: class_map[x])]
        num_classes = len(class_names)
        logger.info(f"Loaded {num_classes} classes")
    except Exception as e:
        logger.error(f"Failed to load class map: {e}")
        logger.info("Using default class names")
        class_names = [f"Class_{i}" for i in range(10)]  # Default
        num_classes = 10
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    try:
        model = load_model(args.model_type, args.model, num_classes, device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Load image
    logger.info(f"Loading image from {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        logger.error(f"Failed to load image: {args.image}")
        return
    
    # Run prediction
    logger.info("Running prediction...")
    prediction = predict_disease(
        model=model,
        image=image,
        class_names=class_names,
        device=device,
        top_k=3
    )
    
    # Format and display results
    formatted = format_predictions(prediction)
    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Disease: {formatted['disease']}")
    logger.info(f"Confidence: {formatted['confidence']:.2f}%")
    logger.info("\nTop Alternatives:")
    for alt in formatted['top_alternatives']:
        logger.info(f"  - {alt['disease']}: {alt['confidence']:.2f}%")
    
    # Generate heatmap if requested
    if args.heatmap:
        logger.info("\nGenerating Grad-CAM heatmap...")
        try:
            heatmap, overlay = generate_gradcam_heatmap(
                model=model,
                image=image,
                device=device
            )
            
            if args.output:
                cv2.imwrite(args.output, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                logger.info(f"Heatmap saved to {args.output}")
            else:
                logger.info("Heatmap generated (use --output to save)")
        except Exception as e:
            logger.error(f"Failed to generate heatmap: {e}")


if __name__ == '__main__':
    main()


