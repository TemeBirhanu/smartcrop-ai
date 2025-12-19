"""
Batch inference utilities
"""

import os
from pathlib import Path
from typing import List, Dict
import json
from tqdm import tqdm

from .predict_image import predict_batch
from .heatmap_cam import generate_gradcam_heatmap
from .postprocess import format_predictions, calculate_severity


def run_batch_inference(
    model,
    image_paths: List[str],
    class_names: List[str],
    output_dir: str,
    generate_heatmaps: bool = True,
    device: str = "cuda"
) -> Dict:
    """
    Run batch inference on multiple images.
    
    Args:
        model: Trained model
        image_paths: List of image file paths
        class_names: List of class names
        output_dir: Directory to save results
        generate_heatmaps: Whether to generate Grad-CAM heatmaps
        device: Device to run on
    
    Returns:
        Dictionary with all results
    """
    import cv2
    import numpy as np
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if generate_heatmaps:
        os.makedirs(os.path.join(output_dir, 'heatmaps'), exist_ok=True)
    
    # Load images
    images = []
    valid_paths = []
    
    for img_path in tqdm(image_paths, desc="Loading images"):
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                valid_paths.append(img_path)
        else:
            print(f"Warning: Image not found: {img_path}")
    
    # Run predictions
    print("Running predictions...")
    predictions = predict_batch(
        model=model,
        images=images,
        class_names=class_names,
        device=device
    )
    
    # Generate heatmaps if requested
    if generate_heatmaps:
        print("Generating heatmaps...")
        for i, (img, pred) in enumerate(tqdm(zip(images, predictions), total=len(images))):
            try:
                heatmap, overlay = generate_gradcam_heatmap(
                    model=model,
                    image=img,
                    device=device
                )
                
                # Save heatmap overlay
                img_name = Path(valid_paths[i]).stem
                heatmap_path = os.path.join(
                    output_dir,
                    'heatmaps',
                    f'{img_name}_heatmap.jpg'
                )
                cv2.imwrite(heatmap_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Error generating heatmap for {valid_paths[i]}: {e}")
    
    # Format results
    results = []
    for img_path, pred in zip(valid_paths, predictions):
        formatted = format_predictions(pred)
        formatted['image_path'] = img_path
        results.append(formatted)
    
    # Save results
    results_path = os.path.join(output_dir, 'predictions.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    if generate_heatmaps:
        print(f"Heatmaps saved to {os.path.join(output_dir, 'heatmaps')}")
    
    return {
        'results': results,
        'output_dir': output_dir,
        'total_images': len(results)
    }

