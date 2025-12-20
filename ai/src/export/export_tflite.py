"""
TFLite model export utilities
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import os


def export_to_tflite(
    model: torch.nn.Module,
    input_shape: tuple,
    output_path: str,
    quantization_type: str = "dynamic",
    representative_dataset: Optional[List[np.ndarray]] = None,
    device: str = "cpu"
) -> str:
    """
    Export PyTorch model to TFLite format.
    
    Note: PyTorch -> TFLite requires intermediate conversion (usually via ONNX -> TensorFlow -> TFLite)
    This is a simplified interface that guides the conversion process.
    
    Args:
        model: PyTorch model to export
        input_shape: Input shape (batch_size, channels, height, width)
        output_path: Path to save TFLite model
        quantization_type: Quantization type ("none", "dynamic", "float16", "int8")
        representative_dataset: Representative dataset for int8 quantization
        device: Device to run export on
    
    Returns:
        Path to exported TFLite model
    """
    print("TFLite export requires intermediate conversion steps:")
    print("1. PyTorch -> ONNX (using export_to_onnx)")
    print("2. ONNX -> TensorFlow (using onnx-tf)")
    print("3. TensorFlow -> TFLite (using TensorFlow Lite Converter)")
    print("\nFor now, exporting to ONNX first...")
    
    # First export to ONNX
    onnx_path = output_path.replace('.tflite', '.onnx')
    
    from .export_onnx import export_to_onnx
    export_to_onnx(
        model=model,
        input_shape=input_shape,
        output_path=onnx_path,
        device=device
    )
    
    print(f"\nNext steps to convert to TFLite:")
    print(f"1. Install: pip install onnx-tf tensorflow")
    print(f"2. Convert ONNX to TensorFlow:")
    print(f"   from onnx_tf.backend import prepare")
    print(f"   import onnx")
    print(f"   onnx_model = onnx.load('{onnx_path}')")
    print(f"   tf_rep = prepare(onnx_model)")
    print(f"   tf_rep.export_graph('tf_model')")
    print(f"3. Convert TensorFlow to TFLite:")
    print(f"   import tensorflow as tf")
    print(f"   converter = tf.lite.TFLiteConverter.from_saved_model('tf_model')")
    if quantization_type == "dynamic":
        print(f"   converter.optimizations = [tf.lite.Optimize.DEFAULT]")
    elif quantization_type == "float16":
        print(f"   converter.optimizations = [tf.lite.Optimize.DEFAULT]")
        print(f"   converter.target_spec.supported_types = [tf.float16]")
    elif quantization_type == "int8":
        print(f"   converter.optimizations = [tf.lite.Optimize.DEFAULT]")
        print(f"   converter.representative_dataset = representative_dataset_gen")
        print(f"   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]")
    print(f"   tflite_model = converter.convert()")
    print(f"   with open('{output_path}', 'wb') as f:")
    print(f"       f.write(tflite_model)")
    
    return onnx_path  # Return ONNX path as intermediate step


def create_representative_dataset(
    data_loader,
    num_samples: int = 100
) -> List[np.ndarray]:
    """
    Create representative dataset for TFLite int8 quantization.
    
    Args:
        data_loader: PyTorch DataLoader
        num_samples: Number of samples to use
    
    Returns:
        List of numpy arrays (representative dataset)
    """
    representative_data = []
    count = 0
    
    for images, _ in data_loader:
        for img in images:
            if count >= num_samples:
                break
            # Convert to numpy and normalize to [0, 255]
            img_np = img.numpy()
            if img_np.min() < 0:  # If normalized, denormalize
                mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
                img_np = (img_np * std + mean) * 255.0
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            representative_data.append(img_np)
            count += 1
        if count >= num_samples:
            break
    
    return representative_data

