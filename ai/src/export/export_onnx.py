"""
ONNX model export utilities
"""

import torch
import torch.onnx
import onnx
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np


def export_to_onnx(
    model: torch.nn.Module,
    input_shape: tuple,
    output_path: str,
    opset_version: int = 11,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    device: str = "cpu"
) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        input_shape: Input shape (batch_size, channels, height, width)
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
        dynamic_axes: Dictionary for dynamic axes (e.g., {'input': {0: 'batch_size'}})
        device: Device to run export on
    
    Returns:
        Path to exported ONNX model
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Default dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    print(f"Exporting model to ONNX format...")
    print(f"Input shape: {input_shape}")
    print(f"Output path: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )
    
    # Verify exported model
    try:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model exported successfully: {output_path}")
        print(f"  Model size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Warning: ONNX model verification failed: {e}")
    
    return str(output_path)


def optimize_onnx_model(onnx_path: str, optimized_path: Optional[str] = None) -> str:
    """
    Optimize ONNX model using onnxoptimizer.
    
    Args:
        onnx_path: Path to ONNX model
        optimized_path: Path to save optimized model (optional)
    
    Returns:
        Path to optimized model
    """
    try:
        import onnxoptimizer
        
        if optimized_path is None:
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        
        model = onnx.load(onnx_path)
        optimized_model = onnxoptimizer.optimize(model)
        onnx.save(optimized_model, optimized_path)
        
        print(f"✓ ONNX model optimized: {optimized_path}")
        return optimized_path
    except ImportError:
        print("Warning: onnxoptimizer not installed. Skipping optimization.")
        return onnx_path

