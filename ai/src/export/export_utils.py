"""
Unified model export utilities
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torch

from .export_onnx import export_to_onnx
from .export_tflite import export_to_tflite
from ..models.classifier_mobilenet import MobileNetV3Classifier
from ..models.classifier_effnet import EfficientNetClassifier
from ..utils.file_utils import load_yaml, ensure_dir
from ..utils.helpers import get_device


def load_model_from_config(config_path: str, model_name: str) -> torch.nn.Module:
    """
    Load model based on configuration.
    
    Args:
        config_path: Path to model config YAML
        model_name: Name of model to load
    
    Returns:
        Loaded PyTorch model
    """
    config = load_yaml(config_path)
    model_config = config.get(model_name, {})
    
    num_classes = model_config.get('num_classes', 10)
    pretrained = model_config.get('pretrained', True)
    
    if model_name == 'mobilenet_v3':
        model = MobileNetV3Classifier(
            num_classes=num_classes,
            pretrained=pretrained
        )
    elif model_name == 'efficientnet_b3':
        model = EfficientNetClassifier(
            num_classes=num_classes,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


def export_model(
    config_path: str,
    model_name: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu"
) -> Dict[str, str]:
    """
    Export model based on export configuration.
    
    Args:
        config_path: Path to export config YAML
        model_name: Name of model to export
        checkpoint_path: Path to model checkpoint (optional)
        device: Device to run export on
    
    Returns:
        Dictionary with export paths
    """
    # Load export config
    export_config = load_yaml(config_path)
    
    # Load model config
    model_config_path = Path(config_path).parent / "model.yaml"
    model = load_model_from_config(str(model_config_path), model_name)
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    model.eval()
    model = model.to(device)
    
    # Get export settings
    input_shape = tuple(export_config.get('input_shape', [1, 3, 224, 224]))
    output_dir = export_config.get('output_dir', 'outputs/models/exported')
    ensure_dir(output_dir)
    
    export_paths = {}
    
    # Export to ONNX
    if export_config.get('onnx', {}).get('enabled', False):
        onnx_config = export_config['onnx']
        onnx_path = Path(output_dir) / onnx_config.get('output_path', f'{model_name}.onnx')
        
        export_to_onnx(
            model=model,
            input_shape=input_shape,
            output_path=str(onnx_path),
            opset_version=onnx_config.get('opset_version', 11),
            dynamic_axes=onnx_config.get('dynamic_axes'),
            device=device
        )
        export_paths['onnx'] = str(onnx_path)
    
    # Export to TFLite
    if export_config.get('tflite', {}).get('enabled', False):
        tflite_config = export_config['tflite']
        tflite_path = Path(output_dir) / tflite_config.get('output_path', f'{model_name}.tflite')
        
        export_to_tflite(
            model=model,
            input_shape=input_shape,
            output_path=str(tflite_path),
            quantization_type=tflite_config.get('quantization_type', 'dynamic'),
            device=device
        )
        export_paths['tflite'] = str(tflite_path)
    
    return export_paths

