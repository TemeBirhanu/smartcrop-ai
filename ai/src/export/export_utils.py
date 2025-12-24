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
    model_config = load_yaml(str(model_config_path))
    
    # Load checkpoint if provided
    checkpoint = None
    if checkpoint_path and Path(checkpoint_path).exists():
        # Use weights_only=False for PyTorch 2.6+ compatibility (we trust our own checkpoints)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Determine number of classes from checkpoint if provided
    num_classes = None
    if checkpoint is not None:
        # Get state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Detect number of classes from checkpoint
        # Look for the classifier output layer (last linear layer)
        for key in reversed(list(state_dict.keys())):
            if 'classifier' in key and 'weight' in key:
                num_classes = state_dict[key].shape[0]
                print(f"Detected {num_classes} classes from checkpoint")
                break
        
        # If not found, try to get from checkpoint metadata
        if num_classes is None and 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
            print(f"Found {num_classes} classes in checkpoint metadata")
    
    # Use config default if checkpoint doesn't have it
    if num_classes is None:
        # Map model_name to config key
        config_key_map = {
            'mobilenet_v3': 'mobilenet',
            'efficientnet_b3': 'efficientnet'
        }
        config_key = config_key_map.get(model_name, model_name)
        model_config_dict = model_config.get(config_key, {})
        num_classes = model_config_dict.get('num_classes', 10)
        print(f"Using {num_classes} classes from config (default)")
    
    # Create model with correct number of classes
    # Map model_name to config key
    config_key_map = {
        'mobilenet_v3': 'mobilenet',
        'efficientnet_b3': 'efficientnet'
    }
    config_key = config_key_map.get(model_name, model_name)
    pretrained = model_config.get(config_key, {}).get('pretrained', False)
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
    
    # Load checkpoint weights
    if checkpoint is not None:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Model weights loaded successfully")
    
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

