"""
Convert Keras/TensorFlow model (.h5 or .keras) to ONNX format for Node.js/Flutter

Usage:
    python convert_keras_to_onnx.py --input final_leaf_classifier.h5 --output leaf_classifier.onnx
"""

import argparse
import tensorflow as tf
import tf2onnx
import onnx
from pathlib import Path


def convert_keras_to_onnx(
    keras_model_path: str,
    output_path: str,
    input_shape: tuple = (None, 128, 128, 3),  # Adjust based on your CONFIG['IMG_SIZE']
    opset_version: int = 13
):
    """
    Convert Keras/TensorFlow model to ONNX format.
    
    Args:
        keras_model_path: Path to .h5 or .keras model file
        output_path: Path to save ONNX model
        input_shape: Input shape (batch, height, width, channels)
        opset_version: ONNX opset version (default: 13)
    """
    print(f"Loading Keras model from {keras_model_path}...")
    
    # Load Keras model
    try:
        model = tf.keras.models.load_model(keras_model_path)
        print(f"✓ Model loaded successfully")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    # Convert to ONNX
    print(f"\nConverting to ONNX format...")
    try:
        # Specify input signature
        spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
        
        # Convert
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=opset_version,
            output_path=output_path
        )
        
        print(f"✓ ONNX model saved to: {output_path}")
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        print(f"✓ ONNX model verified")
        print(f"  ONNX version: {onnx_model.opset_import[0].version}")
        
        return output_path
        
    except Exception as e:
        print(f"✗ Error converting to ONNX: {e}")
        print(f"\nMake sure tf2onnx is installed:")
        print(f"  pip install tf2onnx onnx")
        return None


def convert_keras_to_tflite(
    keras_model_path: str,
    output_path: str
):
    """
    Convert Keras/TensorFlow model to TFLite format for Flutter mobile apps.
    
    Args:
        keras_model_path: Path to .h5 or .keras model file
        output_path: Path to save TFLite model
    """
    print(f"Loading Keras model from {keras_model_path}...")
    
    try:
        model = tf.keras.models.load_model(keras_model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    # Convert to TFLite
    print(f"\nConverting to TFLite format...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optional: Apply optimizations for smaller size
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"✓ TFLite model saved to: {output_path}")
        print(f"  Size: {file_size_mb:.2f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"✗ Error converting to TFLite: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Convert Keras model to ONNX/TFLite')
    parser.add_argument('--input', '-i', required=True, help='Path to input Keras model (.h5 or .keras)')
    parser.add_argument('--output', '-o', required=True, help='Output path (.onnx or .tflite)')
    parser.add_argument('--input-shape', nargs='+', type=int, default=[1, 128, 128, 3],
                        help='Input shape as [batch, height, width, channels] (default: 1 128 128 3)')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version (default: 13)')
    
    args = parser.parse_args()
    
    # Determine output format from file extension
    output_path = Path(args.output)
    
    if output_path.suffix == '.onnx':
        # Convert to ONNX
        input_shape = tuple(args.input_shape)
        convert_keras_to_onnx(
            keras_model_path=args.input,
            output_path=args.output,
            input_shape=input_shape,
            opset_version=args.opset
        )
    elif output_path.suffix == '.tflite':
        # Convert to TFLite
        convert_keras_to_tflite(
            keras_model_path=args.input,
            output_path=args.output
        )
    else:
        print(f"✗ Unknown output format: {output_path.suffix}")
        print(f"  Supported formats: .onnx (for Node.js) or .tflite (for Flutter)")
        return


if __name__ == '__main__':
    main()

