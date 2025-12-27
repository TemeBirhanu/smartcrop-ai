"""
Export YOLOv8 model to ONNX format for Node.js usage
"""

import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)


def export_yolo_to_onnx(
    weights_path: str = None,
    output_path: str = None,
    imgsz: int = 640,
    simplify: bool = True
):
    """
    Export YOLOv8 model to ONNX format.
    
    Args:
        weights_path: Path to YOLOv8 .pt weights file
        output_path: Output ONNX file path (optional)
        imgsz: Image size for export
        simplify: Whether to simplify the ONNX model
    """
    # Find trained YOLOv8 model
    if weights_path is None:
        runs_dir = Path('runs/detect')
        trained_yolo_path = None
        
        if runs_dir.exists():
            train_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('train')])
            if train_dirs:
                for train_dir in reversed(train_dirs):  # Most recent first
                    model_path = train_dir / 'weights' / 'best.pt'
                    if model_path.exists():
                        trained_yolo_path = model_path
                        break
        
        if trained_yolo_path is None or not trained_yolo_path.exists():
            print("❌ Trained YOLOv8 model not found!")
            print("   Expected location: runs/detect/train*/weights/best.pt")
            print("   Please train YOLOv8 first or provide --weights-path")
            return None
        
        weights_path = str(trained_yolo_path)
    
    weights_path = Path(weights_path)
    if not weights_path.exists():
        print(f"❌ Weights file not found: {weights_path}")
        return None
    
    print(f"📦 Loading YOLOv8 from: {weights_path}")
    model = YOLO(str(weights_path))
    
    # Determine output path
    if output_path is None:
        output_path = weights_path.parent / f"{weights_path.stem}.onnx"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📤 Exporting to ONNX: {output_path}")
    print(f"   Image size: {imgsz}x{imgsz}")
    print(f"   Simplify: {simplify}")
    
    # Export to ONNX
    try:
        model.export(
            format='onnx',
            imgsz=imgsz,
            simplify=simplify,
            opset=12  # ONNX opset version
        )
        
        # Ultralytics saves to same directory as weights with .onnx extension
        exported_path = weights_path.parent / f"{weights_path.stem}.onnx"
        
        if exported_path.exists():
            # Move to desired output path if different
            if exported_path != output_path:
                exported_path.rename(output_path)
            
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ YOLOv8 exported successfully!")
            print(f"   Output: {output_path}")
            print(f"   Size: {size_mb:.2f} MB")
            return str(output_path)
        else:
            print(f"❌ Export failed - output file not found")
            return None
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Export YOLOv8 to ONNX')
    parser.add_argument('--weights-path', type=str, default=None,
                       help='Path to YOLOv8 .pt weights file (auto-detected if not provided)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output ONNX file path')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for export (default: 640)')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Disable ONNX simplification')
    
    args = parser.parse_args()
    
    export_yolo_to_onnx(
        weights_path=args.weights_path,
        output_path=args.output,
        imgsz=args.imgsz,
        simplify=not args.no_simplify
    )

