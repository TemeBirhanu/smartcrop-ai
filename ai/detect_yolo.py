"""
Simple script to use pretrained YOLOv8 for object detection (no training needed)
Usage: python detect_yolo.py --image path/to/image.jpg
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.yolo_detector import YOLODetector


def draw_detections(image, results):
    """Draw bounding boxes on image."""
    img_with_boxes = image.copy()
    
    for i, (box, score, cls) in enumerate(zip(results['boxes'], results['scores'], results['classes'])):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"Object {cls} ({score:.2f})"
        cv2.putText(img_with_boxes, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_with_boxes


def main():
    parser = argparse.ArgumentParser(description='Use pretrained YOLOv8 for detection')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8 model size: n (nano), s (small), m (medium), l (large), x (xlarge)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='outputs/yolo_detection.jpg',
                       help='Output image path')
    
    args = parser.parse_args()
    
    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image not found: {args.image}")
        return
    
    print(f"📸 Loading image: {args.image}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Error: Could not load image: {args.image}")
        return
    
    # Convert BGR to RGB (YOLO expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize YOLOv8 detector (pretrained - no training needed!)
    print(f"🤖 Loading pretrained YOLOv8-{args.model_size}...")
    print("   (This will auto-download on first use)")
    detector = YOLODetector(model_size=args.model_size, pretrained=True)
    
    # Run detection
    print("🔍 Running detection...")
    results = detector.detect(image_rgb, conf_threshold=args.conf)
    
    # Print results
    print(f"\n✅ Detection Results:")
    print(f"   Found {results['count']} objects")
    
    if results['count'] > 0:
        print(f"\n   Detections:")
        for i, (box, score, cls) in enumerate(zip(results['boxes'], results['scores'], results['classes'])):
            x1, y1, x2, y2 = map(int, box)
            print(f"   {i+1}. Object class {cls}: confidence {score:.2f}, box [{x1}, {y1}, {x2}, {y2}]")
    else:
        print("   No objects detected (try lowering --conf threshold)")
    
    # Draw detections
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    img_with_boxes = draw_detections(image, results)
    cv2.imwrite(str(output_path), img_with_boxes)
    print(f"\n💾 Saved result to: {args.output}")
    
    


if __name__ == '__main__':
    main()

