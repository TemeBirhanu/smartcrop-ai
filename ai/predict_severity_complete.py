"""
Complete Severity Pipeline (Classification + Trained YOLOv8 + SAM)
Combines all models to calculate disease severity
"""

import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import all needed modules
from src.models.classifier_mobilenet import MobileNetV3Classifier
from src.models.classifier_effnet import EfficientNetClassifier
from src.models.yolo_detector import YOLODetector
from src.models.segment_sam import SAMSegmenter
from src.inference.predict_image import predict_disease
from src.inference.postprocess import calculate_severity, format_predictions
from src.utils.helpers import get_device
from src.utils.file_utils import load_json


def main():
    parser = argparse.ArgumentParser(description='Complete severity prediction pipeline')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--classifier-model', type=str, 
                       default='outputs/models/checkpoints/best_model.pth',
                       help='Path to classification model checkpoint')
    parser.add_argument('--model-type', type=str,
                       choices=['mobilenet_v3', 'efficientnet_b3'],
                       default=None,
                       help='Model type (mobilenet_v3 or efficientnet_b3). Auto-detected if not provided.')
    parser.add_argument('--sam-checkpoint', type=str,
                       default='outputs/models/sam/sam_vit_b.pth',
                       help='Path to SAM checkpoint')
    parser.add_argument('--output', type=str,
                       default='outputs/severity_result.jpg',
                       help='Output image path')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    print(f"📸 Processing image: {args.image}\n")
    
    # Setup
    device = get_device(args.device)
    print(f"🔧 Using device: {device}\n")
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ Failed to load image: {args.image}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Step 1: Disease Classification
    print("=" * 60)
    print("Step 1: Disease Classification")
    print("=" * 60)
    
    classification_result = None
    
    # Load classification model
    classifier_path = Path(args.classifier_model)
    if not classifier_path.exists():
        print(f"❌ Classification model not found at {classifier_path}")
        print("   Please train a classification model first (Step 6)")
    else:
        # Detect model type and num_classes from checkpoint
        checkpoint = torch.load(classifier_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Detect num_classes
        num_classes = None
        for key in reversed(list(state_dict.keys())):
            if 'classifier' in key and 'weight' in key:
                num_classes = state_dict[key].shape[0]
                break
        
        if num_classes is None:
            num_classes = 17  # Default fallback
        
        # Detect model type (check architecture)
        # Use user-provided model type if available, otherwise auto-detect
        if args.model_type:
            model_type = args.model_type
            print(f"   Model type: {model_type} (from --model-type argument)")
        else:
            # Auto-detect: default to efficientnet_b3, check for mobilenet
            model_type = 'efficientnet_b3'  # Default
            if any('mobilenet' in k.lower() for k in state_dict.keys()):
                model_type = 'mobilenet_v3'
            print(f"   Model type: {model_type} (auto-detected)")
        
        print(f"   Number of classes: {num_classes}")
        
        # Load class names - try multiple locations (same as predict.py)
        class_names = None
        
        # Try to load from checkpoint directory first
        possible_class_maps = [
            Path('outputs/models/checkpoints/class_map.json'),
            Path('data/metadata/class_map.json'),
            classifier_path.parent / 'class_map.json',
        ]
        
        for class_map_path in possible_class_maps:
            if class_map_path.exists():
                try:
                    class_map = load_json(str(class_map_path))
                    class_names = [k for k in sorted(class_map.keys(), key=lambda x: class_map[x])]
                    print(f"   ✓ Loaded {len(class_names)} class names from {class_map_path}")
                    print(f"   Sample classes: {', '.join(class_names[:3])}..." if len(class_names) > 3 else f"   Classes: {', '.join(class_names)}")
                    break
                except Exception as e:
                    print(f"   ⚠️  Failed to load class map from {class_map_path}: {e}")
                    continue
        
        # If still no class names, check if checkpoint contains them
        if class_names is None and 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
            print(f"   ✓ Loaded {len(class_names)} class names from checkpoint")
        
        # Last resort: use generic class names
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]
            print(f"   ⚠️  Using default class names (Class_0, Class_1, ...)")
            print(f"   To use real names, ensure class_map.json exists in outputs/models/checkpoints/")
        
        # Load model
        if model_type == 'mobilenet_v3':
            classifier_model = MobileNetV3Classifier(num_classes=num_classes, pretrained=False)
        else:
            classifier_model = EfficientNetClassifier(num_classes=num_classes, pretrained=False)
        
        classifier_model.load_state_dict(state_dict)
        classifier_model.eval()
        classifier_model = classifier_model.to(device)
        
        # Run classification
        classification_result = predict_disease(
            model=classifier_model,
            image=image,
            class_names=class_names,
            device=device,
            top_k=3
        )
        
        print(f"\n   ✅ Disease: {classification_result['predicted_class']}")
        print(f"   ✅ Confidence: {classification_result['confidence']*100:.2f}%")
    
    if classification_result is None:
        print("⚠️  Cannot proceed without classification result")
        return
    
    # Step 2: YOLOv8 Lesion Detection (Trained Model)
    print("\n" + "=" * 60)
    print("Step 2: YOLOv8 Lesion Detection (Trained Model)")
    print("=" * 60)
    
    # Find trained YOLOv8 model
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
    
    if trained_yolo_path and trained_yolo_path.exists():
        print(f"   Loading trained YOLOv8 from: {trained_yolo_path}")
        yolo_detector = YOLODetector(model_size="n", weights_path=str(trained_yolo_path))
    else:
        print("   ⚠️  Trained YOLOv8 not found. Using pretrained model...")
        yolo_detector = YOLODetector(model_size="n", pretrained=True)
    
    # Run YOLO detection
    yolo_result = yolo_detector.detect(image_rgb, conf_threshold=0.25)
    print(f"\n   ✅ Detected {yolo_result['count']} lesions")
    
    # Step 3: SAM Segmentation
    print("\n" + "=" * 60)
    print("Step 3: SAM Segmentation")
    print("=" * 60)
    
    sam_checkpoint = Path(args.sam_checkpoint)
    if not sam_checkpoint.exists():
        print(f"   ❌ SAM checkpoint not found at {sam_checkpoint}")
        print("   Please download it first (Step 9, Cell 1)")
        sam_result = None
    else:
        print(f"   Loading SAM from: {sam_checkpoint}")
        sam_segmenter = SAMSegmenter(
            model_type="vit_b",
            checkpoint_path=str(sam_checkpoint),
            device=device
        )
        
        # Use YOLO boxes as prompts
        prompt_boxes = None
        if yolo_result['count'] > 0:
            prompt_boxes = yolo_result['boxes']
            print(f"   Using {len(prompt_boxes)} YOLO boxes as prompts")
        
        # Run SAM segmentation
        sam_result = sam_segmenter.segment(
            image=image,
            prompt_boxes=prompt_boxes,
            segment_leaf=True
        )
        
        print(f"\n   ✅ Affected area: {sam_result['affected_area']:.0f} pixels")
        print(f"   ✅ Leaf area: {sam_result['leaf_area']:.0f} pixels")
        print(f"   ✅ Affected percentage: {sam_result['affected_percentage']:.2f}%")
    
    # Step 4: Calculate Severity
    print("\n" + "=" * 60)
    print("Step 4: Severity Calculation")
    print("=" * 60)
    
    if sam_result and sam_result['leaf_area'] > 0:
        severity = calculate_severity(
            lesion_count=yolo_result['count'],
            leaf_area=sam_result['leaf_area'],
            affected_area=sam_result['affected_area']
        )
        
        print(f"\n   ✅ Severity Level: {severity['severity_level']}")
        print(f"   ✅ Affected Percentage: {severity['affected_percentage']:.2f}%")
        print(f"   ✅ Lesion Count: {severity['lesion_count']}")
        print(f"   ✅ Lesion Density: {severity['lesion_density']:.2f} per cm²")
    else:
        print("   ⚠️  Cannot calculate severity: SAM segmentation not available")
        severity = None
    
    # Step 5: Visualize Results
    print("\n" + "=" * 60)
    print("Step 5: Visualization")
    print("=" * 60)
    
    # Create visualization
    vis_image = image_rgb.copy()
    
    # Draw YOLO bounding boxes
    if yolo_result['count'] > 0:
        for box, score in zip(yolo_result['boxes'], yolo_result['scores']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Lesion {score:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Overlay SAM mask
    if sam_result and 'combined_mask' in sam_result:
        mask = sam_result['combined_mask']
        vis_image[mask] = vis_image[mask] * 0.6 + np.array([255, 0, 0]) * 0.4  # Red overlay
    
    # Add text overlay
    text_y = 30
    cv2.putText(vis_image, f"Disease: {classification_result['predicted_class']}", 
               (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    text_y += 30
    cv2.putText(vis_image, f"Confidence: {classification_result['confidence']*100:.1f}%", 
               (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if severity:
        text_y += 30
        cv2.putText(vis_image, f"Severity: {severity['severity_level']}", 
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
        cv2.putText(vis_image, f"Affected: {severity['affected_percentage']:.1f}%", 
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
        cv2.putText(vis_image, f"Lesions: {severity['lesion_count']}", 
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save and display
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Disease: {classification_result['predicted_class']}")
    print(f"Confidence: {classification_result['confidence']*100:.2f}%")
    if severity:
        print(f"\nSeverity:")
        print(f"  Level: {severity['severity_level']}")
        print(f"  Affected Area: {severity['affected_percentage']:.2f}%")
        print(f"  Lesion Count: {severity['lesion_count']}")
        print(f"  Lesion Density: {severity['lesion_density']:.2f} per cm²")
    print(f"\n✅ Complete severity analysis saved to: {output_path}")


if __name__ == '__main__':
    main()

