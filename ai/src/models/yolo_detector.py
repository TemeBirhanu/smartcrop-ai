"""
YOLOv8 detector for disease lesion detection
Uses PRETRAINED YOLOv8 weights - fine-tunes on custom lesion annotations
"""

from ultralytics import YOLO
from typing import List, Dict, Tuple
import torch
import numpy as np


class YOLODetector:
    """
    YOLOv8 wrapper for detecting disease lesions on crop leaves.
    Uses pretrained YOLOv8 weights - only fine-tunes on your lesion data.
    """
    
    def __init__(
        self,
        model_size: str = "n",
        pretrained: bool = True,  # Use pretrained YOLOv8 weights
        weights_path: str = None
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_size: "n" (nano) or "s" (small)
            pretrained: Use pretrained YOLOv8 weights (default: True)
            weights_path: Path to custom weights (optional, for fine-tuned model)
        """
        self.model_size = model_size
        
        if weights_path:
            # Load your fine-tuned model
            self.model = YOLO(weights_path)
        elif pretrained:
            # Load pretrained YOLOv8 (automatically downloads if needed)
            # This is already trained on COCO dataset - powerful feature extraction
            model_name = f"yolov8{model_size}.pt"
            self.model = YOLO(model_name)
        else:
            # Initialize from scratch (not recommended - requires huge dataset)
            self.model = YOLO(f"yolov8{model_size}.yaml")
    
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Dict:
        """
        Detect disease lesions in image.
        
        Args:
            image: Input image (numpy array)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        
        Returns:
            Dictionary with detection results:
            {
                'boxes': List of bounding boxes,
                'scores': List of confidence scores,
                'classes': List of class indices,
                'count': Number of lesions detected
            }
        """
        # Run inference
        results = self.model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Extract results
        result = results[0]
        
        boxes = []
        scores = []
        classes = []
        
        if result.boxes is not None:
            for box in result.boxes:
                # Get box coordinates (x1, y1, x2, y2)
                xyxy = box.xyxy[0].cpu().numpy()
                boxes.append(xyxy)
                
                # Get confidence
                conf = box.conf[0].cpu().numpy()
                scores.append(float(conf))
                
                # Get class
                cls = box.cls[0].cpu().numpy()
                classes.append(int(cls))
        
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'count': len(boxes)
        }
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 50,  # Few epochs needed with pretrained weights
        imgsz: int = 640,
        batch: int = 16
    ):
        """
        Fine-tune YOLO model on custom dataset.
        Only trains on your lesion annotations - backbone stays pretrained.
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs (50-100 is usually enough)
            imgsz: Image size
            batch: Batch size
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch
        )

