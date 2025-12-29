"""
SAM (Segment Anything Model) for precise segmentation of diseased leaf areas
Uses Meta's pretrained SAM model - no training needed
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from pathlib import Path

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️  segment-anything not installed. Install with: pip install segment-anything")


class SAMSegmenter:
    """
    SAM (Segment Anything Model) wrapper for segmenting diseased leaf areas.
    Uses pretrained SAM from Meta - no training needed.
    """
    
    def __init__(
        self,
        model_type: str = "vit_b",  # "vit_h", "vit_l", or "vit_b"
        checkpoint_path: Optional[str] = None,
        device: str = None
    ):
        """
        Initialize SAM segmenter.
        
        Args:
            model_type: SAM model type ("vit_h" largest, "vit_l" large, "vit_b" base)
            checkpoint_path: Path to SAM checkpoint (auto-downloads if None)
            device: Device to run on ("cuda" or "cpu")
        """
        if not SAM_AVAILABLE:
            raise ImportError(
                "segment-anything not installed. "
                "Install with: pip install segment-anything"
            )
        
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set default checkpoint paths
        if checkpoint_path is None:
            checkpoint_path = self._get_default_checkpoint_path(model_type)
        
        self.checkpoint_path = checkpoint_path
        
        # Load SAM model
        self._load_model()
    
    def _get_default_checkpoint_path(self, model_type: str) -> str:
        """Get default checkpoint path for SAM model."""
        # Check common locations
        possible_paths = [
            f"outputs/models/sam/sam_vit_{model_type[-1]}.pth",
            f"models/sam/sam_vit_{model_type[-1]}.pth",
            f"sam_vit_{model_type[-1]}.pth"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        # Return path where checkpoint should be downloaded
        return f"outputs/models/sam/sam_vit_{model_type[-1]}.pth"
    
    def _load_model(self):
        """Load SAM model from checkpoint."""
        checkpoint_path = Path(self.checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at: {self.checkpoint_path}\n"
                f"Please download SAM checkpoint from: https://github.com/facebookresearch/segment-anything#model-checkpoints\n"
                f"Recommended: sam_vit_b_01ec64.pth (375MB) - fastest\n"
                f"Save to: {self.checkpoint_path}"
            )
        
        print(f"📥 Loading SAM model from {self.checkpoint_path}...")
        sam = sam_model_registry[self.model_type](checkpoint=str(checkpoint_path))
        sam.to(device=self.device)
        
        self.predictor = SamPredictor(sam)
        print(f"✅ SAM model loaded on {self.device}")
    
    def segment(
        self,
        image: np.ndarray,
        prompt_boxes: Optional[List[np.ndarray]] = None,
        prompt_points: Optional[List[Tuple[int, int]]] = None,
        segment_leaf: bool = True
    ) -> Dict:
        """
        Segment diseased areas in image.
        
        Args:
            image: Input image (numpy array, RGB or BGR)
            prompt_boxes: List of bounding boxes from YOLO [[x1, y1, x2, y2], ...]
            prompt_points: List of point prompts [(x, y), ...] (optional)
            segment_leaf: If True, also segments entire leaf to get total area
        
        Returns:
            Dictionary with segmentation results:
            {
                'masks': List of segmentation masks,
                'affected_area': float,  # pixels
                'leaf_area': float,  # pixels (if segment_leaf=True)
                'affected_percentage': float,  # (affected_area / leaf_area) * 100
                'num_masks': int
            }
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if it's BGR (OpenCV format)
            if image.dtype == np.uint8:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            image_rgb = image
        
        # Set image in predictor
        self.predictor.set_image(image_rgb)
        
        masks = []
        affected_area = 0.0
        
        # Segment using bounding boxes (from YOLO)
        if prompt_boxes and len(prompt_boxes) > 0:
            for box in prompt_boxes:
                # Convert to format expected by SAM: [x1, y1, x2, y2]
                box_array = np.array(box, dtype=np.float32)
                
                # Run SAM prediction
                mask, scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_array[None, :],
                    multimask_output=False
                )
                
                masks.append(mask[0])
                affected_area += np.sum(mask[0])
        
        # Segment using point prompts (optional)
        if prompt_points and len(prompt_points) > 0:
            points = np.array(prompt_points, dtype=np.float32)
            labels = np.ones(len(prompt_points), dtype=np.int32)  # 1 = foreground
            
            mask, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=None,
                multimask_output=False
            )
            
            masks.append(mask[0])
            affected_area += np.sum(mask[0])
        
        # Combine all masks (union)
        if len(masks) > 0:
            combined_mask = np.zeros_like(masks[0], dtype=bool)
            for mask in masks:
                combined_mask = combined_mask | mask
        else:
            # No prompts provided - return empty result
            h, w = image_rgb.shape[:2]
            combined_mask = np.zeros((h, w), dtype=bool)
        
        affected_area = float(np.sum(combined_mask))
        
        # Segment entire leaf to get total area
        leaf_area = 0.0
        if segment_leaf:
            leaf_area = self._segment_leaf(image_rgb)
        
        # Calculate percentage
        if leaf_area > 0:
            affected_percentage = (affected_area / leaf_area) * 100
        else:
            affected_percentage = 0.0
        
        return {
            'masks': masks,
            'combined_mask': combined_mask,
            'affected_area': affected_area,
            'leaf_area': leaf_area,
            'affected_percentage': round(affected_percentage, 2),
            'num_masks': len(masks)
        }
    
    def _segment_leaf(self, image: np.ndarray) -> float:
        """
        Segment entire leaf to get total leaf area.
        Uses a simple approach: segment the largest green region.
        
        Args:
            image: Input image (RGB)
        
        Returns:
            Leaf area in pixels
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define green color range (for leaves)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green regions
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour (assumed to be the leaf)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            leaf_mask = np.zeros_like(mask)
            cv2.fillPoly(leaf_mask, [largest_contour], 255)
            leaf_area = float(np.sum(leaf_mask > 0))
        else:
            # Fallback: use entire image area
            leaf_area = float(image.shape[0] * image.shape[1])
        
        return leaf_area
    
    def visualize_segmentation(
        self,
        image: np.ndarray,
        segmentation_result: Dict,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize segmentation results on image.
        
        Args:
            image: Original image
            segmentation_result: Result from segment() method
            save_path: Optional path to save visualization
        
        Returns:
            Visualization image
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        else:
            vis_image = image.copy()
        
        # Overlay combined mask
        mask = segmentation_result['combined_mask']
        vis_image[mask] = vis_image[mask] * 0.6 + np.array([255, 0, 0]) * 0.4  # Red overlay
        
        # Draw bounding boxes if available
        # (Would need to be passed separately or stored in result)
        
        # Add text
        text = f"Affected: {segmentation_result['affected_percentage']:.1f}%"
        cv2.putText(vis_image, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        return vis_image

