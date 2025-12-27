"""
Model definitions and loaders
"""

from .classifier_effnet import EfficientNetClassifier
from .classifier_mobilenet import MobileNetV3Classifier
from .yolo_detector import YOLODetector
from .growth_stage_model import GrowthStageClassifier
from .model_utils import load_pretrained_weights, freeze_backbone

# Try to import SAM (optional dependency)
try:
    from .segment_sam import SAMSegmenter
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    SAMSegmenter = None

__all__ = [
    "EfficientNetClassifier",
    "MobileNetV3Classifier",
    "YOLODetector",
    "GrowthStageClassifier",
    "load_pretrained_weights",
    "freeze_backbone",
]

if SAM_AVAILABLE:
    __all__.append("SAMSegmenter")

