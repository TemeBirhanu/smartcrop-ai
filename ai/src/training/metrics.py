"""
Training metrics and evaluation functions
"""

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Tuple
import numpy as np


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        predictions: Predicted class indices
        targets: True class indices
        num_classes: Number of classes
    
    Returns:
        Dictionary with metrics:
        {
            'accuracy': float,
            'precision': float (macro-averaged),
            'recall': float (macro-averaged),
            'f1': float (macro-averaged),
            'per_class_precision': np.ndarray,
            'per_class_recall': np.ndarray,
            'per_class_f1': np.ndarray
        }
    """
    accuracy = accuracy_score(targets, predictions)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average='macro',
        zero_division=0
    )
    
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average=None,
        zero_division=0
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1
    }


def get_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class indices
        targets: True class indices
        num_classes: Number of classes
    
    Returns:
        Confusion matrix (num_classes x num_classes)
    """
    return confusion_matrix(targets, predictions, labels=range(num_classes))

