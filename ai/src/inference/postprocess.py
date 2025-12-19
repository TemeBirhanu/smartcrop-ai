"""
Post-processing functions for predictions
"""

from typing import Dict, List, Tuple
import numpy as np


def calculate_severity(
    lesion_count: int,
    leaf_area: float,
    affected_area: float
) -> Dict[str, float]:
    """
    Calculate disease severity metrics.
    
    Args:
        lesion_count: Number of lesions detected (from YOLO)
        leaf_area: Total leaf area in pixels (from SAM)
        affected_area: Affected area in pixels (from SAM)
    
    Returns:
        Dictionary with severity metrics:
        {
            'lesion_count': int,
            'affected_percentage': float,
            'severity_level': str,
            'lesion_density': float
        }
    """
    # Calculate affected percentage
    if leaf_area > 0:
        affected_percentage = (affected_area / leaf_area) * 100
    else:
        affected_percentage = 0.0
    
    # Calculate lesion density (lesions per unit area)
    if leaf_area > 0:
        lesion_density = lesion_count / (leaf_area / 10000)  # per cm²
    else:
        lesion_density = 0.0
    
    # Determine severity level
    if affected_percentage < 5:
        severity_level = "Low"
    elif affected_percentage < 25:
        severity_level = "Moderate"
    elif affected_percentage < 50:
        severity_level = "High"
    else:
        severity_level = "Critical"
    
    return {
        'lesion_count': lesion_count,
        'affected_percentage': round(affected_percentage, 2),
        'severity_level': severity_level,
        'lesion_density': round(lesion_density, 2)
    }


def format_predictions(
    prediction: Dict,
    severity: Dict = None,
    language: str = "en"
) -> Dict:
    """
    Format predictions for display/API response.
    
    Args:
        prediction: Prediction dictionary from predict_disease
        severity: Severity metrics (optional)
        language: Output language ("en" or "am" for Amharic)
    
    Returns:
        Formatted prediction dictionary
    """
    result = {
        'disease': prediction['predicted_class'],
        'confidence': round(prediction['confidence'] * 100, 2),
        'top_alternatives': [
            {
                'disease': alt['class'],
                'confidence': round(alt['confidence'] * 100, 2)
            }
            for alt in prediction['top_k'][1:]  # Skip first (already in 'disease')
        ]
    }
    
    if severity:
        result['severity'] = severity
    
    # Add language-specific formatting if needed
    if language == "am":
        # Amharic translations would go here
        pass
    
    return result


def combine_detections(
    classification_result: Dict,
    yolo_result: Dict,
    sam_result: Dict = None
) -> Dict:
    """
    Combine results from classification, YOLO detection, and SAM segmentation.
    
    Args:
        classification_result: Disease classification result
        yolo_result: YOLO lesion detection result
        sam_result: SAM segmentation result (optional)
    
    Returns:
        Combined result dictionary
    """
    combined = {
        'disease': classification_result,
        'lesions': {
            'count': yolo_result.get('count', 0),
            'boxes': yolo_result.get('boxes', []),
            'scores': yolo_result.get('scores', [])
        }
    }
    
    if sam_result:
        combined['segmentation'] = sam_result
        
        # Calculate severity if we have all components
        if 'leaf_area' in sam_result and 'affected_area' in sam_result:
            severity = calculate_severity(
                lesion_count=yolo_result.get('count', 0),
                leaf_area=sam_result['leaf_area'],
                affected_area=sam_result['affected_area']
            )
            combined['severity'] = severity
    
    return combined

