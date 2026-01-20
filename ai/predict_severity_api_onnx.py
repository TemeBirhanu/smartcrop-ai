"""
API wrapper using ONNX classifiers (no model class imports)
Loads classifier from ONNX in `outputs/models/exported` and YOLO weights from
`runs/detect/.../weights/best.pt` (or a provided path). By default SAM is skipped (fast HSV fallback);
use `--use-sam` to enable SAM segmentation.
Returns JSON output.
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.yolo_detector import YOLODetector
from src.inference.postprocess import calculate_severity
from src.utils.helpers import get_device
from src.utils.file_utils import load_json


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_IMAGE_SIZE = 224


def find_default_onnx():
    exported = Path('outputs/models/exported')
    candidates = [exported / 'efficientnet_b3.onnx', exported / 'mobilenet_v3.onnx']
    for c in candidates:
        if c.exists():
            return str(c)
    # fallback: any .onnx in folder
    if exported.exists():
        for f in exported.iterdir():
            if f.suffix == '.onnx':
                return str(f)
    return None


def load_class_map():
    possible_class_maps = [
        Path('outputs/models/checkpoints/class_map.json'),
        Path('data/metadata/class_map.json'),
    ]

    for p in possible_class_maps:
        if p.exists():
            try:
                class_map = load_json(str(p))
                return [k for k in sorted(class_map.keys(), key=lambda x: class_map[x])]
            except Exception:
                continue
    return None


def preprocess_for_onnx(image: np.ndarray, image_size: int = DEFAULT_IMAGE_SIZE):
    # Expect input image as BGR (cv2.imread)
    if len(image.shape) == 3 and image.shape[2] == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image

    img_resized = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = img_resized.astype(np.float32) / 255.0
    # Normalize
    img = (img - MEAN) / STD
    # Determine ONNX layout later; return both HWC and CHW
    chw = img.transpose(2, 0, 1)
    nhwc = img[np.newaxis, :, :, :]
    nchw = chw[np.newaxis, :, :, :]
    return nchw, nhwc


def softmax(x: np.ndarray):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def onnx_predict(session: ort.InferenceSession, image: np.ndarray, class_names, top_k: int = 3):
    # Get input info
    input_meta = session.get_inputs()[0]
    input_shape = [s if isinstance(s, int) else -1 for s in input_meta.shape]

    nchw, nhwc = preprocess_for_onnx(image)

    # Decide layout: assume NCHW if shape[1]==3, otherwise NHWC
    use_nchw = False
    if len(input_shape) >= 4:
        if input_shape[1] == 3 or input_shape[1] == -1:
            use_nchw = True
        elif input_shape[-1] == 3:
            use_nchw = False
    # fallback to nchw
    if use_nchw:
        input_data = nchw.astype(np.float32)
    else:
        input_data = nhwc.astype(np.float32)

    input_name = input_meta.name
    outputs = session.run(None, {input_name: input_data})
    # assume logits in first output
    out = outputs[0]
    if out.ndim == 2:
        probs = softmax(out)
        probs = probs[0]
    elif out.ndim == 1:
        probs = softmax(out[np.newaxis, :])[0]
    else:
        # Unexpected output
        raise RuntimeError(f"Unexpected ONNX model output shape: {out.shape}")

    # Top-k
    k = min(top_k, len(class_names))
    top_indices = np.argsort(probs)[::-1][:k]
    top_k_results = []
    for idx in top_indices:
        top_k_results.append({
            'class': class_names[int(idx)],
            'confidence': float(probs[int(idx)])
        })

    return {
        'predicted_class': top_k_results[0]['class'],
        'confidence': top_k_results[0]['confidence'],
        'top_k': top_k_results
    }


def compute_leaf_mask_hsv(image_bgr: np.ndarray, kernel_size: int = 5):
    """Fast HSV-based leaf mask. Returns boolean mask (H x W) where True indicates green leaf pixels."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Broad green range; tune as needed for your dataset
    lower = np.array([25, 40, 40], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


def fill_mask(mask: np.ndarray, kernel_size: int = 31):
    """Fill holes and small gaps in a binary mask; returns boolean mask."""
    if mask.dtype != np.uint8:
        m = (mask > 0).astype(np.uint8) * 255
    else:
        m = mask.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, -1)
    return filled > 0


def compute_lesion_mask_in_box(image_bgr: np.ndarray, box: list):
    """Estimate lesion mask inside a bounding box by finding non-green pixels inside the filled leaf region in the box."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = max(0, x2); y2 = max(0, y2)
    roi = image_bgr[y1:y2+1, x1:x2+1]
    if roi.size == 0:
        return np.zeros((0, 0), dtype=bool)
    green_roi = compute_leaf_mask_hsv(roi, kernel_size=3)
    filled_roi = fill_mask(green_roi, kernel_size=15)
    lesion_roi = np.logical_and(filled_roi, np.logical_not(green_roi))
    return lesion_roi, filled_roi



def predict_severity_json(image_path=None, image=None, classifier_onnx=None, sam_checkpoint=None, yolo_weights=None, skip_sam: bool = True):
    """
    Predict severity pipeline. Accepts either `image_path` (str/Path) or an in-memory `image` (numpy ndarray BGR).
    When `image` is provided, the function will use it directly and will NOT return any file paths.
    """
    result = {"success": True}

    device = get_device(None)
    if image is None:
        if image_path is None:
            return {"success": False, "error": "No image or image_path provided"}
        image = cv2.imread(str(image_path))
        if image is None:
            return {"success": False, "error": f"Failed to load image: {image_path}"}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 1: Classification via ONNX
    onnx_path = Path(classifier_onnx) if classifier_onnx else (Path(find_default_onnx()) if find_default_onnx() else None)
    if onnx_path and onnx_path.exists():
        try:
            # Load class names
            class_names = load_class_map()
            if class_names is None:
                # try to read companion .data file which might include class_names or use default fallback
                companion = onnx_path.with_suffix(onnx_path.suffix + '.data')
                if companion.exists():
                    try:
                        meta = load_json(str(companion))
                        if 'class_names' in meta:
                            class_names = meta['class_names']
                    except Exception:
                        pass

            if class_names is None:
                # default placeholder
                class_names = [f"Class_{i}" for i in range(17)]

            sess_options = ort.SessionOptions()
            # Use CPU execution provider by default; if GPU providers available they'll be used by ort automatically if set
            session = ort.InferenceSession(str(onnx_path), sess_options, providers=['CPUExecutionProvider'])

            classification_result = onnx_predict(session, image, class_names, top_k=3)

            result["disease"] = {
                "predicted_class": classification_result['predicted_class'],
                "confidence": float(classification_result['confidence']),
                "top_predictions": [
                    {"class": p['class'], "confidence": float(p['confidence'])} for p in classification_result['top_k'][:3]
                ]
            }
        except Exception as e:
            result["classification_error"] = str(e)
    else:
        result["classification_error"] = f"Classifier ONNX model not found: {onnx_path}"

    # Step 2: YOLOv8 Lesion Detection (uses .pt weights)
    yolo_result = None
    try:
        trained_yolo_path = Path(yolo_weights) if yolo_weights else Path('runs/detect/train/weights/best.pt')
        if trained_yolo_path.exists():
            yolo_detector = YOLODetector(model_size="n", weights_path=str(trained_yolo_path))
        else:
            # fallback to default pretrained
            yolo_detector = YOLODetector(model_size="n", pretrained=True)

        yolo_result = yolo_detector.detect(image_rgb, conf_threshold=0.25)
        result["detections"] = {
            "count": int(yolo_result['count']),
            "boxes": [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in yolo_result['boxes']],
            "scores": [float(s) for s in yolo_result['scores']],
            "classes": [int(c) for c in yolo_result['classes']]
        }
    except Exception as e:
        result["detections_error"] = str(e)
        result["detections"] = {"count": 0, "boxes": [], "scores": [], "classes": []}

    # Step 3: Segmentation (SAM or fast HSV-based if skip_sam)
    sam_result = None
    if skip_sam:
        try:
            # Fast HSV-based leaf mask (no SAM) with refined lesion detection
            green_mask = compute_leaf_mask_hsv(image)
            leaf_mask_filled = fill_mask(green_mask, kernel_size=31)
            leaf_area = int(leaf_mask_filled.sum())
            affected_area = 0
            lesion_count = 0
            # For each YOLO box, find lesion pixels as non-green pixels inside the filled leaf area
            if yolo_result and yolo_result.get('count', 0) > 0:
                for box in yolo_result['boxes']:
                    lesion_roi, filled_roi = compute_lesion_mask_in_box(image, box)
                    # lesion_roi is relative to ROI; count only lesion pixels
                    affected_area += int(lesion_roi.sum())
                # use YOLO's box count as lesion count
                lesion_count = int(yolo_result['count'])
            else:
                # Fallback: no YOLO boxes -> compute lesions across full leaf mask
                lesion_mask_full = np.logical_and(leaf_mask_filled, np.logical_not(green_mask))
                affected_area = int(lesion_mask_full.sum())

                # estimate lesion count via contours (filter small noise by min area)
                lesion_mask_uint8 = (lesion_mask_full.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(lesion_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_area = 25  # minimum contour area in pixels to count as a lesion (tune as needed)
                lesion_count = sum(1 for c in contours if cv2.contourArea(c) >= min_area)

            affected_percentage = float(round((affected_area / leaf_area * 100) if leaf_area > 0 else 0.0, 2))
            sam_result = {
                'leaf_area': leaf_area,
                'affected_area': affected_area,
                'affected_percentage': affected_percentage
            }
            result["segmentation"] = {
                "leaf_area": float(sam_result['leaf_area']),
                "affected_area": float(sam_result['affected_area']),
                "affected_percentage": float(sam_result['affected_percentage'])
            }

            # If we used the fallback (no YOLO boxes), update yolo_result count so severity calc uses lesion_count
            if not yolo_result or int(yolo_result.get('count', 0)) == 0:
                yolo_result = yolo_result or {"count": 0, "boxes": [], "scores": [], "classes": []}
                yolo_result['count'] = int(lesion_count)
        except Exception as e:
            result["segmentation_error"] = str(e)
    else:
        sam_checkpoint_path = Path(sam_checkpoint) if sam_checkpoint else Path('outputs/models/sam/sam_vit_b_int8.pth')
        if sam_checkpoint_path.exists():
            try:
                # Import SAMSegmenter lazily to avoid heavy imports when SAM is not used
                from src.models.segment_sam import SAMSegmenter

                sam_segmenter = SAMSegmenter(
                    model_type="vit_b",
                    checkpoint_path=str(sam_checkpoint_path),
                    device=device
                )
                
                prompt_boxes = None
                if yolo_result and yolo_result['count'] > 0:
                    prompt_boxes = yolo_result['boxes']
                
                sam_result = sam_segmenter.segment(
                    image=image,
                    prompt_boxes=prompt_boxes,
                    segment_leaf=True
                )
                
                result["segmentation"] = {
                    "leaf_area": float(sam_result['leaf_area']),
                    "affected_area": float(sam_result['affected_area']),
                    "affected_percentage": float(sam_result['affected_percentage'])
                }
            except Exception as e:
                result["segmentation_error"] = str(e)
        else:
            result["segmentation_error"] = f"SAM checkpoint not found: {sam_checkpoint_path}"

    # Step 4: Calculate Severity
    if sam_result and sam_result['leaf_area'] > 0 and yolo_result:
        try:
            severity = calculate_severity(
                lesion_count=yolo_result['count'],
                leaf_area=sam_result['leaf_area'],
                affected_area=sam_result['affected_area']
            )

            result["severity"] = {
                "level": severity['severity_level'],
                "affected_percentage": float(severity['affected_percentage']),
                "lesion_count": int(severity['lesion_count']),
                "lesion_density": float(severity['lesion_density'])
            }
        except Exception as e:
            result["severity_error"] = str(e)

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Severity prediction API (ONNX classifier)')
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--classifier-onnx', default=None, help='Path to ONNX classifier file')
    parser.add_argument('--sam-checkpoint', default='outputs/models/sam/sam_vit_b.pth')
    parser.add_argument('--use-sam', action='store_true', help='Use SAM for segmentation (slower, more accurate)')
    parser.add_argument('--yolo-weights', default='runs/detect/train/weights/best.pt', help='Path to YOLO .pt weights (falls back to pretrained)')

    args = parser.parse_args()

    result = predict_severity_json(
        image_path=args.image,
        classifier_onnx=args.classifier_onnx,
        sam_checkpoint=args.sam_checkpoint,
        yolo_weights=args.yolo_weights,
        skip_sam=not args.use_sam
    )

    print(json.dumps(result, indent=2))
