import sys
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

# Ensure the ai/ folder is importable (add project root / ai to sys.path)
ROOT = Path(__file__).resolve().parents[1]
AI_DIR = ROOT / "ai"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

# Import prediction helper functions from the ai folder
from predict_severity_api_onnx import (
    predict_severity_json,
    onnx_predict,
    find_default_onnx,
    load_class_map,
    load_json,
)
import cv2

app = FastAPI(title="SmartCrop Severity API (ONNX, Backend)")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predictwithseverity")
async def predict_with_severity(
    file: UploadFile = File(...),
    use_sam: bool = Query(False, description="If true, use SAM for segmentation (slower)."),
    classifier_onnx: str | None = Query(None, description="Path to ONNX classifier file"),
    yolo_weights: str | None = Query(None, description="Path to YOLO .pt weights file"),
):
    pred_dir = Path(__file__).parent / "predicted"
    pred_dir.mkdir(parents=True, exist_ok=True)
    try:
        suffix = Path(file.filename).suffix or ".jpg"
        filename = f"{uuid4().hex}{suffix}"
        dest_path = pred_dir / filename
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)

        # Call prediction using the saved path (response will include this path)
        result = predict_severity_json(
            image_path=str(dest_path),
            classifier_onnx=classifier_onnx,
            sam_checkpoint=None,
            yolo_weights=yolo_weights,
            skip_sam=not use_sam,
        )

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    classifier_onnx: str | None = Query(None, description="Path to ONNX classifier file (defaults to mobilenet if available)"),
):
    pred_dir = Path(__file__).parent / "predicted"
    pred_dir.mkdir(parents=True, exist_ok=True)
    try:
        suffix = Path(file.filename).suffix or ".jpg"
        filename = f"{uuid4().hex}{suffix}"
        dest_path = pred_dir / filename
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)

        # Select ONNX model (prefer provided path; else prefer MobileNet ONNX)
        if classifier_onnx:
            onnx_path = Path(classifier_onnx)
        else:
            mobilenet_candidate = Path(ROOT) / 'outputs' / 'models' / 'exported' / 'mobilenet_v3.onnx'
            if mobilenet_candidate.exists():
                onnx_path = mobilenet_candidate
            else:
                onnx_path = Path(find_default_onnx()) if find_default_onnx() else None

        if onnx_path is None or not Path(onnx_path).exists():
            raise HTTPException(status_code=400, detail=f"ONNX classifier not found: {onnx_path}")

        # Load class names (attempt via class map or companion .data file)
        class_names = load_class_map()
        if class_names is None:
            companion = onnx_path.with_suffix(onnx_path.suffix + '.data')
            if companion.exists():
                try:
                    meta = load_json(str(companion))
                    if 'class_names' in meta:
                        class_names = meta['class_names']
                except Exception:
                    pass
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(17)]

        # Run ONNX classification
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        image = cv2.imread(str(dest_path))
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to load image after upload")

        classification_result = onnx_predict(session, image, class_names, top_k=3)

        result = {
            "success": True,
            "image_path": str(dest_path),
            "disease": {
                "predicted_class": classification_result['predicted_class'],
                "confidence": float(classification_result['confidence']),
                "top_predictions": [
                    {"class": p['class'], "confidence": float(p['confidence'])} for p in classification_result['top_k'][:3]
                ]
            }
        }

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the app directly using the app object so the script works when executed from the Backend/ folder
    uvicorn.run(app, host="0.0.0.0", port=8000)
