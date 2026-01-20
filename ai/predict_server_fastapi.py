"""
FastAPI server wrapping predict_severity_api_onnx.predict_severity_json

Endpoints:
- GET /health - simple health check
- POST /predictwithseverity - complete pipeline: classification (ONNX), detection (YOLO), segmentation (SAM or HSV), and severity calculation
  - use_sam (bool) - default false (fast HSV fallback)
  - classifier_onnx (str) - path to ONNX classifier
  - yolo_weights (str) - path to YOLO .pt weights
- POST /predict - classification-only endpoint (uses MobileNet ONNX by default if available)
  - classifier_onnx (str) - optional path to ONNX classifier; defaults to MobileNet ONNX when present

Examples:
curl -F "file=@image.JPG" "http://127.0.0.1:8000/predictwithseverity"
curl -F "file=@image.JPG" "http://127.0.0.1:8000/predict"

Run server:
python predict_server_fastapi.py
or
uvicorn predict_server_fastapi:app --host 0.0.0.0 --port 8000
"""

import os
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

# Import prediction function from existing script
from predict_severity_api_onnx import predict_severity_json, onnx_predict, find_default_onnx, load_class_map, load_json
import cv2

app = FastAPI(title="SmartCrop Severity API (ONNX)")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predictwithseverity")
async def predict_with_severity(
    file: UploadFile = File(...),
    use_sam: bool = Query(False, description="If true, use SAM for segmentation (slower)."),
    classifier_onnx: str | None = Query(None, description="Path to ONNX classifier file"),
    yolo_weights: str | None = Query(None, description="Path to YOLO .pt weights file")
):
    try:
        contents = await file.read()
        import numpy as np
        arr = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode uploaded image")

        # Call prediction using in-memory image (no disk save)
        result = predict_severity_json(
            image=image,
            classifier_onnx=classifier_onnx,
            sam_checkpoint=None,
            yolo_weights=yolo_weights,
            skip_sam=not use_sam
        )

        result.pop('image_path', None)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint: classification-only using MobileNet ONNX (defaults to mobilenet if available)
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    classifier_onnx: str | None = Query(None, description="Path to ONNX classifier file (defaults to mobilenet if available)")
):
    try:
        contents = await file.read()
        import numpy as np
        arr = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode uploaded image")

        # Select ONNX model (prefer provided path; if not provided prefer MobileNet ONNX)
        if classifier_onnx:
            onnx_path = Path(classifier_onnx)
        else:
            mobilenet_candidate = Path('outputs/models/exported/mobilenet_v3.onnx')
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

        classification_result = onnx_predict(session, image, class_names, top_k=3)

        result = {
            "success": True,
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
    # Run with: python predict_server_fastapi.py
    uvicorn.run("predict_server_fastapi:app", host="0.0.0.0", port=8000)
