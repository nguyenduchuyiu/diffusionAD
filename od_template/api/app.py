#!/usr/bin/env python3
"""
FastAPI app for YOLOv8 object detection inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
import sys
import os
import tempfile
import uuid

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import ObjectDetectionInference

app = FastAPI(title="Object Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global inference_engine
    
    # Model path - can be configured via environment variable
    model_path = os.environ.get('MODEL_PATH', 'best_yolo.pt')
    confidence = float(os.environ.get('CONFIDENCE', '0.25'))
    iou = float(os.environ.get('IOU', '0.45'))
    
    try:
        inference_engine = ObjectDetectionInference(
            model_path, 
            confidence=confidence, 
            iou=iou
        )
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Object Detection API",
        "status": "running",
        "model_loaded": inference_engine is not None,
        "model_type": "YOLOv8"
    }

@app.get("/classes")
async def get_classes():
    """Get available classes"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": inference_engine.class_names,
        "num_classes": len(inference_engine.class_names)
    }

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    confidence: float = Query(default=None, description="Confidence threshold"),
    iou: float = Query(default=None, description="IoU threshold"),
    draw_boxes: bool = Query(default=False, description="Return image with drawn bounding boxes")
):
    """
    Detect objects in image
    
    Returns:
        JSON with detection results including bounding boxes
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict
        result = inference_engine.predict(image, confidence=confidence, iou=iou)
        
        response = {
            "success": True,
            "detections": result,
            "filename": file.filename
        }
        
        # Optionally return annotated image
        if draw_boxes and result['num_detections'] > 0:
            annotated_img = inference_engine.draw_detections(image, result['detections'])
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(temp_file.name, annotated_img_bgr)
            
            response["annotated_image_path"] = temp_file.name
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    confidence: float = Query(default=None, description="Confidence threshold"),
    iou: float = Query(default=None, description="IoU threshold")
):
    """
    Detect objects in multiple images
    
    Returns:
        JSON with batch detection results
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "File must be an image"
                })
                continue
            
            # Read and process image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid image format"
                })
                continue
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Predict
            detections = inference_engine.predict(image, confidence=confidence, iou=iou)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "detections": detections
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "results": results,
        "total_processed": len(results)
    }

@app.get("/annotated_image/{image_path:path}")
async def get_annotated_image(image_path: str):
    """
    Download annotated image
    """
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        path=image_path,
        media_type="image/jpeg",
        filename=f"annotated_{uuid.uuid4().hex}.jpg"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
