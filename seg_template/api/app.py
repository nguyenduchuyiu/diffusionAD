#!/usr/bin/env python3
"""
FastAPI app for semantic segmentation inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
import sys
import os
import tempfile
import base64
from io import BytesIO
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import SegmentationInference

app = FastAPI(title="Segmentation API", version="1.0.0")

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
    model_path = os.environ.get('MODEL_PATH', 'best_segmentation.pth')
    threshold = float(os.environ.get('THRESHOLD', '0.5'))
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found")
        inference_engine = None
        return
    
    try:
        inference_engine = SegmentationInference(model_path, threshold=threshold)
        print("Segmentation model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        inference_engine = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Segmentation API",
        "status": "running",
        "model_loaded": inference_engine is not None,
        "model_type": "Semantic Segmentation"
    }

@app.get("/model_info")
async def get_model_info():
    """Get model information"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": inference_engine.metadata['model_name'],
        "encoder": inference_engine.metadata['encoder'],
        "num_classes": inference_engine.metadata['num_classes'],
        "activation": inference_engine.metadata['activation'],
        "threshold": inference_engine.threshold
    }

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    threshold: float = Query(default=None, description="Segmentation threshold"),
    input_size: int = Query(default=512, description="Input image size"),
    return_overlay: bool = Query(default=True, description="Return overlay image"),
    return_mask: bool = Query(default=False, description="Return mask as base64")
):
    """
    Segment defects in image
    
    Returns:
        JSON with segmentation results including mask and overlay
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
        result = inference_engine.predict(image, input_size=input_size, threshold=threshold)
        
        response = {
            "success": True,
            "filename": file.filename,
            "segmentation": {
                "defect_area_ratio": result['defect_area_ratio'],
                "confidence": result['confidence'],
                "has_defect": result['defect_area_ratio'] > 0.01,  # 1% threshold
                "image_size": {
                    "width": result['original_size'][0],
                    "height": result['original_size'][1]
                }
            }
        }
        
        # Add overlay as base64 if requested
        if return_overlay:
            overlay_pil = Image.fromarray(result['overlay'])
            buffer = BytesIO()
            overlay_pil.save(buffer, format='PNG')
            overlay_b64 = base64.b64encode(buffer.getvalue()).decode()
            response["overlay_base64"] = overlay_b64
        
        # Add mask as base64 if requested
        if return_mask:
            mask_pil = Image.fromarray(result['mask'])
            buffer = BytesIO()
            mask_pil.save(buffer, format='PNG')
            mask_b64 = base64.b64encode(buffer.getvalue()).decode()
            response["mask_base64"] = mask_b64
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    threshold: float = Query(default=None, description="Segmentation threshold"),
    input_size: int = Query(default=512, description="Input image size")
):
    """
    Segment defects in multiple images
    
    Returns:
        JSON with batch segmentation results
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    total_defect_area = 0
    images_with_defects = 0
    
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
            result = inference_engine.predict(image, input_size=input_size, threshold=threshold)
            
            # Track statistics
            total_defect_area += result['defect_area_ratio']
            if result['defect_area_ratio'] > 0.01:  # 1% threshold
                images_with_defects += 1
            
            results.append({
                "filename": file.filename,
                "success": True,
                "segmentation": {
                    "defect_area_ratio": result['defect_area_ratio'],
                    "confidence": result['confidence'],
                    "has_defect": result['defect_area_ratio'] > 0.01,
                    "image_size": {
                        "width": result['original_size'][0],
                        "height": result['original_size'][1]
                    }
                }
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    # Calculate batch statistics
    successful_results = [r for r in results if r['success']]
    
    return {
        "success": True,
        "results": results,
        "batch_statistics": {
            "total_processed": len(results),
            "successful": len(successful_results),
            "failed": len(results) - len(successful_results),
            "images_with_defects": images_with_defects,
            "avg_defect_area_ratio": total_defect_area / len(successful_results) if successful_results else 0,
            "defect_detection_rate": images_with_defects / len(successful_results) if successful_results else 0
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
