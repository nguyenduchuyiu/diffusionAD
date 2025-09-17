#!/usr/bin/env python3
"""
FastAPI app for image classification inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import ImageClassificationInference

app = FastAPI(title="Image Classification API", version="1.0.0")

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
    model_path = os.environ.get('MODEL_PATH', 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found")
        return
    
    try:
        inference_engine = ImageClassificationInference(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Image Classification API",
        "status": "running",
        "model_loaded": inference_engine is not None
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
async def predict_image(file: UploadFile = File(...)):
    """
    Predict image class
    
    Returns:
        JSON with prediction results
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
        result = inference_engine.predict(image)
        
        return {
            "success": True,
            "prediction": result,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict multiple images
    
    Returns:
        JSON with batch prediction results
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
            prediction = inference_engine.predict(image)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "prediction": prediction
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
