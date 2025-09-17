#!/usr/bin/env python3
"""
FastAPI application for anomaly detection using diffusion model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import torch
import json
from collections import defaultdict

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import preprocess_image, predict, load_checkpoint, defaultdict_from_json, denormalize_image, min_max_norm, cvt2heatmap, show_cam_on_image
from models import UNetModel, SegmentationSubNetwork, GaussianDiffusionModel, get_beta_schedule

app = FastAPI(
    title="Anomaly Detection API",
    description="API for image anomaly detection using diffusion model",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
unet_model = None
seg_model = None
ddpm = None
args = None
device = None

# Configuration
CONFIG = {
    'model_path': 'outputs/model/diff-params-ARGS=1/PCB5/params-last.pt',
    'args_path': 'args/args1.json',
    'upload_dir': 'uploads',
    'results_dir': 'results'
}

def load_models():
    """Load trained diffusion models"""
    global unet_model, seg_model, ddpm, args, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model checkpoint
    model_paths = [CONFIG['model_path'], os.path.join('..', CONFIG['model_path'])]
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                ckpt_state = load_checkpoint(model_path, device)
                print(f"Model checkpoint loaded successfully from {model_path}")
                break
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")
                continue
    else:
        print("No model checkpoint found")
        return
    
    # Load args
    args_paths = [CONFIG['args_path'], os.path.join('..', CONFIG['args_path'])]
    for args_path in args_paths:
        if os.path.exists(args_path):
            try:
                with open(args_path) as f:
                    args = json.load(f)
                args = defaultdict_from_json(args)
                print(f"Args loaded successfully from {args_path}")
                break
            except Exception as e:
                print(f"Failed to load args from {args_path}: {e}")
                continue
    else:
        print("No args file found")
        return
    
    try:
        # Initialize models
        unet_model = UNetModel(
            args['img_size'][0], 
            args['base_channels'], 
            channel_mults=args['channel_mults'], 
            dropout=args["dropout"], 
            n_heads=args["num_heads"], 
            n_head_channels=args["num_head_channels"],
            in_channels=args["channels"]
        ).to(device)

        seg_model = SegmentationSubNetwork(in_channels=6, out_channels=1).to(device)

        # Initialize DDPM
        betas = get_beta_schedule(args['T'], args['beta_schedule'])
        ddpm = GaussianDiffusionModel(
            args['img_size'], 
            betas, 
            loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], 
            noise=args["noise_fn"], 
            img_channels=args["channels"]
        )
        
        # Load state dicts
        unet_model.load_state_dict(ckpt_state['unet_model_state_dict'])
        seg_model.load_state_dict(ckpt_state['seg_model_state_dict'])
        unet_model.eval()
        seg_model.eval()
        
        print("All models loaded and initialized successfully")
        
    except Exception as e:
        print(f"Failed to initialize models: {e}")
        unet_model = None
        seg_model = None
        ddpm = None

def setup_directories():
    """Setup required directories"""
    os.makedirs(CONFIG['upload_dir'], exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)

def image_to_base64(image_array):
    """Convert numpy image array to base64 string"""
    # Convert to uint8 if needed
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def create_heatmap_base64(heatmap):
    """Convert heatmap to base64 string"""
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap='hot')
    plt.colorbar()
    plt.axis('off')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def predict_single_image(image_array, return_visualizations=True, heatmap_threshold=0.6):
    """Predict anomaly for single image using diffusion model"""
    global unet_model, seg_model, ddpm, args, device
    
    if unet_model is None or seg_model is None or ddpm is None:
        raise ValueError("Models not loaded")
    
    # Preprocess image
    if len(image_array.shape) == 3:
        image_tensor = torch.from_numpy(np.transpose(image_array.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0)
    else:
        raise ValueError("Image must be 3D array (H, W, C)")
    
    # Resize to model input size
    image_tensor = torch.nn.functional.interpolate(image_tensor, size=args['img_size'], mode='bilinear', align_corners=False)
    image_tensor = image_tensor.to(device)

    normal_t = args["eval_normal_t"]
    noiser_t = args["eval_noisier_t"]
    
    normal_t_tensor = torch.tensor([normal_t], device=device).repeat(image_tensor.shape[0])
    noiser_t_tensor = torch.tensor([noiser_t], device=device).repeat(image_tensor.shape[0])

    with torch.no_grad():
        _, pred_x_0_condition, pred_x_0_normal, pred_x_0_noisier, x_normal_t, x_noiser_t, pred_x_t_noisier = ddpm.norm_guided_one_step_denoising_eval(unet_model, image_tensor, normal_t_tensor, noiser_t_tensor, args)
        pred_mask_logits = seg_model(torch.cat((image_tensor, pred_x_0_condition), dim=1))
            
    pred_mask = torch.sigmoid(pred_mask_logits)
    out_mask = pred_mask

    # Calculate anomaly score
    topk_out_mask = torch.flatten(out_mask[0], start_dim=1)
    topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
    image_score = torch.mean(topk_out_mask).cpu().item()

    result = {
        "anomaly_score": float(image_score),
        "is_anomaly": image_score > heatmap_threshold
    }
    
    if return_visualizations:
        # Create visualizations
        raw_image = denormalize_image(image_tensor)
        recon_condition = denormalize_image(pred_x_0_condition)
        
        # Create heatmap
        mask_data = out_mask[0, 0].cpu().numpy().astype(np.float32)
        mask_data[mask_data < heatmap_threshold] = 0
        ano_map = cv2.GaussianBlur(mask_data, (15, 15), 4)
        ano_map = min_max_norm(ano_map)
        ano_map_heatmap = cvt2heatmap(ano_map * 255.0)
        
        # Create overlay
        raw_image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        ano_map_overlay = show_cam_on_image(raw_image_bgr, ano_map_heatmap)
        ano_map_overlay = cv2.cvtColor(ano_map_overlay, cv2.COLOR_BGR2RGB)
        
        result.update({
            "original_image": raw_image,
            "reconstructed_image": recon_condition,
            "anomaly_mask": (out_mask[0][0].cpu().numpy() * 255.0).astype(np.uint8),
            "heatmap_overlay": ano_map_overlay,
            "heatmap": ano_map
        })
    
    return result

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    setup_directories()
    load_models()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Anomaly Detection API",
        "version": "1.0.0",
        "method": "diffusion_model",
        "models_available": unet_model is not None and seg_model is not None and ddpm is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "unet_loaded": unet_model is not None,
        "seg_model_loaded": seg_model is not None,
        "ddpm_loaded": ddpm is not None,
        "device": str(device) if device else None
    }

@app.post("/predict")
async def predict_anomaly(
    file: UploadFile = File(...),
    return_heatmap: bool = Form(True),
    threshold: float = Form(0.6)
):
    """
    Predict anomaly in uploaded image using diffusion model
    
    Args:
        file: Image file
        return_heatmap: Whether to return visualization heatmap
        threshold: Anomaly threshold
    
    Returns:
        JSON response with prediction results
    """
    try:
        # Check if models are available
        if unet_model is None or seg_model is None or ddpm is None:
            raise HTTPException(status_code=503, detail="Diffusion models not available")
        
        # Read and validate image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict anomaly
        result = predict_single_image(image, return_visualizations=return_heatmap, heatmap_threshold=threshold)
        
        response = {
            "method": "diffusion_model",
            "anomaly_score": result["anomaly_score"],
            "is_anomaly": result["is_anomaly"],
            "confidence": abs(result["anomaly_score"]),
            "threshold": threshold
        }
        
        # Add visualizations if requested
        if return_heatmap and "original_image" in result:
            response["original_image"] = image_to_base64(result["original_image"])
            response["reconstructed_image"] = image_to_base64(result["reconstructed_image"])
            response["anomaly_mask"] = image_to_base64(result["anomaly_mask"])
            response["heatmap_overlay"] = image_to_base64(result["heatmap_overlay"])
            response["heatmap"] = create_heatmap_base64(result["heatmap"])
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    threshold: float = Form(0.6)
):
    """
    Predict anomaly for multiple images using diffusion model
    
    Args:
        files: List of image files
        threshold: Anomaly threshold
    
    Returns:
        JSON response with batch prediction results
    """
    try:
        # Check if models are available
        if unet_model is None or seg_model is None or ddpm is None:
            raise HTTPException(status_code=503, detail="Diffusion models not available")
        
        if len(files) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
        
        images = []
        filenames = []
        
        # Process all images
        for file in files:
            contents = await file.read()
            if not contents:
                continue
            
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            filenames.append(file.filename)
        
        if not images:
            raise HTTPException(status_code=400, detail="No valid images found")
        
        # Predict anomalies for all images
        results = []
        for image in images:
            result = predict_single_image(image, return_visualizations=False, heatmap_threshold=threshold)
            results.append(result)
        
        # Format response
        response = {
            "method": "diffusion_model",
            "num_images": len(results),
            "threshold": threshold,
            "results": []
        }
        
        for i, result in enumerate(results):
            response["results"].append({
                "filename": filenames[i],
                "anomaly_score": result["anomaly_score"],
                "is_anomaly": result["is_anomaly"],
                "confidence": abs(result["anomaly_score"])
            })
        
        # Summary statistics
        anomaly_count = sum(1 for r in results if r["is_anomaly"])
        response["summary"] = {
            "total_images": len(results),
            "anomaly_count": anomaly_count,
            "normal_count": len(results) - anomaly_count,
            "anomaly_rate": anomaly_count / len(results) if results else 0
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models"""
    info = {
        "diffusion_model": {
            "unet_available": unet_model is not None,
            "seg_model_available": seg_model is not None,
            "ddpm_available": ddpm is not None,
            "model_path": CONFIG['model_path'],
            "args_path": CONFIG['args_path'],
            "device": str(device) if device else None
        }
    }
    
    if args:
        info["diffusion_model"]["img_size"] = args.get('img_size', 'unknown')
        info["diffusion_model"]["base_channels"] = args.get('base_channels', 'unknown')
        info["diffusion_model"]["T"] = args.get('T', 'unknown')
    
    return info

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
