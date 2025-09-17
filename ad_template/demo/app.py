#!/usr/bin/env python3
"""
Streamlit demo app for anomaly detection using diffusion model
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import torch
import json
from collections import defaultdict

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from inference import preprocess_image, predict, load_checkpoint, defaultdict_from_json, denormalize_image, min_max_norm, cvt2heatmap, show_cam_on_image
    from models import UNetModel, SegmentationSubNetwork, GaussianDiffusionModel, get_beta_schedule
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Diffusion Anomaly Detection Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .anomaly-score {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .normal-score {
        color: #28a745;
    }
    .anomaly-score-high {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained diffusion models with caching"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration paths
    model_paths = ['outputs/model/diff-params-ARGS=1/PCB5/params-last.pt', os.path.join('..', 'outputs/model/diff-params-ARGS=1/PCB5/params-last.pt')]
    args_paths = ['args/args1.json', os.path.join('..', 'args/args1.json')]
    
    # Load model checkpoint
    ckpt_state = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                ckpt_state = load_checkpoint(model_path, device)
                st.success(f"Model checkpoint loaded from {model_path}")
                break
            except Exception as e:
                st.warning(f"Failed to load model from {model_path}: {e}")
    
    if ckpt_state is None:
        st.error("No model checkpoint found!")
        return None
    
    # Load args
    args = None
    for args_path in args_paths:
        if os.path.exists(args_path):
            try:
                with open(args_path) as f:
                    args = json.load(f)
                args = defaultdict_from_json(args)
                st.success(f"Args loaded from {args_path}")
                break
            except Exception as e:
                st.warning(f"Failed to load args from {args_path}: {e}")
    
    if args is None:
        st.error("No args file found!")
        return None
    
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
        
        st.success("All diffusion models loaded successfully!")
        
        return {
            'unet_model': unet_model,
            'seg_model': seg_model,
            'ddpm': ddpm,
            'args': args,
            'device': device
        }
        
    except Exception as e:
        st.error(f"Failed to initialize models: {e}")
        return None

def predict_single_image(models, image_array, heatmap_threshold=0.6):
    """Predict anomaly for single image using diffusion model"""
    unet_model = models['unet_model']
    seg_model = models['seg_model']
    ddpm = models['ddpm']
    args = models['args']
    device = models['device']
    
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

    # Create visualizations
    raw_image = denormalize_image(image_tensor)
    recon_condition = denormalize_image(pred_x_0_condition)
    recon_noisier_t = denormalize_image(pred_x_0_noisier)
    
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
    
    return {
        "anomaly_score": float(image_score),
        "is_anomaly": image_score > heatmap_threshold,
        "original_image": raw_image,
        "reconstructed_image": recon_condition,
        "recon_noisier": recon_noisier_t,
        "anomaly_mask": (out_mask[0][0].cpu().numpy() * 255.0).astype(np.uint8),
        "heatmap_overlay": ano_map_overlay,
        "heatmap": ano_map
    }

def display_prediction_results(result, method, image):
    """Display prediction results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Anomaly Score",
            value=f"{result['anomaly_score']:.4f}",
            delta=None
        )
    
    with col2:
        is_anomaly = result['is_anomaly']
        status = "Anomaly" if is_anomaly else "Normal"
        color = "red" if is_anomaly else "green"
        st.markdown(f"<div style='color: {color}; font-size: 1.2em; font-weight: bold;'>{status}</div>", 
                   unsafe_allow_html=True)
    
    with col3:
        confidence = abs(result['anomaly_score'])
        st.metric(
            label="Confidence",
            value=f"{confidence:.4f}",
            delta=None
        )
    
    # Display visualizations for diffusion model
    if 'heatmap' in result:
        st.subheader("Visual Analysis")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.subheader("Original")
            st.image(result['original_image'], use_container_width=True)
        
        with col2:
            st.subheader("Reconstruction")
            st.image(result['reconstructed_image'], use_container_width=True)
            
        with col3:
            st.subheader("Recon (Noisier)")
            st.image(result['recon_noisier'], use_container_width=True)
        
        with col4:
            st.subheader("Anomaly Mask")
            st.image(result['anomaly_mask'], use_container_width=True)
            
        with col5:
            st.subheader("Heatmap Overlay")
            st.image(result['heatmap_overlay'], use_container_width=True)

def batch_analysis_page(models):
    """Batch analysis page"""
    st.header("Batch Analysis")
    
    uploaded_files = st.file_uploader(
        "Upload multiple images for batch analysis",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        threshold = st.slider("Anomaly Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
        
        if st.button("Analyze Batch"):
            if models is None:
                st.error("Diffusion models not available!")
                return
            
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f'Processing {uploaded_file.name}...')
                
                # Convert uploaded file to image
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                # Predict
                result = predict_single_image(models, image_array, heatmap_threshold=threshold)
                
                results.append({
                    'filename': uploaded_file.name,
                    'anomaly_score': result['anomaly_score'],
                    'is_anomaly': result['is_anomaly'],
                    'confidence': abs(result['anomaly_score'])
                })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text('Analysis complete!')
            
            # Display results
            df = pd.DataFrame(results)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Images", len(results))
            with col2:
                anomaly_count = sum(r['is_anomaly'] for r in results)
                st.metric("Anomalies Found", anomaly_count)
            with col3:
                normal_count = len(results) - anomaly_count
                st.metric("Normal Images", normal_count)
            with col4:
                anomaly_rate = anomaly_count / len(results) * 100 if results else 0
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
            
            # Results table
            st.subheader("Detailed Results")
            st.dataframe(df)
            
            # Visualizations
            st.subheader("Score Distribution")
            fig = px.histogram(df, x='anomaly_score', color='is_anomaly',
                             title='Anomaly Score Distribution',
                             labels={'anomaly_score': 'Anomaly Score', 'count': 'Count'})
            st.plotly_chart(fig)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="anomaly_detection_results.csv",
                mime="text/csv"
            )

def main():
    """Main application"""
    # Title
    st.markdown("<h1 class='main-header'>🔍 Diffusion Anomaly Detection Demo</h1>", unsafe_allow_html=True)
    
    # Load models
    if 'models' not in st.session_state:
        st.session_state.models = load_models()
    
    models = st.session_state.models
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Page selection
    page = st.sidebar.selectbox("Select Page", ["Single Image Analysis", "Batch Analysis"])
    
    # Model info
    st.sidebar.subheader("Model Information")
    if models:
        st.sidebar.text(f"Device: {models['device']}")
        st.sidebar.text(f"Image Size: {models['args']['img_size']}")
        st.sidebar.text(f"Base Channels: {models['args']['base_channels']}")
        st.sidebar.text(f"Timesteps: {models['args']['T']}")
    else:
        st.sidebar.error("Models not loaded!")
    
    if page == "Single Image Analysis":
        if not models:
            st.error("No models available! Please check model files.")
            st.stop()
        
        # Main content
        st.header("Single Image Analysis")
        
        # Threshold slider
        threshold = st.slider("Anomaly Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image for anomaly detection",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.subheader("Uploaded Image")
            st.image(image, caption=uploaded_file.name, use_container_width=True)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Prediction
            if st.button("Detect Anomaly", type="primary"):
                with st.spinner("Analyzing image..."):
                    result = predict_single_image(models, image_array, heatmap_threshold=threshold)
                    
                    st.subheader("Detection Results")
                    display_prediction_results(result, "diffusion_model", image_array)
        
        # Sample images section
        st.subheader("Try with Sample Images")
        sample_dir = Path("../datasets/RealIAD/PCB5/test")
        if sample_dir.exists():
            sample_subdirs = [d for d in sample_dir.iterdir() if d.is_dir()]
            if sample_subdirs:
                selected_subdir = st.selectbox(
                    "Select category",
                    sample_subdirs,
                    format_func=lambda x: x.name
                )
                
                sample_images = list(selected_subdir.glob("*.jpg")) + list(selected_subdir.glob("*.png"))
                if sample_images:
                    selected_sample = st.selectbox(
                        "Select a sample image",
                        sample_images[:10],  # Limit to first 10 images
                        format_func=lambda x: x.name
                    )
                    
                    if st.button("Analyze Sample"):
                        sample_image = np.array(Image.open(selected_sample))
                        st.image(sample_image, caption=str(selected_sample), use_container_width=True)
                        
                        with st.spinner("Analyzing sample image..."):
                            result = predict_single_image(models, sample_image, heatmap_threshold=threshold)
                            display_prediction_results(result, "diffusion_model", sample_image)
    
    elif page == "Batch Analysis":
        batch_analysis_page(models)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Diffusion Anomaly Detection Demo | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
