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
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from inference import preprocess_image, predict, load_checkpoint, defaultdict_from_json, denormalize_image, min_max_norm, cvt2heatmap, show_cam_on_image, predict_batch, predict_single_image_array
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
    /* Ensure all images in visualization columns have same height */
    div[data-testid="column"] div[data-testid="stImage"] img {
        max-height: 300px;
        width: 100%;
        object-fit: contain;
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
    
    # Use the function from inference.py
    return predict_single_image_array(unet_model, seg_model, ddpm, image_array, args, device, heatmap_threshold)

def predict_batch_images(models, image_arrays, batch_size=8, heatmap_threshold=0.6, progress_bar=None, status_text=None):
    """Predict anomalies for a batch of images with parallel processing"""
    unet_model = models['unet_model']
    seg_model = models['seg_model']
    ddpm = models['ddpm']
    args = models['args']
    device = models['device']
    
    # Progress callback function
    def progress_callback(progress, status):
        if progress_bar is not None and status_text is not None:
            progress_bar.progress(progress)
            status_text.text(status)
    
    # Use predict_batch from inference.py
    results = predict_batch(
        unet_model, seg_model, ddpm, image_arrays, args, device, 
        heatmap_threshold=heatmap_threshold, 
        batch_size=batch_size,
        progress_callback=progress_callback if progress_bar is not None else None
    )
    
    return results

def display_prediction_results(result, method, image):
    """Display prediction results"""
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    with col4:
        if 'inference_time' in result:
            st.metric(
                label="Inference Latency",
                value=f"{result['inference_time']*1000:.2f} ms",
                delta=None
            )
        else:
            st.metric(
                label="Inference Latency",
                value="N/A",
                delta=None
            )
    
    # Display visualizations for diffusion model
    if 'heatmap' in result:
        st.subheader("Visual Analysis")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.subheader("Original Image")
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
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider("Anomaly Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
        with col2:
            batch_size = st.slider("Batch Size", min_value=1, max_value=32, value=8, step=1)
        
        if st.button("Analyze Batch"):
            if models is None:
                st.error("Diffusion models not available!")
                return
            
            # Convert uploaded files to image arrays
            image_arrays = []
            filenames = []
            
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                image_arrays.append(image_array)
                filenames.append(uploaded_file.name)
            
            # Process in batches
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text(f'Initializing... Processing {len(uploaded_files)} images in batches of {batch_size}...')
            
            results = predict_batch_images(models, image_arrays, batch_size=batch_size, heatmap_threshold=threshold, 
                                         progress_bar=progress_bar, status_text=status_text)
            
            # Add filenames to results
            for i, result in enumerate(results):
                result['filename'] = filenames[i]
                result['confidence'] = abs(result['anomaly_score'])
            
            progress_bar.progress(1.0)
            status_text.text(f'Analysis complete! Processed {len(results)} images.')
            
            # Display results
            df = pd.DataFrame(results)
            
            # Summary statistics
            col1, col2, col3, col4, col5 = st.columns(5)
            
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
            with col5:
                avg_latency = df['inference_time'].mean() * 1000  # Convert to ms
                st.metric("Avg Latency", f"{avg_latency:.2f} ms")
            
            # Results table
            st.subheader("Detailed Results")
            # Format latency column for display
            df_display = df.copy()
            df_display['inference_time_ms'] = df_display['inference_time'].apply(lambda x: f"{x*1000:.2f} ms")
            df_display = df_display[['filename', 'anomaly_score', 'is_anomaly', 'confidence', 'inference_time_ms']]
            st.dataframe(df_display)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Score Distribution")
                fig = px.histogram(df, x='anomaly_score', color='is_anomaly',
                                 title='Anomaly Score Distribution',
                                 labels={'anomaly_score': 'Anomaly Score', 'count': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Latency Distribution")
                fig = px.histogram(df, x='inference_time', 
                                 title='Inference Time Distribution',
                                 labels={'inference_time': 'Inference Time (seconds)', 'count': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
            
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
