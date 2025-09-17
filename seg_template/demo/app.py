#!/usr/bin/env python3
"""
Streamlit demo app for semantic segmentation
"""

import streamlit as st
import cv2
import numpy as np
import sys
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import SegmentationInference

@st.cache_resource
def load_model(model_path, threshold):
    """Load model with caching"""
    try:
        return SegmentationInference(model_path, threshold=threshold)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def display_segmentation_results(image, result, show_overlay=True, show_mask=True):
    """Display segmentation results"""
    
    cols = st.columns(3 if show_overlay and show_mask else 2)
    
    with cols[0]:
        st.image(image, caption="Original Image", use_container_width=True)
    
    col_idx = 1
    if show_mask:
        with cols[col_idx]:
            st.image(result['mask'], caption=f"Segmentation Mask", use_container_width=True)
            col_idx += 1
    
    if show_overlay:
        with cols[col_idx]:
            st.image(result['overlay'], caption="Overlay", use_container_width=True)

def create_defect_analysis_chart(defect_ratio):
    """Create defect analysis visualization"""
    
    # Create gauge chart for defect ratio
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = defect_ratio * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Defect Area (%)"},
        delta = {'reference': 1.0},  # 1% threshold
        gauge = {
            'axis': {'range': [None, 10]},  # 0-10%
            'bar': {'color': "red" if defect_ratio > 0.01 else "green"},
            'steps': [
                {'range': [0, 1], 'color': "lightgreen"},
                {'range': [1, 5], 'color': "yellow"},
                {'range': [5, 10], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 1.0
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    st.set_page_config(
        page_title="Segmentation Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Semantic Segmentation Demo")
    st.write("Upload images to detect and segment defects using U-Net/DeepLabv3+")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="best_segmentation.pth",
        help="Path to the trained segmentation model"
    )
    
    # Segmentation parameters
    threshold = st.sidebar.slider(
        "Segmentation Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Threshold for binary segmentation"
    )
    
    input_size = st.sidebar.selectbox(
        "Input Size",
        [256, 512, 768, 1024],
        index=1,
        help="Input image size for the model"
    )
    
    # Load model
    inference_engine = load_model(model_path, threshold)
    
    if inference_engine is None:
        st.error(f"Could not load model from: {model_path}")
        st.info("Please make sure the model file exists and is valid")
        return
    
    # Display model info
    st.sidebar.success("✅ Model loaded successfully")
    st.sidebar.write(f"**Model:** {inference_engine.metadata['model_name']}")
    st.sidebar.write(f"**Encoder:** {inference_engine.metadata['encoder']}")
    st.sidebar.write(f"**Classes:** {inference_engine.metadata['num_classes']}")
    st.sidebar.write(f"**Device:** {inference_engine.device}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image file for defect segmentation"
        )
        
        # Display options
        st.subheader("Display Options")
        show_mask = st.checkbox("Show Segmentation Mask", value=True)
        show_overlay = st.checkbox("Show Overlay", value=True)
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Convert PIL to numpy array
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                # RGBA image, convert to RGB
                image_np = image_np[:, :, :3]
            elif len(image_np.shape) == 2:
                # Grayscale, convert to RGB
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    
    with col2:
        st.header("Segmentation Results")
        
        if uploaded_file is not None:
            # Segment button
            if st.button("🔍 Segment Defects", type="primary"):
                with st.spinner("Segmenting image..."):
                    try:
                        # Predict
                        result = inference_engine.predict(
                            image_np, 
                            input_size=input_size, 
                            threshold=threshold
                        )
                        
                        # Display results
                        st.success("✅ Segmentation Complete!")
                        
                        # Main metrics
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric(
                                label="Defect Area",
                                value=f"{result['defect_area_ratio']:.1%}",
                                delta="Defective" if result['defect_area_ratio'] > 0.01 else "Clean"
                            )
                        
                        with col_b:
                            st.metric(
                                label="Confidence",
                                value=f"{result['confidence']:.3f}",
                                delta=f"Threshold: {threshold}"
                            )
                        
                        with col_c:
                            defect_pixels = int(result['defect_area_ratio'] * 
                                               result['original_size'][0] * result['original_size'][1])
                            st.metric(
                                label="Defect Pixels",
                                value=f"{defect_pixels:,}",
                                delta=f"Size: {result['original_size'][0]}x{result['original_size'][1]}"
                            )
                        
                        # Defect analysis gauge
                        st.write("**Defect Analysis:**")
                        gauge_fig = create_defect_analysis_chart(result['defect_area_ratio'])
                        st.plotly_chart(gauge_fig, use_container_width=True)
                        
                        # Status indicator
                        if result['defect_area_ratio'] > 0.05:  # 5%
                            st.error("🚨 **HIGH DEFECT LEVEL DETECTED**")
                        elif result['defect_area_ratio'] > 0.01:  # 1%
                            st.warning("⚠️ **DEFECTS DETECTED**")
                        else:
                            st.success("✅ **NO SIGNIFICANT DEFECTS**")
                        
                    except Exception as e:
                        st.error(f"Error during segmentation: {e}")
        else:
            st.info("👆 Upload an image to see segmentation results")
    
    # Display segmentation visualization
    if uploaded_file is not None and 'result' in locals():
        st.header("🎨 Segmentation Visualization")
        display_segmentation_results(image_np, result, show_overlay, show_mask)
    
    # Batch processing section
    st.header("📊 Batch Processing")
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Upload multiple images for batch segmentation",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,
        help="Upload multiple images to segment them all at once"
    )
    
    if uploaded_files and len(uploaded_files) > 0:
        st.info(f"Selected {len(uploaded_files)} images for batch processing")
        
        if st.button("🔄 Process Batch", type="secondary"):
            with st.spinner(f"Processing {len(uploaded_files)} images..."):
                batch_results = []
                progress_bar = st.progress(0)
                
                # Process each image
                for i, file in enumerate(uploaded_files):
                    try:
                        image = Image.open(file)
                        image_np = np.array(image)
                        
                        # Handle different image formats
                        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                            image_np = image_np[:, :, :3]
                        elif len(image_np.shape) == 2:
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                        
                        result = inference_engine.predict(
                            image_np, 
                            input_size=input_size, 
                            threshold=threshold
                        )
                        
                        batch_results.append({
                            'Filename': file.name,
                            'Defect Area (%)': f"{result['defect_area_ratio']:.2%}",
                            'Confidence': f"{result['confidence']:.3f}",
                            'Has Defect': "Yes" if result['defect_area_ratio'] > 0.01 else "No",
                            'Image Size': f"{result['original_size'][0]}x{result['original_size'][1]}",
                            'Status': ("High Risk" if result['defect_area_ratio'] > 0.05 
                                     else "Defective" if result['defect_area_ratio'] > 0.01 
                                     else "Clean")
                        })
                        
                    except Exception as e:
                        batch_results.append({
                            'Filename': file.name,
                            'Defect Area (%)': 'Error',
                            'Confidence': '0.000',
                            'Has Defect': 'Error',
                            'Image Size': 'N/A',
                            'Status': f'Error: {str(e)}'
                        })
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display batch results
                st.success(f"✅ Processed {len(batch_results)} images")
                
                # Results table
                batch_df = pd.DataFrame(batch_results)
                st.dataframe(batch_df, use_container_width=True)
                
                # Batch statistics
                if len(batch_results) > 1:
                    st.write("**Batch Summary:**")
                    
                    # Count statistics
                    successful_results = [r for r in batch_results if r['Status'] != 'Error']
                    defective_count = len([r for r in successful_results if r['Has Defect'] == 'Yes'])
                    high_risk_count = len([r for r in successful_results if r['Status'] == 'High Risk'])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Images", len(batch_results))
                    
                    with col2:
                        st.metric("Defective Images", defective_count)
                    
                    with col3:
                        st.metric("High Risk Images", high_risk_count)
                    
                    with col4:
                        defect_rate = defective_count / len(successful_results) if successful_results else 0
                        st.metric("Defect Rate", f"{defect_rate:.1%}")
                    
                    # Status distribution chart
                    if successful_results:
                        st.write("**Status Distribution:**")
                        status_counts = {}
                        for result in successful_results:
                            status = result['Status']
                            status_counts[status] = status_counts.get(status, 0) + 1
                        
                        status_df = pd.DataFrame([
                            {'Status': k, 'Count': v} for k, v in status_counts.items()
                        ])
                        
                        fig = px.pie(
                            status_df, 
                            values='Count', 
                            names='Status', 
                            title="Image Status Distribution",
                            color_discrete_map={
                                'Clean': 'green',
                                'Defective': 'orange', 
                                'High Risk': 'red'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="batch_segmentation_results.csv",
                    mime="text/csv"
                )
    
    # Information section
    with st.expander("ℹ️ Model Information"):
        st.write("**Segmentation Model Details:**")
        if inference_engine:
            st.write(f"- **Architecture:** {inference_engine.metadata['model_name']}")
            st.write(f"- **Encoder:** {inference_engine.metadata['encoder']}")
            st.write(f"- **Classes:** {inference_engine.metadata['num_classes']}")
            st.write(f"- **Activation:** {inference_engine.metadata['activation']}")
            st.write(f"- **Threshold:** {threshold}")
        
        st.write("**Use Cases:**")
        st.write("- PCB defect detection")
        st.write("- Surface defect segmentation")
        st.write("- Quality control inspection")
        st.write("- Manufacturing defect analysis")
        
        st.write("**Metrics:**")
        st.write("- **IoU (Intersection over Union):** Overlap between prediction and ground truth")
        st.write("- **Dice Coefficient:** Similarity measure for segmentation")
        st.write("- **Defect Area Ratio:** Percentage of image area classified as defective")

if __name__ == "__main__":
    main()