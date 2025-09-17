#!/usr/bin/env python3
"""
Streamlit demo app for YOLOv8 object detection
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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import ObjectDetectionInference

@st.cache_resource
def load_model(model_path, confidence, iou):
    """Load model with caching"""
    try:
        return ObjectDetectionInference(model_path, confidence=confidence, iou=iou)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def draw_bounding_boxes(image, detections):
    """Draw bounding boxes on image using PIL/matplotlib style"""
    img = image.copy()
    
    # Define colors for different classes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    # Convert PIL to OpenCV format if needed
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        class_id = detection['class_id']
        
        # Get color for this class
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw label background
        cv2.rectangle(
            img, 
            (x1, y1 - label_size[1] - 10), 
            (x1 + label_size[0], y1), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            img, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            2
        )
    
    return img

def main():
    st.set_page_config(
        page_title="Object Detection Demo",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 YOLOv8 Object Detection Demo")
    st.write("Upload an image to detect objects and defects using YOLOv8")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="best_yolo.pt",
        help="Path to the trained YOLO model file"
    )
    
    # Detection parameters
    confidence = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.25, 
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    iou = st.sidebar.slider(
        "IoU Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.45, 
        step=0.05,
        help="IoU threshold for Non-Maximum Suppression"
    )
    
    # Load model
    inference_engine = load_model(model_path, confidence, iou)
    
    if inference_engine is None:
        st.error(f"Could not load model from: {model_path}")
        st.info("Loading default YOLOv8n model instead...")
        inference_engine = load_model("yolov8n.pt", confidence, iou)
        if inference_engine is None:
            st.error("Could not load any model. Please check your installation.")
            return
    
    # Display model info
    st.sidebar.success("✅ Model loaded successfully")
    st.sidebar.write(f"**Classes:** {len(inference_engine.class_names)}")
    
    # Class names
    with st.sidebar.expander("Class Names"):
        for class_id, class_name in inference_engine.class_names.items():
            st.write(f"{class_id}: {class_name}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image file for object detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Convert PIL to numpy array
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                # RGB image
                pass
            elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
                # RGBA image, convert to RGB
                image_np = image_np[:, :, :3]
            else:
                st.error("Unsupported image format")
                return
    
    with col2:
        st.header("Detection Results")
        
        if uploaded_file is not None:
            # Detect button
            if st.button("🎯 Detect Objects", type="primary"):
                with st.spinner("Detecting objects..."):
                    try:
                        # Predict
                        result = inference_engine.predict(image_np, confidence=confidence, iou=iou)
                        
                        # Display results
                        if result['num_detections'] > 0:
                            st.success(f"✅ Found {result['num_detections']} objects!")
                            
                            # Show annotated image
                            annotated_img = draw_bounding_boxes(image_np, result['detections'])
                            st.image(annotated_img, caption="Detected Objects", use_container_width=True)
                            
                            # Detection summary
                            st.write("**Detection Summary:**")
                            
                            # Count detections by class
                            class_counts = {}
                            for detection in result['detections']:
                                class_name = detection['class_name']
                                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                            
                            # Display class counts
                            for class_name, count in class_counts.items():
                                st.metric(f"{class_name}", count)
                            
                            # Detailed detections table
                            st.write("**All Detections:**")
                            detection_data = []
                            for i, detection in enumerate(result['detections']):
                                x1, y1, x2, y2 = detection['bbox']
                                detection_data.append({
                                    'ID': i + 1,
                                    'Class': detection['class_name'],
                                    'Confidence': f"{detection['confidence']:.3f}",
                                    'BBox': f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})",
                                    'Width': f"{x2-x1:.0f}",
                                    'Height': f"{y2-y1:.0f}"
                                })
                            
                            detection_df = pd.DataFrame(detection_data)
                            st.dataframe(detection_df, use_container_width=True)
                            
                            # Confidence distribution chart
                            if len(result['detections']) > 1:
                                st.write("**Confidence Distribution:**")
                                conf_data = [d['confidence'] for d in result['detections']]
                                classes = [d['class_name'] for d in result['detections']]
                                
                                fig = px.histogram(
                                    x=conf_data,
                                    nbins=10,
                                    title="Detection Confidence Distribution",
                                    labels={'x': 'Confidence', 'y': 'Count'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            st.info("No objects detected. Try adjusting the confidence threshold.")
                        
                        # Show image info
                        st.write("**Image Info:**")
                        st.write(f"Size: {result['image_size']['width']} x {result['image_size']['height']}")
                        
                    except Exception as e:
                        st.error(f"Error during detection: {e}")
        else:
            st.info("👆 Upload an image to see detections")
    
    # Additional features
    st.header("📊 Batch Processing")
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Upload multiple images for batch object detection",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,
        help="Upload multiple images to detect objects in all of them at once"
    )
    
    if uploaded_files and len(uploaded_files) > 0:
        if st.button("🔄 Process Batch", type="secondary"):
            with st.spinner(f"Processing {len(uploaded_files)} images..."):
                batch_results = []
                total_detections = 0
                
                # Process each image
                for i, file in enumerate(uploaded_files):
                    try:
                        image = Image.open(file)
                        image_np = np.array(image)
                        
                        # Handle different image formats
                        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                            image_np = image_np[:, :, :3]
                        
                        result = inference_engine.predict(image_np, confidence=confidence, iou=iou)
                        total_detections += result['num_detections']
                        
                        # Get top detection for summary
                        top_detection = None
                        if result['detections']:
                            top_detection = max(result['detections'], key=lambda x: x['confidence'])
                        
                        batch_results.append({
                            'Filename': file.name,
                            'Detections': result['num_detections'],
                            'Top Class': top_detection['class_name'] if top_detection else 'None',
                            'Top Confidence': f"{top_detection['confidence']:.3f}" if top_detection else '0.000',
                            'Image Size': f"{result['image_size']['width']}x{result['image_size']['height']}"
                        })
                        
                    except Exception as e:
                        batch_results.append({
                            'Filename': file.name,
                            'Detections': 0,
                            'Top Class': 'Error',
                            'Top Confidence': '0.000',
                            'Image Size': 'N/A',
                            'Error': str(e)
                        })
                
                # Display batch results
                st.success(f"✅ Processed {len(batch_results)} images")
                st.info(f"Total detections found: {total_detections}")
                
                # Results table
                batch_df = pd.DataFrame(batch_results)
                st.dataframe(batch_df, use_container_width=True)
                
                # Summary statistics
                if len(batch_results) > 1:
                    st.write("**Batch Summary:**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Images", len(batch_results))
                    
                    with col2:
                        st.metric("Total Detections", total_detections)
                    
                    with col3:
                        avg_detections = batch_df['Detections'].mean()
                        st.metric("Avg Detections/Image", f"{avg_detections:.1f}")
                    
                    with col4:
                        images_with_detections = sum(1 for x in batch_results if x['Detections'] > 0)
                        st.metric("Images with Objects", f"{images_with_detections}/{len(batch_results)}")
                
                # Class distribution chart
                if total_detections > 0:
                    st.write("**Class Distribution:**")
                    class_counts = {}
                    for result in batch_results:
                        if result['Top Class'] != 'None' and result['Top Class'] != 'Error':
                            class_counts[result['Top Class']] = class_counts.get(result['Top Class'], 0) + 1
                    
                    if class_counts:
                        class_df = pd.DataFrame([
                            {'Class': k, 'Count': v} for k, v in class_counts.items()
                        ])
                        fig = px.pie(class_df, values='Count', names='Class', title="Top Class Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="batch_detections.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
