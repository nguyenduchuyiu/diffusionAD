#!/usr/bin/env python3
"""
Streamlit demo app for image classification
"""

import streamlit as st
import cv2
import numpy as np
import sys
import os
from PIL import Image
import plotly.express as px
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import ImageClassificationInference

@st.cache_resource
def load_model(model_path):
    """Load model with caching"""
    if not os.path.exists(model_path):
        return None
    try:
        return ImageClassificationInference(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Image Classification Demo",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ Image Classification Demo")
    st.write("Upload an image to classify it using the trained model")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="best_model.pth",
        help="Path to the trained model file"
    )
    
    # Load model
    inference_engine = load_model(model_path)
    
    if inference_engine is None:
        st.error(f"Could not load model from: {model_path}")
        st.info("Please make sure the model file exists and is valid")
        return
    
    # Display model info
    st.sidebar.success("✅ Model loaded successfully")
    st.sidebar.write(f"**Classes:** {len(inference_engine.class_names)}")
    st.sidebar.write(f"**Device:** {inference_engine.device}")
    
    # Class names
    with st.sidebar.expander("Class Names"):
        for i, class_name in enumerate(inference_engine.class_names):
            st.write(f"{i}: {class_name}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image file for classification"
        )
        
        # Input size slider
        input_size = st.slider(
            "Input Size", 
            min_value=128, 
            max_value=512, 
            value=224, 
            step=32,
            help="Input image size for the model"
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
        st.header("Prediction Results")
        
        if uploaded_file is not None:
            # Predict button
            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("Classifying..."):
                    try:
                        # Predict
                        result = inference_engine.predict(image_np, input_size=input_size)
                        
                        # Display results
                        st.success("✅ Classification Complete!")
                        
                        # Main prediction
                        st.metric(
                            label="Predicted Class",
                            value=result['predicted_class'],
                            delta=f"{result['confidence']:.3f} confidence"
                        )
                        
                        # Confidence bar
                        st.write("**Confidence:**")
                        st.progress(result['confidence'])
                        st.write(f"{result['confidence']:.1%}")
                        
                        # All class probabilities
                        st.write("**All Class Probabilities:**")
                        
                        # Create dataframe for plotting
                        prob_df = pd.DataFrame([
                            {"Class": class_name, "Probability": prob}
                            for class_name, prob in result['probabilities'].items()
                        ])
                        prob_df = prob_df.sort_values('Probability', ascending=True)
                        
                        # Horizontal bar chart
                        fig = px.bar(
                            prob_df, 
                            x='Probability', 
                            y='Class',
                            orientation='h',
                            title="Class Probabilities",
                            color='Probability',
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed probabilities table
                        with st.expander("Detailed Probabilities"):
                            prob_df_display = prob_df.sort_values('Probability', ascending=False)
                            prob_df_display['Probability'] = prob_df_display['Probability'].apply(lambda x: f"{x:.4f}")
                            st.dataframe(prob_df_display, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
        else:
            st.info("👆 Upload an image to see predictions")
    
    # Additional features
    st.header("📊 Batch Processing")
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Upload multiple images for batch processing",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,
        help="Upload multiple images to process them all at once"
    )
    
    if uploaded_files and len(uploaded_files) > 0:
        if st.button("🔄 Process Batch", type="secondary"):
            with st.spinner(f"Processing {len(uploaded_files)} images..."):
                batch_results = []
                
                # Process each image
                for i, file in enumerate(uploaded_files):
                    try:
                        image = Image.open(file)
                        image_np = np.array(image)
                        
                        # Handle different image formats
                        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                            image_np = image_np[:, :, :3]
                        
                        result = inference_engine.predict(image_np, input_size=input_size)
                        
                        batch_results.append({
                            'Filename': file.name,
                            'Predicted Class': result['predicted_class'],
                            'Confidence': result['confidence']
                        })
                        
                    except Exception as e:
                        batch_results.append({
                            'Filename': file.name,
                            'Predicted Class': 'Error',
                            'Confidence': 0.0,
                            'Error': str(e)
                        })
                
                # Display batch results
                st.success(f"✅ Processed {len(batch_results)} images")
                
                # Results table
                batch_df = pd.DataFrame(batch_results)
                st.dataframe(batch_df, use_container_width=True)
                
                # Summary statistics
                if len(batch_results) > 1:
                    st.write("**Batch Summary:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Images", len(batch_results))
                    
                    with col2:
                        avg_confidence = batch_df['Confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                    
                    with col3:
                        most_common = batch_df['Predicted Class'].mode().iloc[0] if len(batch_df) > 0 else "N/A"
                        st.metric("Most Common", most_common)
                
                # Download results
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
