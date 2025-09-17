#!/usr/bin/env python3
"""
Inference utilities for image classification
"""

import torch
import cv2
import numpy as np
from models import load_trained_model
from utils import get_transforms, load_image

def predict_image(model, image, class_names, device='cpu', input_size=224):
    """
    Predict single image - core inference function
    
    Args:
        model: Trained model
        image: Image as numpy array or path
        class_names: List of class names
        device: Device to run inference
        input_size: Input image size
    
    Returns:
        dict: Prediction results
    """
    model.eval()
    
    # Load image if path provided
    if isinstance(image, str):
        image = load_image(image)
    
    # Preprocess - convert numpy array to PIL Image first
    from PIL import Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    transform = get_transforms(input_size, augment=False)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Format results
    result = {
        'predicted_class': class_names[predicted.item()],
        'confidence': float(confidence.item()),
        'probabilities': {
            class_names[i]: float(prob)
            for i, prob in enumerate(probabilities[0])
        }
    }
    
    return result

class ImageClassificationInference:
    """Image classification inference wrapper"""
    
    def __init__(self, model_path, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model, self.class_names = load_trained_model(model_path, self.device)
        self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
        print(f"Classes: {self.class_names}")
    
    def predict(self, image, input_size=224):
        """Predict single image"""
        return predict_image(
            self.model, 
            image, 
            self.class_names, 
            self.device, 
            input_size
        )
    
    def predict_batch(self, images, input_size=224, batch_size=32):
        """Predict batch of images"""
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_results = []
            
            for img in batch_images:
                result = self.predict(img, input_size)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
