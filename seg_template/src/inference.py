#!/usr/bin/env python3
"""
Inference utilities for semantic segmentation
"""

import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt

from models import load_trained_model, get_preprocessing_fn

def to_numpy(tensor):
    """Safely convert torch tensor (GPU/CPU) to numpy array"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)

def predict_image(model, image, device='cpu', input_size=512, threshold=0.5, 
                 preprocessing_fn=None):
    """
    Predict segmentation mask for single image
    
    Args:
        model: Trained segmentation model
        image: Input image (numpy array or PIL Image)
        device: Device to run inference
        input_size: Input image size
        threshold: Threshold for binary segmentation
        preprocessing_fn: Preprocessing function for encoder
    
    Returns:
        dict: Prediction results with mask and overlay
    """
    model.eval()
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Store original image and size
    original_image = image.copy()
    original_size = (image.shape[1], image.shape[0])  # (width, height)
    
    # Prepare transforms
    transform = A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Apply transforms
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Apply preprocessing if provided
    if preprocessing_fn:
        # Move to CPU for preprocessing, then back to device
        input_cpu = input_tensor.cpu()
        try:
            input_preprocessed = preprocessing_fn(input_cpu)
            input_tensor = input_preprocessed.to(device)
        except Exception as e:
            # If preprocessing fails, skip it (already normalized by albumentations)
            print(f"Preprocessing failed, skipping: {e}")
            pass
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        
        # Handle different output formats
        if isinstance(output, dict):
            # Some models return dict
            prediction = output['out'] if 'out' in output else output
        else:
            prediction = output
        
        # Apply activation if needed (for models without built-in activation)
        if prediction.shape[1] == 1:  # Binary segmentation
            prediction = torch.sigmoid(prediction)
        else:  # Multi-class
            prediction = torch.softmax(prediction, dim=1)
    
    # Convert to numpy safely
    prediction = to_numpy(prediction.squeeze())
    
    # Resize back to original size
    if prediction.ndim == 3:  # Multi-class
        # Take argmax for multi-class
        prediction = np.argmax(prediction, axis=0)
        mask = cv2.resize(prediction.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
        binary_mask = mask
    else:  # Binary
        mask = cv2.resize(prediction, original_size, interpolation=cv2.INTER_LINEAR)
        # Apply threshold for binary mask
        binary_mask = (mask > threshold).astype(np.uint8) * 255
    
    # Calculate confidence/probability
    confidence = float(np.max(mask)) if hasattr(mask, 'shape') else 1.0
    
    # Create overlay
    overlay = create_overlay(original_image, binary_mask if 'binary_mask' in locals() else mask)
    
    result = {
        'mask': binary_mask if 'binary_mask' in locals() else mask,
        'confidence': confidence,
        'overlay': overlay,
        'original_size': original_size,
        'defect_area_ratio': calculate_defect_ratio(binary_mask if 'binary_mask' in locals() else mask)
    }
    
    return result

def create_overlay(image, mask, alpha=0.5, color=[255, 0, 0]):
    """Create overlay of mask on image"""
    
    # Convert tensors to numpy safely
    image = to_numpy(image)
    mask = to_numpy(mask)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Ensure mask is binary
    if mask.max() > 1:
        mask_binary = mask > 127
    else:
        mask_binary = mask > 0.5
    
    # Create colored mask
    overlay = image.copy()
    if np.any(mask_binary):  # Only apply overlay if there are positive pixels
        overlay[mask_binary] = (
            image[mask_binary] * (1 - alpha) + 
            np.array(color) * alpha
        ).astype(np.uint8)
    
    return overlay

def calculate_defect_ratio(mask):
    """Calculate ratio of defect area to total area"""
    # Convert tensor to numpy safely
    mask = to_numpy(mask)
    
    if mask.max() > 1:
        defect_pixels = np.sum(mask > 127)
    else:
        defect_pixels = np.sum(mask > 0.5)
    
    total_pixels = mask.shape[0] * mask.shape[1]
    return float(defect_pixels) / total_pixels

class SegmentationInference:
    """Segmentation inference wrapper"""
    
    def __init__(self, model_path, device='auto', threshold=0.5):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.threshold = threshold
        
        # Load model
        self.model, self.metadata = load_trained_model(model_path, self.device)
        self.model.to(self.device)
        
        # Get preprocessing function
        try:
            self.preprocessing_fn = get_preprocessing_fn(self.metadata['encoder'])
        except:
            self.preprocessing_fn = None
        
        print(f"Model loaded: {self.metadata['model_name']} with {self.metadata['encoder']}")
        print(f"Device: {self.device}")
        print(f"Classes: {self.metadata['num_classes']}")
        print(f"Input size: 512 (default)")
    
    def predict(self, image, input_size=512, threshold=None):
        """Predict single image"""
        thresh = threshold if threshold is not None else self.threshold
        
        return predict_image(
            self.model, 
            image, 
            self.device, 
            input_size, 
            thresh,
            self.preprocessing_fn
        )
    
    def predict_batch(self, images, input_size=512, threshold=None):
        """Predict batch of images"""
        results = []
        thresh = threshold if threshold is not None else self.threshold
        
        for image in images:
            result = self.predict(image, input_size, thresh)
            results.append(result)
        
        return results
    
    def visualize_prediction(self, image, save_path=None):
        """Visualize prediction with overlay"""
        result = self.predict(image)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Predicted mask
        mask_np = to_numpy(result['mask'])
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title(f'Predicted Mask (Conf: {result["confidence"]:.3f})')
        axes[1].axis('off')
        
        # Overlay
        overlay_np = to_numpy(result['overlay'])
        axes[2].imshow(overlay_np)
        axes[2].set_title(f'Overlay (Defect: {result["defect_area_ratio"]:.1%})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        return fig, result
    
    def save_results(self, image, result, output_dir, filename_base):
        """Save prediction results"""
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save mask
        mask_np = to_numpy(result['mask'])
        mask_path = output_path / f"{filename_base}_mask.png"
        cv2.imwrite(str(mask_path), mask_np)
        
        # Save overlay
        overlay_np = to_numpy(result['overlay'])
        overlay_path = output_path / f"{filename_base}_overlay.png"
        overlay_bgr = cv2.cvtColor(overlay_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(overlay_path), overlay_bgr)
        
        # Save original (for reference)
        original_path = output_path / f"{filename_base}_original.png"
        image_np = to_numpy(image)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(original_path), image_bgr)
        else:
            cv2.imwrite(str(original_path), image_np)
        
        print(f"Results saved to {output_path}")
        return {
            'mask_path': mask_path,
            'overlay_path': overlay_path,
            'original_path': original_path
        }

def load_image(image_path):
    """Load image from path"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def batch_inference(model_path, input_dir, output_dir, threshold=0.5):
    """Run batch inference on directory of images"""
    from pathlib import Path
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load inference engine
    inference_engine = SegmentationInference(model_path, threshold=threshold)
    
    # Process all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(ext)))
    
    print(f"Found {len(image_files)} images to process")
    
    results_summary = []
    
    for image_file in image_files:
        try:
            # Load image
            image = load_image(image_file)
            
            # Predict
            result = inference_engine.predict(image)
            
            # Save results
            filename_base = image_file.stem
            saved_paths = inference_engine.save_results(
                image, result, output_path, filename_base
            )
            
            # Add to summary
            results_summary.append({
                'filename': image_file.name,
                'defect_area_ratio': result['defect_area_ratio'],
                'confidence': result['confidence'],
                'has_defect': result['defect_area_ratio'] > 0.01,  # 1% threshold
                'mask_path': saved_paths['mask_path'],
                'overlay_path': saved_paths['overlay_path']
            })
            
            print(f"Processed {image_file.name}: "
                  f"Defect ratio: {result['defect_area_ratio']:.1%}, "
                  f"Confidence: {result['confidence']:.3f}")
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
    
    # Save summary
    import pandas as pd
    summary_df = pd.DataFrame(results_summary)
    summary_path = output_path / 'inference_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nBatch inference completed!")
    print(f"Processed {len(results_summary)} images")
    print(f"Results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    
    return results_summary