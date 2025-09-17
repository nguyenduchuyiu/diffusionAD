#!/usr/bin/env python3
"""
Utility functions for segmentation
"""

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
from PIL import Image
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):
    """Dataset for segmentation tasks"""
    
    def __init__(self, images_dir, masks_dir, transform=None, preprocessing=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(list(self.images_dir.glob(ext)))
        
        self.transform = transform
        self.preprocessing = preprocessing
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = image_path.stem + '.png'  # Assume masks are PNG
        mask_path = self.masks_dir / mask_name
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Create dummy mask if not found
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            print(f"Warning: Mask not found for {image_path.name}, using zero mask")
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            preprocessed = self.preprocessing(image=image, mask=mask)
            image = preprocessed['image']
            mask = preprocessed['mask']
        
        # Convert mask to proper format
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        
        # Normalize mask to 0-1 if needed
        if mask.max() > 1:
            mask = mask / 255.0
        
        return image, mask

def get_transforms(input_size=512, augment=True):
    """Get augmentation transforms for training/validation"""
    
    if augment:
        # Training transforms with augmentation
        transform = A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:
        # Validation transforms (no augmentation)
        transform = A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    return transform

def get_preprocessing(encoder_name):
    """Get preprocessing function for encoder"""
    # For simplicity, we'll use standard ImageNet normalization
    # instead of encoder-specific preprocessing to avoid shape issues
    return None  # Will use standard normalization in transforms

def prepare_data_loaders(data_dir, batch_size=8, input_size=512, encoder_name='resnet18'):
    """Prepare data loaders for segmentation"""
    
    data_path = Path(data_dir)
    
    # Check for different data structures
    train_images = None
    train_masks = None
    val_images = None
    val_masks = None
    
    # Structure 1: data/{train,val}/{images,masks}/
    if (data_path / 'train' / 'images').exists():
        train_images = data_path / 'train' / 'images'
        train_masks = data_path / 'train' / 'masks'
        val_images = data_path / 'val' / 'images'
        val_masks = data_path / 'val' / 'masks'
    
    # Structure 2: data/{images,masks}/ (split manually)
    elif (data_path / 'images').exists() and (data_path / 'masks').exists():
        # Use all data for training (you might want to implement train/val split)
        train_images = data_path / 'images'
        train_masks = data_path / 'masks'
        val_images = data_path / 'images'  # Same as train for now
        val_masks = data_path / 'masks'
        print("Warning: Using same data for train and validation. Consider splitting your data.")
    
    else:
        raise ValueError(f"No valid data structure found in {data_dir}")
    
    # Get transforms
    train_transform = get_transforms(input_size, augment=True)
    val_transform = get_transforms(input_size, augment=False)
    
    # Create datasets (no preprocessing for now)
    train_dataset = SegmentationDataset(
        train_images, 
        train_masks, 
        transform=train_transform,
        preprocessing=None
    )
    
    val_dataset = None
    if val_images.exists() and val_masks.exists():
        val_dataset = SegmentationDataset(
            val_images, 
            val_masks, 
            transform=val_transform,
            preprocessing=None
        )
    
    # Create data loaders (use num_workers=0 to avoid multiprocessing issues with lambda)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True
        )
    
    return train_loader, val_loader

def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU (Intersection over Union)"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()

def calculate_dice(pred, target, threshold=0.5):
    """Calculate Dice coefficient"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    total = pred_binary.sum() + target_binary.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection / total).item()

def visualize_prediction(image, mask, prediction, save_path=None):
    """Visualize image, ground truth mask, and prediction"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if isinstance(image, torch.Tensor):
        # Denormalize if needed
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        image = np.clip(image, 0, 1)
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().squeeze()
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy().squeeze()
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def create_overlay(image, mask, alpha=0.5, color=[255, 0, 0]):
    """Create overlay of mask on image"""
    
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().squeeze()
    
    # Convert mask to 0-255
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = color[0] * (mask > 127) / 255
    colored_mask[:, :, 1] = color[1] * (mask > 127) / 255
    colored_mask[:, :, 2] = color[2] * (mask > 127) / 255
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    
    return overlay

def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def create_sample_dataset(output_dir, num_samples=10, image_size=512):
    """Create sample segmentation dataset for testing"""
    
    output_path = Path(output_dir)
    
    # Create directory structure
    dirs_to_create = [
        output_path / 'train' / 'images',
        output_path / 'train' / 'masks',
        output_path / 'val' / 'images',
        output_path / 'val' / 'masks'
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Sample segmentation dataset structure created at: {output_path}")
    print("Expected structure:")
    print("data/")
    print("├── train/")
    print("│   ├── images/")
    print("│   └── masks/")
    print("└── val/")
    print("    ├── images/")
    print("    └── masks/")
    print("\nMasks should be:")
    print("- Grayscale images (0-255)")
    print("- Same filename as corresponding image")
    print("- 0 = background, 255 = defect/object")

def load_image(image_path):
    """Load image from path"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image