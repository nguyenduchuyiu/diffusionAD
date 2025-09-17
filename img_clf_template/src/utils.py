#!/usr/bin/env python3
"""
Utility functions
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
import os

def get_transforms(input_size=224, augment=True):
    """Get data transforms"""
    if augment:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def load_image(image_path):
    """Load image from path"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def prepare_data_loaders(data_dir, batch_size=32, input_size=224):
    """Prepare data loaders from directory structure"""
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    
    train_transform = get_transforms(input_size, augment=True)
    val_transform = get_transforms(input_size, augment=False)
    
    # Assume data structure: data_dir/{train,val}/class_name/images
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        # If no train/val split, use all data as train
        train_dataset = ImageFolder(data_dir, transform=train_transform)
        val_dataset = None
        val_loader = None
    else:
        train_dataset = ImageFolder(train_dir, transform=train_transform)
        val_dataset = ImageFolder(val_dir, transform=val_transform) if os.path.exists(val_dir) else None
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    return train_loader, val_loader, class_names, num_classes

def save_checkpoint(model, optimizer, epoch, loss, accuracy, save_path, 
                   model_name, class_names, num_classes):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'model_name': model_name,
        'class_names': class_names,
        'num_classes': num_classes
    }, save_path)

def get_device():
    """Get available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
