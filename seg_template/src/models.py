#!/usr/bin/env python3
"""
Segmentation models for defect detection
Supports U-Net and DeepLabv3+ with various backbones
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from pathlib import Path

def get_model(model_name='unet', encoder='resnet18', num_classes=2, activation='sigmoid'):
    """
    Get segmentation model by name
    
    Args:
        model_name: Model architecture ('unet', 'deeplabv3plus', 'fpn', 'pspnet')
        encoder: Encoder backbone ('resnet18', 'resnet34', 'resnet50', etc.)
        num_classes: Number of output classes
        activation: Output activation ('sigmoid', 'softmax', None)
    
    Returns:
        torch.nn.Module: Segmentation model
    """
    # Map model names to smp models
    model_map = {
        'unet': smp.Unet,
        'deeplabv3plus': smp.DeepLabV3Plus,
        'fpn': smp.FPN,
        'pspnet': smp.PSPNet,
        'linknet': smp.Linknet,
        'pan': smp.PAN
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {list(model_map.keys())}")
    
    # Create model
    model_class = model_map[model_name]
    
    model = model_class(
        encoder_name=encoder,
        encoder_weights="imagenet",  # Use ImageNet pretrained weights
        in_channels=3,  # RGB images
        classes=num_classes,
        activation=activation
    )
    
    return model

def get_loss_function(loss_name='dice', num_classes=2):
    """
    Get loss function for segmentation
    
    Args:
        loss_name: Loss function name ('dice', 'jaccard', 'focal', 'bce', 'ce')
        num_classes: Number of classes
    
    Returns:
        Loss function
    """
    if loss_name == 'dice':
        return smp.losses.DiceLoss(mode='binary' if num_classes == 2 else 'multiclass')
    elif loss_name == 'jaccard':
        return smp.losses.JaccardLoss(mode='binary' if num_classes == 2 else 'multiclass')
    elif loss_name == 'focal':
        return smp.losses.FocalLoss(mode='binary' if num_classes == 2 else 'multiclass')
    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_name == 'combined':
        # Combine Dice + BCE for better training
        dice_loss = smp.losses.DiceLoss(mode='binary' if num_classes == 2 else 'multiclass')
        bce_loss = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
        return CombinedLoss(dice_loss, bce_loss)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

class CombinedLoss(nn.Module):
    """Combined loss function (Dice + BCE/CE)"""
    
    def __init__(self, loss1, loss2, weight1=0.5, weight2=0.5):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.weight1 = weight1
        self.weight2 = weight2
    
    def forward(self, pred, target):
        return self.weight1 * self.loss1(pred, target) + self.weight2 * self.loss2(pred, target)

def get_metrics(num_classes=2):
    """
    Get evaluation metrics for segmentation
    
    Args:
        num_classes: Number of classes
    
    Returns:
        List of metric functions
    """
    import torch
    
    class IoUMetric:
        def __init__(self, threshold=0.5):
            self.threshold = threshold
            
        def __call__(self, pred, target):
            pred = torch.sigmoid(pred) if pred.shape[1] == 1 else torch.softmax(pred, dim=1)
            if pred.shape[1] == 1:
                pred_binary = (pred > self.threshold).float()
                target_binary = (target > self.threshold).float()
            else:
                pred_binary = torch.argmax(pred, dim=1).float()
                target_binary = target.float()
            
            intersection = (pred_binary * target_binary).sum()
            union = pred_binary.sum() + target_binary.sum() - intersection
            
            if union == 0:
                return torch.tensor(1.0)
            return intersection / union
    
    class DiceMetric:
        def __init__(self, threshold=0.5):
            self.threshold = threshold
            
        def __call__(self, pred, target):
            pred = torch.sigmoid(pred) if pred.shape[1] == 1 else torch.softmax(pred, dim=1)
            if pred.shape[1] == 1:
                pred_binary = (pred > self.threshold).float()
                target_binary = (target > self.threshold).float()
            else:
                pred_binary = torch.argmax(pred, dim=1).float()
                target_binary = target.float()
            
            intersection = (pred_binary * target_binary).sum()
            total = pred_binary.sum() + target_binary.sum()
            
            if total == 0:
                return torch.tensor(1.0)
            return 2.0 * intersection / total
    
    class AccuracyMetric:
        def __init__(self, threshold=0.5):
            self.threshold = threshold
            
        def __call__(self, pred, target):
            pred = torch.sigmoid(pred) if pred.shape[1] == 1 else torch.softmax(pred, dim=1)
            if pred.shape[1] == 1:
                pred_binary = (pred > self.threshold).float()
                target_binary = (target > self.threshold).float()
            else:
                pred_binary = torch.argmax(pred, dim=1).float()
                target_binary = target.float()
            
            correct = (pred_binary == target_binary).float().sum()
            total = target_binary.numel()
            
            return correct / total
    
    metrics = [
        IoUMetric(threshold=0.5),
        DiceMetric(threshold=0.5),
        AccuracyMetric(threshold=0.5),
    ]
    
    # Add class names for easier identification
    metrics[0].__class__.__name__ = "IoU"
    metrics[1].__class__.__name__ = "Fscore"  # Dice is also called F-score
    metrics[2].__class__.__name__ = "Accuracy"
    
    return metrics

def save_checkpoint(model, optimizer, epoch, loss, metrics_dict, save_path, 
                   model_name, encoder, num_classes, activation):
    """Save model checkpoint with metadata"""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics_dict,
        'model_name': model_name,
        'encoder': encoder,
        'num_classes': num_classes,
        'activation': activation,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")

def load_trained_model(model_path, device='cpu'):
    """
    Load trained segmentation model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        tuple: (model, metadata)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model info
    model_name = checkpoint.get('model_name', 'unet')
    encoder = checkpoint.get('encoder', 'resnet18')
    num_classes = checkpoint.get('num_classes', 2)
    activation = checkpoint.get('activation', 'sigmoid')
    
    # Create model
    model = get_model(
        model_name=model_name, 
        encoder=encoder, 
        num_classes=num_classes,
        activation=activation
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    metadata = {
        'model_name': model_name,
        'encoder': encoder,
        'num_classes': num_classes,
        'activation': activation,
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0),
        'metrics': checkpoint.get('metrics', {})
    }
    
    return model, metadata

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_info(model_name, encoder):
    """Get information about model architecture"""
    
    # Approximate parameter counts (in millions)
    param_counts = {
        'unet': {
            'resnet18': 14.3,
            'resnet34': 24.4,
            'resnet50': 35.7,
        },
        'deeplabv3plus': {
            'resnet18': 15.8,
            'resnet34': 25.9,
            'resnet50': 37.2,
        }
    }
    
    info = {
        'architecture': model_name,
        'encoder': encoder,
        'approx_params_m': param_counts.get(model_name, {}).get(encoder, 'Unknown'),
        'pretrained': 'ImageNet',
        'suitable_for': 'PCB defect detection, surface defect detection'
    }
    
    return info

# Preprocessing functions
def get_preprocessing_fn(encoder_name):
    """Get preprocessing function for encoder"""
    return smp.encoders.get_preprocessing_fn(encoder_name, pretrained='imagenet')