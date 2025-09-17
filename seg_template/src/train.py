#!/usr/bin/env python3
"""
Training script for semantic segmentation
Supports U-Net, DeepLabv3+ with various backbones
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import os
import yaml
import numpy as np
from pathlib import Path

from models import get_model, get_loss_function, get_metrics, save_checkpoint
from utils import prepare_data_loaders, get_device, calculate_iou, calculate_dice, visualize_prediction

def train_epoch(model, train_loader, criterion, optimizer, device, metrics):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_metrics = {metric.__class__.__name__: 0.0 for metric in metrics}
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Ensure mask has correct shape and type for binary segmentation
        if len(masks.shape) == 4 and masks.shape[1] == 1:  # [B, 1, H, W]
            masks = masks.squeeze(1)  # [B, H, W] for BCE with logits
        elif len(masks.shape) == 3:  # [B, H, W] - already correct for BCE
            pass
        
        # For binary segmentation with sigmoid activation
        if outputs.shape[1] == 1:
            # BCE expects target same shape as input
            if len(masks.shape) == 3:  # [B, H, W]
                masks = masks.unsqueeze(1)  # [B, 1, H, W]
            masks = masks.float()  # Ensure float type for BCEWithLogitsLoss
        else:
            masks = masks.long()   # Ensure long type for CrossEntropyLoss
        
        # Calculate loss
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate metrics
        with torch.no_grad():
            for metric in metrics:
                metric_value = metric(outputs, masks.long() if masks.dtype != torch.float else masks)
                running_metrics[metric.__class__.__name__] += metric_value.item()
        
        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'IoU': f'{running_metrics["IoU"] / (batch_idx + 1):.4f}',
            'Dice': f'{running_metrics["Fscore"] / (batch_idx + 1):.4f}'
        })
    
    # Calculate average metrics
    avg_loss = running_loss / len(train_loader)
    avg_metrics = {k: v / len(train_loader) for k, v in running_metrics.items()}
    
    return avg_loss, avg_metrics

def validate(model, val_loader, criterion, device, metrics):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    running_metrics = {metric.__class__.__name__: 0.0 for metric in metrics}
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Ensure mask has correct shape and type for binary segmentation
            if len(masks.shape) == 4 and masks.shape[1] == 1:  # [B, 1, H, W]
                masks = masks.squeeze(1)  # [B, H, W] for BCE with logits
            elif len(masks.shape) == 3:  # [B, H, W] - already correct for BCE
                pass
            
            # For binary segmentation with sigmoid activation
            if outputs.shape[1] == 1:
                # BCE expects target same shape as input
                if len(masks.shape) == 3:  # [B, H, W]
                    masks = masks.unsqueeze(1)  # [B, 1, H, W]
                masks = masks.float()  # Ensure float type for BCEWithLogitsLoss
            else:
                masks = masks.long()   # Ensure long type for CrossEntropyLoss
            
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            # Calculate metrics
            for metric in metrics:
                metric_value = metric(outputs, masks.long() if masks.dtype != torch.float else masks)
                running_metrics[metric.__class__.__name__] += metric_value.item()
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'IoU': f'{running_metrics["IoU"] / (batch_idx + 1):.4f}',
                'Dice': f'{running_metrics["Fscore"] / (batch_idx + 1):.4f}'
            })
    
    avg_loss = running_loss / len(val_loader)
    avg_metrics = {k: v / len(val_loader) for k, v in running_metrics.items()}
    
    return avg_loss, avg_metrics

def main():
    # Load configuration
    config_path = 'config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'data_dir': 'data',
            'model': 'unet',
            'encoder': 'resnet18',
            'epochs': 50,
            'batch_size': 8,
            'lr': 0.001,
            'input_size': 512,
            'save_path': 'best_segmentation.pth',
            'num_classes': 2,
            'activation': 'sigmoid'
        }
    
    print("Semantic Segmentation Training")
    print("=" * 50)
    print(f"Model: {config['model']}")
    print(f"Encoder: {config['encoder']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Input size: {config['input_size']}")
    print(f"Number of classes: {config['num_classes']}")
    print(f"Activation: {config['activation']}")
    print("=" * 50)
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Prepare data loaders
    try:
        train_loader, val_loader = prepare_data_loaders(
            config['data_dir'], 
            batch_size=config['batch_size'],
            input_size=config['input_size'],
            encoder_name=config['encoder']
        )
        
        print(f"Train samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Val samples: {len(val_loader.dataset)}")
        else:
            print("No validation data found")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure your data follows the expected structure:")
        print("data/")
        print("├── train/")
        print("│   ├── images/")
        print("│   └── masks/")
        print("└── val/")
        print("    ├── images/")
        print("    └── masks/")
        return
    
    # Create model
    try:
        model = get_model(
            model_name=config['model'],
            encoder=config['encoder'],
            num_classes=config['num_classes'],
            activation=config['activation']
        )
        model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Setup loss function and metrics (use simple BCE for now)
    criterion = get_loss_function('bce', config['num_classes'])  # Use simple BCE loss
    metrics = get_metrics(config['num_classes'])
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, verbose=True
    )
    
    # Training loop
    best_iou = 0.0
    best_dice = 0.0
    
    print("\nStarting training...")
    
    for epoch in range(config['epochs']):
        print(f'\nEpoch {epoch+1}/{config["epochs"]}')
        print('-' * 50)
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, metrics
        )
        
        # Validate
        if val_loader:
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, metrics
            )
            current_iou = val_metrics['IoU']
            current_dice = val_metrics['Fscore']
            current_loss = val_loss
        else:
            val_loss, val_metrics = 0, {}
            current_iou = train_metrics['IoU']
            current_dice = train_metrics['Fscore']
            current_loss = train_loss
        
        # Update scheduler
        scheduler.step(current_loss)
        
        # Print metrics
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Train IoU: {train_metrics["IoU"]:.4f}')
        print(f'Train Dice: {train_metrics["Fscore"]:.4f}')
        
        if val_loader:
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Val IoU: {val_metrics["IoU"]:.4f}')
            print(f'Val Dice: {val_metrics["Fscore"]:.4f}')
        
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model based on IoU
        if current_iou > best_iou:
            best_iou = current_iou
            best_dice = current_dice
            
            # Prepare metrics for saving
            metrics_dict = {
                'train_loss': train_loss,
                'train_iou': train_metrics['IoU'],
                'train_dice': train_metrics['Fscore'],
                'val_loss': val_loss if val_loader else 0,
                'val_iou': val_metrics.get('IoU', 0),
                'val_dice': val_metrics.get('Fscore', 0),
            }
            
            save_checkpoint(
                model, optimizer, epoch, current_loss, metrics_dict,
                config['save_path'], config['model'], config['encoder'],
                config['num_classes'], config['activation']
            )
            
            print(f'Best model saved! IoU: {current_iou:.4f}, Dice: {current_dice:.4f}')
    
    print(f'\nTraining completed!')
    print(f'Best IoU: {best_iou:.4f}')
    print(f'Best Dice: {best_dice:.4f}')
    print(f'Model saved to: {config["save_path"]}')

if __name__ == '__main__':
    main()