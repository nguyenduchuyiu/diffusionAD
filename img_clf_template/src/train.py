#!/usr/bin/env python3
"""
Training script for image classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os

from models import get_model, freeze_backbone
from utils import prepare_data_loaders, save_checkpoint, get_device, count_parameters

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{running_loss/(pbar.n+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

def main():
    # Load configuration from file
    config_path = 'config.yaml'  # or 'config.json'
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'data_dir': 'data',
            'model': 'resnet18',
            'epochs': 20,
            'batch_size': 32,
            'lr': 0.001,
            'input_size': 224,
            'save_path': 'best_model.pth'
        }
    
    # Create args object from config
    class Args:
        def __init__(self, config):
            for key, value in config.items():
                setattr(self, key, value)
    
    args = Args(config)
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, class_names, num_classes = prepare_data_loaders(
        args.data_dir, args.batch_size, args.input_size
    )
    
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Val samples: {len(val_loader.dataset)}")
    
    # Model
    model = get_model(model_name=args.model, num_classes=num_classes)
    
    model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Training setup
    #TODO: OPTIMIZER AND SCHEDULER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            current_acc = val_acc
        else:
            val_loss, val_acc = 0, 0
            current_acc = train_acc
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        if val_loader:
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if current_acc > best_acc:
            best_acc = current_acc
            save_checkpoint(
                model, optimizer, epoch, train_loss, current_acc, 
                args.save_path, args.model, class_names, num_classes
            )
            print(f'Best model saved with accuracy: {current_acc:.2f}%')
    
    print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()
