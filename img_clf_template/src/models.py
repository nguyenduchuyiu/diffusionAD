#!/usr/bin/env python3
"""
Model definitions and loading utilities
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet34_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import EfficientNet_B0_Weights
from torchvision.models import MobileNet_V2_Weights
from pathlib import Path

def get_model(model_name, num_classes): #TODO: MODEL
    """Get model by name"""
    if model_name == 'resnet18':
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    freeze_backbone(model)
    
    return model

def load_trained_model(model_path, device='cpu'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Get model info from checkpoint
    model_name = checkpoint.get('model_name', 'resnet18')
    num_classes = checkpoint.get('num_classes', 2)
    class_names = checkpoint.get('class_names', [f'class_{i}' for i in range(num_classes)])
    
    # Create and load model
    model = get_model(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, class_names

def freeze_backbone(model):
    """Freeze backbone layers"""
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
