#!/usr/bin/env python3
"""
Model utilities for YOLOv8 object detection
Note: Most model functionality is handled by ultralytics YOLO class
"""

from ultralytics import YOLO
import os

def load_yolo_model(model_path):
    """
    Load YOLO model from path or download pretrained
    
    Args:
        model_path: Path to YOLO model file (.pt)
    
    Returns:
        YOLO: Loaded YOLO model
    """
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found, downloading default YOLOv8n...")
        model_path = 'yolov8n.pt'  # This will auto-download
    
    model = YOLO(model_path)
    return model

def get_model_info(model):
    """
    Get information about YOLO model
    
    Args:
        model: YOLO model instance
    
    Returns:
        dict: Model information
    """
    info = {
        'model_type': 'YOLOv8',
        'task': 'detection',
        'classes': model.names if hasattr(model, 'names') else {},
        'num_classes': len(model.names) if hasattr(model, 'names') else 0
    }
    return info