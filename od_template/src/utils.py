#!/usr/bin/env python3
"""
Utility functions for YOLOv8 object detection
"""

import cv2
import numpy as np
import os
from pathlib import Path
import yaml
import shutil

def load_image(image_path):
    """Load image from path"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def create_yolo_dataset_yaml(data_dir, class_names=None):
    """
    Create YOLO dataset YAML configuration
    
    Args:
        data_dir: Path to dataset directory
        class_names: List of class names (optional, will auto-detect)
    
    Returns:
        str: Path to created YAML file
    """
    data_path = Path(data_dir)
    
    # Auto-detect classes if not provided
    if class_names is None:
        # Try to detect from label files
        label_dir = data_path / 'labels' / 'train'
        if label_dir.exists():
            classes = set()
            for label_file in label_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            classes.add(class_id)
            
            if classes:
                max_class = max(classes)
                class_names = [f'class_{i}' for i in range(max_class + 1)]
            else:
                class_names = ['defect']  # Default for defect detection
        else:
            class_names = ['defect']  # Default
    
    # Create dataset YAML
    dataset_yaml = {
        'path': str(data_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    # Save YAML file
    yaml_path = data_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    return str(yaml_path)

def check_dataset_structure(data_dir):
    """
    Check if dataset has proper YOLO structure
    
    Expected structure:
    data/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    
    Args:
        data_dir: Path to dataset directory
    
    Returns:
        tuple: (is_valid, missing_dirs)
    """
    data_path = Path(data_dir)
    
    required_dirs = [
        data_path / 'images' / 'train',
        data_path / 'images' / 'val',
        data_path / 'labels' / 'train',
        data_path / 'labels' / 'val'
    ]
    
    missing_dirs = [d for d in required_dirs if not d.exists()]
    is_valid = len(missing_dirs) == 0
    
    return is_valid, missing_dirs

def convert_bbox_format(bbox, from_format='xyxy', to_format='yolo', img_width=None, img_height=None):
    """
    Convert bounding box between different formats
    
    Args:
        bbox: Bounding box coordinates
        from_format: Source format ('xyxy', 'xywh', 'yolo')
        to_format: Target format ('xyxy', 'xywh', 'yolo')
        img_width: Image width (required for yolo format)
        img_height: Image height (required for yolo format)
    
    Returns:
        list: Converted bounding box coordinates
    """
    if from_format == to_format:
        return bbox
    
    # Convert to xyxy first
    if from_format == 'xywh':
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
    elif from_format == 'yolo':
        if img_width is None or img_height is None:
            raise ValueError("Image dimensions required for YOLO format conversion")
        center_x, center_y, w, h = bbox
        center_x *= img_width
        center_y *= img_height
        w *= img_width
        h *= img_height
        x1 = center_x - w / 2
        y1 = center_y - h / 2
        x2 = center_x + w / 2
        y2 = center_y + h / 2
    else:  # from_format == 'xyxy'
        x1, y1, x2, y2 = bbox
    
    # Convert to target format
    if to_format == 'xyxy':
        return [x1, y1, x2, y2]
    elif to_format == 'xywh':
        return [x1, y1, x2 - x1, y2 - y1]
    elif to_format == 'yolo':
        if img_width is None or img_height is None:
            raise ValueError("Image dimensions required for YOLO format conversion")
        center_x = (x1 + x2) / 2 / img_width
        center_y = (y1 + y2) / 2 / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height
        return [center_x, center_y, w, h]

def get_device():
    """Get best available device"""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    except ImportError:
        # If torch is not available, return 'cpu' and let ultralytics handle device detection
        return 'cpu'

def create_sample_dataset(output_dir, num_images=10):
    """
    Create a sample dataset for testing (downloads COCO sample if needed)
    
    Args:
        output_dir: Directory to create sample dataset
        num_images: Number of sample images to create
    """
    output_path = Path(output_dir)
    
    # Create directory structure
    dirs_to_create = [
        output_path / 'images' / 'train',
        output_path / 'images' / 'val',
        output_path / 'labels' / 'train',
        output_path / 'labels' / 'val'
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataset YAML
    create_yolo_dataset_yaml(output_dir, ['defect'])
    
    print(f"Sample dataset structure created at: {output_path}")
    print("Add your images and labels to the appropriate directories:")
    print("- Images: images/train/ and images/val/")
    print("- Labels: labels/train/ and labels/val/")
    print("- Labels should be in YOLO format: class_id center_x center_y width height (normalized 0-1)")

def validate_yolo_labels(label_dir, verbose=False):
    """
    Validate YOLO format label files
    
    Args:
        label_dir: Directory containing label files
        verbose: Print detailed validation info
    
    Returns:
        tuple: (is_valid, errors)
    """
    label_path = Path(label_dir)
    errors = []
    
    if not label_path.exists():
        return False, [f"Label directory does not exist: {label_path}"]
    
    label_files = list(label_path.glob('*.txt'))
    if not label_files:
        return False, ["No label files found"]
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label_file.name}:{line_num} - Expected 5 values, got {len(parts)}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    
                    # Validate ranges
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        errors.append(f"{label_file.name}:{line_num} - Coordinates out of range [0,1]")
                    
                    if class_id < 0:
                        errors.append(f"{label_file.name}:{line_num} - Negative class ID")
                        
                except ValueError as e:
                    errors.append(f"{label_file.name}:{line_num} - Invalid number format: {e}")
                    
        except Exception as e:
            errors.append(f"{label_file.name} - Error reading file: {e}")
    
    if verbose:
        print(f"Validated {len(label_files)} label files")
        if errors:
            print(f"Found {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
    
    return len(errors) == 0, errors