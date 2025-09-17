#!/usr/bin/env python3
"""
Training script for YOLOv8 object detection
"""

import os
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
from utils import get_device

def prepare_yolo_dataset(data_dir):
    """
    Prepare YOLO dataset configuration
    Supports multiple dataset structures:
    1. Standard YOLO: data/{images,labels}/{train,val}/
    2. Alternative: data/{train,valid,test}/{images,labels}/
    3. Existing data.yaml file
    """
    data_path = Path(data_dir)
    
    # Check if data.yaml already exists
    existing_yaml = data_path / 'data.yaml'
    if existing_yaml.exists():
        print(f"Found existing data.yaml: {existing_yaml}")
        # Validate the YAML file
        try:
            with open(existing_yaml, 'r') as f:
                yaml_content = yaml.safe_load(f)
            
            # Check if all required paths exist
            if 'path' in yaml_content and 'train' in yaml_content and 'val' in yaml_content:
                yaml_path = Path(yaml_content['path']) if 'path' in yaml_content else data_path
                train_path = yaml_path / yaml_content['train']
                val_path = yaml_path / yaml_content['val']
                
                if train_path.exists() and val_path.exists():
                    print(f"Dataset structure validated: {yaml_content['nc']} classes")
                    print(f"Classes: {yaml_content.get('names', [])}")
                    return str(existing_yaml)
                else:
                    print(f"Warning: Paths in data.yaml don't exist")
                    print(f"Train path: {train_path} (exists: {train_path.exists()})")
                    print(f"Val path: {val_path} (exists: {val_path.exists()})")
        except Exception as e:
            print(f"Error reading existing data.yaml: {e}")
    
    # Check for alternative structure: train/, valid/, test/ directories
    alt_structure_dirs = [
        data_path / 'train' / 'images',
        data_path / 'train' / 'labels',
        data_path / 'valid' / 'images',
        data_path / 'valid' / 'labels'
    ]
    
    if all(d.exists() for d in alt_structure_dirs):
        print("Found alternative dataset structure: train/, valid/, test/")
        
        # Auto-detect classes from label files
        classes = set()
        label_files = list((data_path / 'train' / 'labels').glob('*.txt'))
        class_names = []
        
        if label_files:
            for label_file in label_files[:20]:  # Check first 20 files
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.strip().split()[0])
                                classes.add(class_id)
                except Exception:
                    continue
            
            if classes:
                max_class = max(classes)
                class_names = [f'class_{i}' for i in range(max_class + 1)]
        
        # Create dataset YAML
        dataset_yaml = {
            'path': str(data_path.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images' if (data_path / 'test' / 'images').exists() else 'valid/images',
            'nc': len(classes) if classes else 1,
            'names': class_names if class_names else ['object']
        }
        
        # Save dataset YAML
        yaml_path = data_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        print(f"Auto-detected {dataset_yaml['nc']} classes: {dataset_yaml['names']}")
        print(f"Dataset configuration saved to: {yaml_path}")
        return str(yaml_path)
    
    # Check for standard YOLO structure
    standard_dirs = [
        data_path / 'images' / 'train',
        data_path / 'images' / 'val',
        data_path / 'labels' / 'train',
        data_path / 'labels' / 'val'
    ]
    
    missing_dirs = [d for d in standard_dirs if not d.exists()]
    if len(missing_dirs) == 0:
        print("Found standard YOLO dataset structure")
        
        # Auto-detect classes
        classes = set()
        label_files = list((data_path / 'labels' / 'train').glob('*.txt'))
        class_names = []
        
        if label_files:
            for label_file in label_files[:20]:
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.strip().split()[0])
                                classes.add(class_id)
                except Exception:
                    continue
            
            if classes:
                max_class = max(classes)
                class_names = [f'class_{i}' for i in range(max_class + 1)]
        
        # Create dataset YAML
        dataset_yaml = {
            'path': str(data_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/val',
            'nc': len(classes) if classes else 1,
            'names': class_names if class_names else ['object']
        }
        
        # Save dataset YAML
        yaml_path = data_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        print(f"Auto-detected {dataset_yaml['nc']} classes: {dataset_yaml['names']}")
        print(f"Dataset configuration saved to: {yaml_path}")
        return str(yaml_path)
    
    # No valid structure found
    print("Error: No valid YOLO dataset structure found!")
    print("\nSupported structures:")
    print("1. Standard YOLO:")
    print("   data/")
    print("   ├── images/")
    print("   │   ├── train/")
    print("   │   └── val/")
    print("   └── labels/")
    print("       ├── train/")
    print("       └── val/")
    print("\n2. Alternative:")
    print("   data/")
    print("   ├── train/")
    print("   │   ├── images/")
    print("   │   └── labels/")
    print("   └── valid/")
    print("       ├── images/")
    print("       └── labels/")
    print("\n3. Existing data.yaml file")
    
    return None

def main():
    # Load configuration from file
    config_path = 'config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration for object detection
        config = {
            'data_dir': 'data',
            'model': 'yolov8n.pt',
            'epochs': 50,
            'batch_size': 16,
            'lr': 0.01,
            'input_size': 640,
            'save_path': 'best_yolo.pt',
            'confidence': 0.25,
            'iou': 0.45
        }
    
    print("YOLOv8 Object Detection Training")
    print("=" * 50)
    print(f"Model: {config['model']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Input size: {config['input_size']}")
    print(f"Data directory: {config['data_dir']}")
    print("=" * 50)
    
    # Prepare dataset
    dataset_yaml = prepare_yolo_dataset(config['data_dir'])
    if dataset_yaml is None:
        print("Error: Could not prepare dataset. Please check your data structure.")
        return
    
    device = get_device()
    print(f"Using device: {device}")
    
    try:
        # Load YOLOv8 model
        print(f"Loading YOLO model: {config['model']}")
        model = YOLO(config['model'])
        
        # Train the model
        print("Starting training...")
        results = model.train(
            data=dataset_yaml,
            epochs=config['epochs'],
            batch=config['batch_size'],
            lr0=config['lr'],
            imgsz=config['input_size'],
            name='yolo_training',
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            patience=20,     # Early stopping patience
            device=device,    # Will auto-detect GPU if available
            workers=4,
            exist_ok=True
        )
        
        # Save the best model
        best_model_path = model.trainer.best
        if best_model_path and os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, config['save_path'])
            print(f"Best model saved to: {config['save_path']}")
        else:
            print("Warning: Could not find best model file")
        
        # Print training results
        print("\nTraining completed!")
        print(f"Results saved in: runs/detect/yolo_training")
        
        # Try to get training metrics
        try:
            if hasattr(results, 'maps') and results.maps is not None:
                if len(results.maps) > 0:
                    print(f"Best mAP50: {max(results.maps):.4f}")
                else:
                    print("Best mAP50: N/A")
            else:
                print("Training metrics not available")
        except Exception as e:
            print(f"Could not retrieve training metrics: {e}")
        
        # Validation
        print("\nRunning validation...")
        try:
            val_results = model.val(data=dataset_yaml)
            if hasattr(val_results, 'box') and hasattr(val_results.box, 'map50'):
                print(f"Validation mAP50: {val_results.box.map50:.4f}")
            if hasattr(val_results, 'box') and hasattr(val_results.box, 'map'):
                print(f"Validation mAP50-95: {val_results.box.map:.4f}")
        except Exception as e:
            print(f"Validation error: {e}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Make sure your dataset is properly formatted for YOLO.")
        print("Each image should have a corresponding .txt file with annotations in YOLO format:")
        print("class_id center_x center_y width height (normalized 0-1)")

if __name__ == '__main__':
    main()
