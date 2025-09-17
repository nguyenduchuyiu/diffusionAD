#!/usr/bin/env python3
"""
Inference utilities for YOLOv8 object detection
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image

def predict_image(model, image, confidence=0.25, iou=0.45):
    """
    Predict single image - core inference function for object detection
    
    Args:
        model: Trained YOLO model
        image: Image as numpy array or path
        confidence: Confidence threshold
        iou: IoU threshold for NMS
    
    Returns:
        dict: Detection results with bounding boxes
    """
    # Load image if path provided
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Ensure image is numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Run inference
    results = model(image, conf=confidence, iou=iou)
    
    # Parse results
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                detection = {
                    'bbox': [float(x) for x in box],  # [x1, y1, x2, y2]
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': model.names[cls] if hasattr(model, 'names') else f'class_{cls}'
                }
                detections.append(detection)
    
    # Calculate image dimensions for relative coordinates
    h, w = image.shape[:2]
    
    # Format final result
    result = {
        'detections': detections,
        'num_detections': len(detections),
        'image_size': {'width': w, 'height': h}
    }
    
    return result

class ObjectDetectionInference:
    """YOLOv8 object detection inference wrapper"""
    
    def __init__(self, model_path, confidence=0.25, iou=0.45):
        self.confidence = confidence
        self.iou = iou
        
        # Load YOLO model
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found, loading default YOLOv8n")
            self.model = YOLO('yolov8n.pt')
        else:
            self.model = YOLO(model_path)
        
        # Get class names
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        
        print(f"Model loaded: {model_path}")
        print(f"Classes: {self.class_names}")
        print(f"Confidence threshold: {confidence}")
        print(f"IoU threshold: {iou}")
    
    def predict(self, image, confidence=None, iou=None):
        """Predict single image"""
        conf = confidence if confidence is not None else self.confidence
        iou_thresh = iou if iou is not None else self.iou
        
        return predict_image(self.model, image, conf, iou_thresh)
    
    def predict_batch(self, images, confidence=None, iou=None):
        """Predict batch of images"""
        results = []
        conf = confidence if confidence is not None else self.confidence
        iou_thresh = iou if iou is not None else self.iou
        
        for image in images:
            result = predict_image(self.model, image, conf, iou_thresh)
            results.append(result)
        
        return results
    
    def draw_detections(self, image, detections, thickness=2):
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image (numpy array)
            detections: List of detection dictionaries
            thickness: Line thickness for bounding boxes
            
        Returns:
            numpy array: Image with drawn bounding boxes
        """
        img = image.copy()
        
        # Define colors for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(
                img, 
                (x1, y1 - label_size[1] - 10), 
                (x1 + label_size[0], y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                img, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )
        
        return img
    
    def save_annotated_image(self, image, detections, output_path):
        """Save image with annotations"""
        annotated_img = self.draw_detections(image, detections)
        
        # Convert RGB to BGR for OpenCV saving
        if len(annotated_img.shape) == 3:
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_path, annotated_img)
        print(f"Annotated image saved to: {output_path}")
