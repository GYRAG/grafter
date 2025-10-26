#!/usr/bin/env python3
"""
Root detection service for API integration.
Adapted from live_video_detection.py for Flask backend.
"""

import cv2
import numpy as np
import torch
import time
import base64
from io import BytesIO
from PIL import Image
import json
import os

# Fix PyTorch 2.6 compatibility issues
try:
    # Add safe globals for ultralytics models
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            'ultralytics.nn.tasks.DetectionModel',
            'ultralytics.nn.modules.block.C2f',
            'ultralytics.nn.modules.block.Bottleneck',
            'ultralytics.nn.modules.block.SPPF',
            'ultralytics.nn.modules.conv.Conv',
            'ultralytics.nn.modules.head.Detect',
            'ultralytics.utils.torch_utils.initialize_weights'
        ])
except Exception as e:
    print(f"Warning: Could not set safe globals: {e}")

# Import YOLO with error handling
try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error importing ultralytics: {e}")
    YOLO = None

class RootDetectorAPI:
    """Root detection service for API endpoints."""
    
    def __init__(self, model_path="yolov8n.pt"):
        self.model_path = model_path
        self.model = None
        self.confidence_threshold = 0.1
        self.tolerance = 30  # pixels tolerance for center alignment
        
        # COCO class names for debugging
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
    def load_model(self):
        """Load the YOLOv8n model."""
        print("Loading YOLOv8n root detection model...")
        
        if YOLO is None:
            print("Error: ultralytics not available")
            return False
        
        # Try multiple possible paths (prioritize main directory yolov8n.pt)
        possible_paths = [
            "yolov8n.pt",  # Main directory first
            self.model_path,
            "../yolov8n.pt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    # Try loading with different approaches
                    print(f"Attempting to load model from: {path}")
                    
                    # Try with weights_only=False for PyTorch 2.6 compatibility
                    try:
                        import torch
                        # Temporarily set weights_only to False
                        original_load = torch.load
                        def safe_load(*args, **kwargs):
                            kwargs['weights_only'] = False
                            return original_load(*args, **kwargs)
                        torch.load = safe_load
                        
                        self.model = YOLO(path)
                        self.model_path = path
                        print(f"Model loaded successfully from: {path}")
                        return True
                    except Exception as e:
                        print(f"Error loading model from {path}: {e}")
                        # Restore original torch.load
                        torch.load = original_load
                        continue
                        
                except Exception as e:
                    print(f"Error loading model from {path}: {e}")
                    continue
        
        print(f"Model not found. Tried paths: {possible_paths}")
        return False
    
    def set_confidence_threshold(self, threshold):
        """Set confidence threshold for detection."""
        self.confidence_threshold = max(0.1, min(0.9, threshold))
    
    def detect_roots_in_image(self, image_data):
        """
        Detect roots in a single image.
        
        Args:
            image_data: Base64 encoded image string or numpy array
            
        Returns:
            dict: Detection results with boxes, scores, alignment info
        """
        try:
            # Convert base64 to image if needed
            if isinstance(image_data, str):
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                # Decode base64
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                image = image_data
            
            # Resize image for faster processing (max 640px width)
            height, width = image.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Run inference
            start_time = time.time()
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            inference_time = time.time() - start_time
            
            # Process results
            result = results[0]
            height, width = image.shape[:2]
            camera_center_x = width // 2
            camera_center_y = height // 2
            
            detections = []
            aligned_count = 0
            
            print(f"DEBUG: Image size: {width}x{height}, Confidence threshold: {self.confidence_threshold}")
            print(f"DEBUG: Inference time: {inference_time:.3f}s")
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                print(f"DEBUG: Found {len(boxes)} detections")
                for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                    class_name = self.coco_classes[int(cls)] if int(cls) < len(self.coco_classes) else f"class_{int(cls)}"
                    print(f"DEBUG: Detection {i}: {class_name} (class={int(cls)}), confidence={score:.3f}, box={box}")
            else:
                print("DEBUG: No detections found")
                boxes = []
                scores = []
                classes = []
            
            if len(boxes) > 0:
                for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Calculate object center
                    obj_center_x = (x1 + x2) // 2
                    obj_center_y = (y1 + y2) // 2
                    
                    # Check alignment with camera center
                    x_aligned = abs(obj_center_x - camera_center_x) <= self.tolerance
                    y_aligned = abs(obj_center_y - camera_center_y) <= self.tolerance
                    is_aligned = x_aligned and y_aligned
                    
                    if is_aligned:
                        aligned_count += 1
                    
                    class_name = self.coco_classes[int(class_id)] if int(class_id) < len(self.coco_classes) else f"class_{int(class_id)}"
                    detection = {
                        "id": i + 1,
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "center": [int(obj_center_x), int(obj_center_y)],
                        "confidence": float(score),
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "is_aligned": is_aligned,
                        "alignment_offset": {
                            "x": int(obj_center_x - camera_center_x),
                            "y": int(obj_center_y - camera_center_y)
                        }
                    }
                    detections.append(detection)
            
            # Prepare response
            response = {
                "success": True,
                "detections": detections,
                "stats": {
                    "total_roots": len(detections),
                    "aligned_roots": aligned_count,
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "image_size": [width, height],
                    "camera_center": [camera_center_x, camera_center_y],
                    "confidence_threshold": self.confidence_threshold,
                    "tolerance": self.tolerance
                },
                "timestamp": time.time()
            }
            
            return response
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "detections": [],
                "stats": {
                    "total_roots": 0,
                    "aligned_roots": 0,
                    "inference_time_ms": 0
                }
            }
    
    def get_model_info(self):
        """Get information about the loaded model."""
        if self.model is None:
            return {"loaded": False, "error": "Model not loaded"}
        
        return {
            "loaded": True,
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "tolerance": self.tolerance,
            "model_type": "YOLOv8n"
        }
