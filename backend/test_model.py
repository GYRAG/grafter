#!/usr/bin/env python3
"""
Test script to debug model loading issues.
"""

import os
import sys

print("Testing model loading...")
print(f"Current directory: {os.getcwd()}")
print(f"Python version: {sys.version}")

# Check if model file exists
model_paths = [
    "../branchera.v1i.coco/output/root_detection/weights/best.pt",
    "branchera.v1i.coco/output/root_detection/weights/best.pt",
    "../output/root_detection/weights/best.pt",
    "output/root_detection/weights/best.pt"
]

print("\nChecking model file paths:")
for path in model_paths:
    exists = os.path.exists(path)
    print(f"  {path}: {'EXISTS' if exists else 'NOT FOUND'}")
    if exists:
        size = os.path.getsize(path)
        print(f"    Size: {size:,} bytes")

# Try importing ultralytics
print("\nTesting ultralytics import:")
try:
    from ultralytics import YOLO
    print("SUCCESS: ultralytics imported successfully")
    
    # Try loading model
    model_path = "../branchera.v1i.coco/output/root_detection/weights/best.pt"
    if os.path.exists(model_path):
        print(f"\nTrying to load model from: {model_path}")
        try:
            # Method 1: Try with weights_only=False
            import torch
            original_load = torch.load
            def safe_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = safe_load
            
            model = YOLO(model_path)
            print("SUCCESS: Model loaded successfully!")
            print(f"Model info: {model.model}")
            
            # Restore original torch.load
            torch.load = original_load
            
        except Exception as e:
            print(f"ERROR: Error loading model: {e}")
            print(f"Error type: {type(e)}")
            # Restore original torch.load
            torch.load = original_load
    else:
        print(f"ERROR: Model file not found: {model_path}")
        
except ImportError as e:
    print(f"ERROR: Error importing ultralytics: {e}")

print("\nTest completed.")
