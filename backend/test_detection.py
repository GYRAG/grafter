#!/usr/bin/env python3
"""
Test script to check what the YOLOv8n model detects.
"""

import cv2
import numpy as np
from root_detector import RootDetectorAPI

def test_model_detection():
    """Test what the model detects on a simple image."""
    
    # Initialize detector
    detector = RootDetectorAPI()
    
    # Load model
    print("Loading YOLOv8n model...")
    success = detector.load_model()
    if not success:
        print("Failed to load model!")
        return
    
    print("Model loaded successfully!")
    
    # Create a simple test image (white background with some shapes)
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background
    
    # Draw some test shapes that might be detected
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 0, 0), 2)  # Black rectangle
    cv2.circle(test_image, (400, 150), 50, (0, 0, 0), 2)  # Black circle
    cv2.line(test_image, (50, 300), (300, 350), (0, 0, 0), 3)  # Black line
    
    # Test detection
    print("\nTesting detection on simple image...")
    result = detector.detect_roots_in_image(test_image)
    
    print(f"Detection result: {result}")
    
    if result['success']:
        print(f"Number of detections: {len(result['detections'])}")
        for i, detection in enumerate(result['detections']):
            print(f"Detection {i}: confidence={detection['confidence']:.3f}, box={detection['box']}")
    else:
        print(f"Detection failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_model_detection()
