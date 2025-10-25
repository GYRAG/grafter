#!/usr/bin/env python3
"""
Convert YOLOv8 model to TensorFlow.js format for React integration.
"""

import os
from ultralytics import YOLO
import tensorflow as tf

def convert_yolo_to_tfjs(model_path, output_dir="tfjs_model"):
    """
    Convert YOLOv8 model to TensorFlow.js format.
    
    Args:
        model_path (str): Path to the trained YOLOv8 model (.pt file)
        output_dir (str): Directory to save TensorFlow.js model
    """
    print("Converting YOLOv8 model to TensorFlow.js format...")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return False
    
    try:
        # Load YOLOv8 model
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to TensorFlow.js format
        print("Exporting to TensorFlow.js format...")
        model.export(
            format='tfjs',
            imgsz=640,  # Input image size
            optimize=True,  # Optimize for web deployment
            half=False,  # Use full precision (better compatibility)
            int8=False,  # Use full precision
            dynamic=False,  # Static input shape
            simplify=True,  # Simplify model
            opset=None,  # Use default opset
            workspace=4,  # Workspace size in GB
            nms=False,  # Don't include NMS in model
            batch=1  # Batch size of 1 for web inference
        )
        
        # The export creates files in the current directory
        # Move them to the output directory
        tfjs_files = [f for f in os.listdir('.') if f.startswith('yolov8') and f.endswith('.tfjs')]
        
        if tfjs_files:
            print(f"Found TensorFlow.js files: {tfjs_files}")
            
            # Move files to output directory
            for file in tfjs_files:
                src = file
                dst = os.path.join(output_dir, file)
                os.rename(src, dst)
                print(f"Moved {src} to {dst}")
        
        print(f"✅ Model converted successfully!")
        print(f"📁 TensorFlow.js model saved to: {output_dir}/")
        print("\nNext steps:")
        print("1. Copy the tfjs_model/ directory to your React project's public/ folder")
        print("2. Update the model path in your React component")
        print("3. Install TensorFlow.js dependencies: npm install @tensorflow/tfjs @tensorflow/tfjs-backend-webgl")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return False

def main():
    """Main function."""
    print("YOLOv8 to TensorFlow.js Converter")
    print("=" * 50)
    
    # Default model path
    model_path = "output/root_detection/weights/best.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please train your model first or provide the correct path.")
        return False
    
    # Convert model
    success = convert_yolo_to_tfjs(model_path)
    
    if success:
        print("\n🎉 Conversion completed successfully!")
        print("\nFiles created:")
        print("- tfjs_model/ directory with TensorFlow.js model files")
        print("- Ready for React integration")
    else:
        print("\n❌ Conversion failed!")
    
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
