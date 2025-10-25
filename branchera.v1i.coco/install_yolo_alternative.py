#!/usr/bin/env python3
"""
Alternative installation script using YOLOv8 instead of Detectron2.
YOLOv8 is easier to install on Windows and provides excellent object detection performance.
"""

import subprocess
import sys
import torch

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def install_yolo():
    """Install YOLOv8 and dependencies."""
    print("🚀 Installing YOLOv8 as alternative to Detectron2")
    print("=" * 60)
    
    # Install ultralytics (YOLOv8)
    if not run_command("pip install ultralytics", "Installing YOLOv8 (ultralytics)"):
        return False
    
    # Install additional dependencies
    dependencies = [
        "opencv-python",
        "matplotlib",
        "pillow",
        "numpy",
        "tqdm",
        "pandas",
        "seaborn"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    return True

def verify_installation():
    """Verify that YOLOv8 is installed correctly."""
    print("\n🔍 Verifying YOLOv8 installation...")
    
    try:
        from ultralytics import YOLO
        import torch
        import cv2
        
        print(f"✅ YOLOv8 imported successfully")
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main installation function."""
    print("🌿 Installing YOLOv8 Alternative for Branch Detection")
    print("=" * 60)
    print("Note: YOLOv8 is easier to install on Windows and provides")
    print("excellent object detection performance for your use case.")
    print("=" * 60)
    
    # Install YOLOv8
    if not install_yolo():
        print("\n❌ Installation failed")
        return False
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Verification failed")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 YOLOv8 installation completed successfully!")
    print("\nNext steps:")
    print("1. Run: python prepare_dataset_yolo.py")
    print("2. Run: python train_model_yolo.py")
    print("3. Run: python evaluate_model_yolo.py")
    print("4. Run: python inference_demo_yolo.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
