#!/usr/bin/env python3
"""
Setup script for Detectron2 branch detection training environment.
Optimized for RTX 4070 Mobile with 8GB VRAM.
"""

import subprocess
import sys
import torch
import os

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

def check_gpu():
    """Check GPU availability and CUDA version."""
    print("\n🔍 Checking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"✅ GPU detected: {gpu_name}")
        print(f"✅ CUDA version: {cuda_version}")
        print(f"✅ Available GPUs: {gpu_count}")
        
        # Check VRAM
        if gpu_count > 0:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 6:
                print("⚠️  Warning: GPU has less than 6GB VRAM. Consider reducing batch size further.")
            elif gpu_memory >= 8:
                print("✅ GPU memory is sufficient for training with batch size 2")
        
        return True
    else:
        print("❌ No GPU detected. Training will be very slow on CPU.")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", f"{python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install PyTorch with CUDA support
    if not run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "Installing PyTorch with CUDA 12.1 support"
    ):
        return False
    
    # Install Detectron2
    if not run_command(
        "pip install 'git+https://github.com/facebookresearch/detectron2.git'",
        "Installing Detectron2"
    ):
        return False
    
    # Install other dependencies
    dependencies = [
        "opencv-python",
        "pycocotools",
        "matplotlib",
        "tensorboard",
        "pillow",
        "numpy",
        "tqdm"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    return True

def verify_installation():
    """Verify that all packages are installed correctly."""
    print("\n🔍 Verifying installation...")
    
    try:
        import detectron2
        print(f"✅ Detectron2 version: {detectron2.__version__}")
        
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        from pycocotools.coco import COCO
        print("✅ pycocotools imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Detectron2 environment for branch detection training")
    print("=" * 60)
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        return False
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Setup failed during verification")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python prepare_dataset.py")
    print("2. Run: python train_model.py")
    
    if not gpu_available:
        print("\n⚠️  Warning: No GPU detected. Training will be very slow on CPU.")
        print("Consider using Google Colab or another GPU-enabled environment.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
