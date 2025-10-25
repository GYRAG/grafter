#!/usr/bin/env python3
"""
Training script for root detection using YOLOv8.
Optimized for RTX 4070 Mobile with 8GB VRAM.
"""

import os
import time
from datetime import datetime
import torch
from ultralytics import YOLO
import json

def check_gpu():
    """Check GPU availability and provide recommendations."""
    print("\nChecking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU detected: {gpu_name}")
        print(f"Available GPUs: {gpu_count}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 6:
            print("Warning: GPU has less than 6GB VRAM. Consider reducing batch size.")
            return False
        elif gpu_memory >= 8:
            print("GPU memory is sufficient for training with batch size 8")
            return True
        else:
            print("GPU memory should be sufficient for training")
            return True
    else:
        print("No GPU detected. Training will be very slow on CPU.")
        return False

def train_root_model():
    """Train YOLOv8 model for root detection."""
    print("Root Detection Model Training (YOLOv8)")
    print("=" * 60)
    
    # Check GPU
    gpu_ok = check_gpu()
    if not gpu_ok:
        print("Proceeding with training despite GPU memory concerns...")
    
    # Check if dataset exists
    dataset_path = "data.yaml"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return False
    
    # Load YOLOv8 model
    print("\nLoading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # nano version for quick training
    
    # Training parameters optimized for RTX 4070 Mobile (8GB VRAM)
    training_args = {
        'data': dataset_path,
        'epochs': 100,  # More epochs for better accuracy
        'batch': 8,     # Batch size for 8GB VRAM
        'imgsz': 640,   # Image size
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 4,   # Number of workers
        'patience': 15, # Early stopping patience
        'save': True,
        'save_period': 20,  # Save checkpoint every 20 epochs
        'cache': False, # Disable caching to save memory
        'project': 'output',
        'name': 'root_detection',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.01,    # Initial learning rate
        'lrf': 0.01,    # Final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,     # Box loss gain
        'cls': 0.5,     # Classification loss gain
        'dfl': 1.5,     # DFL loss gain
        'val': True,    # Validate during training
        'plots': True,  # Generate training plots
        'verbose': True,
    }
    
    print("Model loaded successfully")
    print(f"   Model: YOLOv8n (nano)")
    print(f"   Batch size: {training_args['batch']}")
    print(f"   Epochs: {training_args['epochs']}")
    print(f"   Image size: {training_args['imgsz']}")
    print(f"   Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Start training
    print("\nStarting training...")
    print(f"   Training will take approximately 30-40 minutes on RTX 4070 Mobile")
    print(f"   Progress will be saved to: output/root_detection/")
    
    start_time = time.time()
    
    try:
        # Train the model
        results = model.train(**training_args)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed successfully!")
        print(f"   Total training time: {training_time/60:.1f} minutes")
        
        # Save training info
        training_info = {
            "training_time_minutes": training_time / 60,
            "epochs": training_args['epochs'],
            "batch_size": training_args['batch'],
            "model": "YOLOv8n",
            "dataset": "root detection",
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        with open("output/root_detection/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        print(f"   Model saved to: output/root_detection/weights/best.pt")
        print(f"   Training info saved to: output/root_detection/training_info.json")
        
        # Quick validation
        print("\nRunning quick validation...")
        val_results = model.val()
        
        print(f"   mAP50: {val_results.box.map50:.3f}")
        print(f"   mAP50-95: {val_results.box.map:.3f}")
        
        return True
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        return False

def main():
    """Main function."""
    try:
        # Train the model
        success = train_root_model()
        
        if success:
            print("\n" + "=" * 60)
            print("Training pipeline completed successfully!")
            print("\nNext steps:")
            print("1. Run: python inference_root_demo.py (for testing on new images)")
            print("2. Check output/root_detection/ directory for model files and logs")
        
        return success
        
    except Exception as e:
        print(f"\nTraining pipeline failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
