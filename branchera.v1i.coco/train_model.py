#!/usr/bin/env python3
"""
Training script for branch detection using Detectron2.
Optimized for RTX 4070 Mobile with 8GB VRAM.
"""

import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import get_event_storage
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
from detectron2.engine import hooks
from detectron2.utils.events import EventStorage
import logging

# Import our dataset preparation
from prepare_dataset import register_datasets

class BranchTrainer(DefaultTrainer):
    """Custom trainer for branch detection with optimizations for 8GB VRAM."""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
    @classmethod
    def build_train_loader(cls, cfg):
        """Build training data loader with optimizations."""
        return build_detection_train_loader(cfg)
    
    @classmethod
    def build_test_loader(cls, cfg):
        """Build test data loader."""
        return build_detection_test_loader(cfg)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Build evaluator for COCO metrics."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def setup_config():
    """Setup Detectron2 configuration optimized for RTX 4070 Mobile (8GB VRAM)."""
    print("⚙️  Setting up training configuration...")
    
    cfg = get_cfg()
    
    # Use Faster R-CNN with ResNet-50 FPN backbone
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = ("branch_train",)
    cfg.DATASETS.TEST = ("branch_test",)
    cfg.DATALOADER.NUM_WORKERS = 2  # Reduce for 8GB VRAM
    
    # Model configuration
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only 'branch' class
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Training configuration optimized for 8GB VRAM
    cfg.SOLVER.IMS_PER_BATCH = 2  # Small batch size for 8GB VRAM
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 3000  # Quick testing: ~11 epochs
    cfg.SOLVER.STEPS = (2000, 2800)  # Learning rate decay steps
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = 500  # Save checkpoint every 500 iterations
    
    # Mixed precision training for 8GB VRAM
    cfg.SOLVER.AMP.ENABLED = True
    
    # Output configuration
    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Evaluation configuration
    cfg.TEST.EVAL_PERIOD = 1000  # Evaluate every 1000 iterations
    
    print("✅ Configuration setup complete")
    print(f"   Model: Faster R-CNN with ResNet-50 FPN")
    print(f"   Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"   Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"   Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"   Mixed precision: {cfg.SOLVER.AMP.ENABLED}")
    print(f"   Output directory: {cfg.OUTPUT_DIR}")
    
    return cfg

def check_gpu_memory():
    """Check GPU memory and provide recommendations."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔍 GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 6:
            print("⚠️  Warning: GPU has less than 6GB VRAM. Consider reducing batch size to 1.")
            return False
        elif gpu_memory >= 8:
            print("✅ GPU memory is sufficient for training with batch size 2")
            return True
        else:
            print("✅ GPU memory should be sufficient for training")
            return True
    else:
        print("❌ No GPU detected. Training will be very slow on CPU.")
        return False

def train_model():
    """Main training function."""
    print("🌿 Branch Detection Model Training")
    print("=" * 50)
    
    # Check GPU
    gpu_ok = check_gpu_memory()
    if not gpu_ok:
        print("⚠️  Proceeding with training despite GPU memory concerns...")
    
    # Setup logger
    setup_logger()
    
    # Register datasets
    print("\n📝 Registering datasets...")
    register_datasets()
    
    # Setup configuration
    cfg = setup_config()
    
    # Create trainer
    print("\n🏋️  Creating trainer...")
    trainer = BranchTrainer(cfg)
    
    # Start training
    print("\n🚀 Starting training...")
    print(f"   Training will take approximately 20-25 minutes on RTX 4070 Mobile")
    print(f"   Progress will be saved to: {cfg.OUTPUT_DIR}")
    print(f"   Checkpoints saved every {cfg.SOLVER.CHECKPOINT_PERIOD} iterations")
    
    start_time = time.time()
    
    try:
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed successfully!")
        print(f"   Total training time: {training_time/60:.1f} minutes")
        
        # Save final model info
        model_info = {
            "training_time_minutes": training_time / 60,
            "total_iterations": cfg.SOLVER.MAX_ITER,
            "batch_size": cfg.SOLVER.IMS_PER_BATCH,
            "learning_rate": cfg.SOLVER.BASE_LR,
            "model": "Faster R-CNN with ResNet-50 FPN",
            "dataset": "branch detection",
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(cfg.OUTPUT_DIR, "training_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"   Model saved to: {cfg.OUTPUT_DIR}/model_final.pth")
        print(f"   Training info saved to: {cfg.OUTPUT_DIR}/training_info.json")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False

def quick_evaluation():
    """Quick evaluation of the trained model."""
    print("\n📊 Running quick evaluation...")
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join("./output", "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = ("branch_test",)
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Get test dataset
    test_dataset = DatasetCatalog.get("branch_test")
    
    # Run inference on a few samples
    print("   Running inference on 5 test samples...")
    correct_predictions = 0
    total_predictions = 0
    
    for i, sample in enumerate(test_dataset[:5]):
        img = cv2.imread(sample["file_name"])
        outputs = predictor(img)
        
        predicted_boxes = len(outputs["instances"])
        actual_boxes = len(sample["annotations"])
        
        print(f"   Sample {i+1}: Predicted {predicted_boxes} branches, Actual {actual_boxes} branches")
        
        total_predictions += 1
        if abs(predicted_boxes - actual_boxes) <= 1:  # Allow 1 box difference
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions * 100
    print(f"   Quick accuracy: {accuracy:.1f}%")
    
    return accuracy

def main():
    """Main function."""
    try:
        # Train the model
        success = train_model()
        
        if success:
            # Quick evaluation
            quick_evaluation()
            
            print("\n" + "=" * 50)
            print("🎉 Training pipeline completed successfully!")
            print("\nNext steps:")
            print("1. Run: python evaluate_model.py (for detailed evaluation)")
            print("2. Run: python inference_demo.py (for testing on new images)")
            print("3. Check output/ directory for model files and logs")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Training pipeline failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    import cv2
    success = main()
    sys.exit(0 if success else 1)
