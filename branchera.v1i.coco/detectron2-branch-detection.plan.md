# Train Branch Detection Model with Detectron2

## Dataset Analysis
- **322 total images** (268 train / 54 test)
- COCO format with segmentation annotations (640x640)
- **Issue to fix**: Category names need cleaning ("idk", "------------------------------", "branch - v1 2024-04-09 2-44pm")

## Implementation Steps

### 1. Environment Setup
Create Python script to:
- Install Detectron2 with CUDA 11.8/12.1 support for RTX 4070 Mobile
- Install dependencies: torch, torchvision, pycocotools, opencv-python
- Verify GPU availability

**File**: `setup_environment.py`

### 2. Dataset Preparation
Create dataset registration script:
- Register train/test datasets with Detectron2's DatasetCatalog
- Convert COCO annotations for object detection (extract bbox from segmentation)
- Clean up category names: merge all 3 categories into single "branch" class
- Add validation split option (currently only train/test exists)

**File**: `prepare_dataset.py`

**Key issue**: The 3 category IDs (0,1,2) all appear to be branches but have confusing names - will consolidate into single class.

### 3. Training Configuration
Create training script with:
- **Model**: Faster R-CNN with ResNet-50 FPN backbone (good balance speed/accuracy)
- **Quick testing config**:
  - Max iterations: 3000 (approximately 11 epochs with 268 images)
  - Batch size: 2 (suitable for RTX 4070 Mobile 8GB VRAM)
  - Learning rate: 0.001 with warmup
  - Checkpoint every 500 iterations
- GPU optimization: mixed precision (FP16) training - **critical for 8GB VRAM**
- Enable TensorBoard logging

**File**: `train_model.py`

**Files created during training**:
- `output/` - model checkpoints, logs, metrics
- `output/model_final.pth` - final trained model

### 4. Validation Script
Create evaluation script:
- Load trained model
- Run inference on test set
- Calculate COCO metrics (mAP, AP50, AP75)
- Visualize predictions on sample images
- Save visualization to `output/predictions/`

**File**: `evaluate_model.py`

### 5. Quick Inference Demo
Simple script to test model on new images:
- Load trained model
- Run inference on any image
- Display bounding boxes with confidence scores
- Save annotated results

**File**: `inference_demo.py`

## Expected Results
- Training time: ~20-25 minutes on RTX 4070 Mobile (8GB VRAM)
- Expected mAP: 40-60% (depends on annotation quality)
- Model size: ~160MB
- Inference speed: ~15-20 FPS on RTX 4070 Mobile
- Memory usage: ~6-7GB VRAM (leaving headroom)

## Key Files Structure
```
branchera.v1i.coco/
├── train/              # existing training images
├── test/               # existing test images
├── setup_environment.py
├── prepare_dataset.py
├── train_model.py
├── evaluate_model.py
├── inference_demo.py
└── output/             # created during training
    ├── model_final.pth
    ├── metrics.json
    └── predictions/
```

### To-dos
- [ ] Create setup_environment.py to install Detectron2 and dependencies with CUDA support
- [ ] Create prepare_dataset.py to register COCO dataset and clean category names
- [ ] Create train_model.py with Faster R-CNN config optimized for RTX 4070 Mobile (8GB VRAM)
- [ ] Create evaluate_model.py for testing and visualization
- [ ] Create inference_demo.py for quick testing on new images
