# Branch Detection with Detectron2

A complete pipeline for training and deploying a branch detection model using Detectron2, optimized for RTX 4070 Mobile with 8GB VRAM.

## Dataset

- **322 total images** (268 train / 54 test)
- COCO format with segmentation annotations
- Images resized to 640x640
- All categories consolidated into single "branch" class

## Quick Start

### 1. Setup Environment
```bash
python setup_environment.py
```

### 2. Prepare Dataset
```bash
python prepare_dataset.py
```

### 3. Train Model
```bash
python train_model.py
```

### 4. Evaluate Model
```bash
python evaluate_model.py
```

### 5. Run Inference
```bash
# Single image
python inference_demo.py --image path/to/image.jpg

# Batch processing
python inference_demo.py --batch path/to/images/

# Performance benchmark
python inference_demo.py --image path/to/image.jpg --benchmark
```

## Training Configuration

- **Model**: Faster R-CNN with ResNet-50 FPN
- **Batch Size**: 2 (optimized for 8GB VRAM)
- **Max Iterations**: 3000 (~11 epochs)
- **Learning Rate**: 0.001 with warmup
- **Mixed Precision**: Enabled for memory efficiency
- **Expected Training Time**: 20-25 minutes on RTX 4070 Mobile

## Expected Performance

- **Training Time**: ~20-25 minutes
- **Inference Speed**: ~15-20 FPS
- **Expected mAP**: 40-60%
- **Model Size**: ~160MB

## Output Files

After training, the following files will be created:

```
output/
├── model_final.pth          # Trained model
├── training_info.json       # Training metadata
├── evaluation_metrics.json  # Detailed evaluation results
├── predictions/             # Visualization images
└── inference/               # Inference results
```

## Requirements

- Python 3.8+
- CUDA 11.8/12.1
- RTX 4070 Mobile (8GB VRAM) or similar GPU
- 4GB+ system RAM

## Troubleshooting

### GPU Memory Issues
If you encounter GPU memory errors:
1. Reduce batch size to 1 in `train_model.py`
2. Enable gradient checkpointing
3. Use smaller input resolution

### Installation Issues
If Detectron2 installation fails:
1. Update pip: `pip install --upgrade pip`
2. Install PyTorch first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
3. Then install Detectron2: `pip install 'git+https://github.com/facebookresearch/detectron2.git'`

## Model Performance

The model provides:
- **Object Detection**: Bounding boxes around branches
- **Confidence Scores**: Detection confidence for each branch
- **Real-time Inference**: Fast inference suitable for real-time applications

## Customization

To adapt for different use cases:
1. Modify `prepare_dataset.py` for different annotation formats
2. Adjust training parameters in `train_model.py`
3. Change confidence thresholds in `inference_demo.py`
4. Update visualization settings in `evaluate_model.py`
