#!/usr/bin/env python3
"""
Dataset preparation script for YOLOv8 branch detection.
Converts COCO format annotations to YOLO format.
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from pycocotools.coco import COCO

def convert_coco_to_yolo(coco_json_path, image_dir, output_dir, class_name="branch"):
    """
    Convert COCO format annotations to YOLO format.
    """
    print(f"Converting COCO annotations from {coco_json_path}")
    
    # Load COCO data
    coco = COCO(coco_json_path)
    
    # Get all image IDs
    image_ids = coco.getImgIds()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    # Process each image
    converted_count = 0
    skipped_count = 0
    
    for img_id in image_ids:
        # Get image info
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Source image path
        src_img_path = os.path.join(image_dir, img_filename)
        
        if not os.path.exists(src_img_path):
            print(f"⚠️  Image not found: {src_img_path}")
            skipped_count += 1
            continue
        
        # Copy image to output directory
        dst_img_path = os.path.join(output_dir, "images", img_filename)
        shutil.copy2(src_img_path, dst_img_path)
        
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        
        # Create YOLO format label file
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(output_dir, "labels", label_filename)
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # Get bounding box (COCO format: [x, y, width, height])
                bbox = ann['bbox']
                x, y, w, h = bbox
                
                # Convert to YOLO format (normalized center coordinates and dimensions)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                
                # Ensure values are within [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # Skip invalid boxes
                if width <= 0 or height <= 0:
                    continue
                
                # Write YOLO format: class_id x_center y_center width height
                # All objects are class 0 (branch)
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        converted_count += 1
    
    print(f"Converted {converted_count} images")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images")
    
    return converted_count

def create_yolo_config(output_dir, class_names=["branch"]):
    """Create YOLO configuration file."""
    config_content = f"""# YOLOv8 configuration for branch detection
path: {os.path.abspath(output_dir)}
train: images
val: images  # Using same images for validation in this case

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""
    
    config_path = os.path.join(output_dir, "data.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Created YOLO config: {config_path}")
    return config_path

def split_dataset(data_dir, train_ratio=0.8):
    """Split dataset into train and validation sets."""
    print(f"\nSplitting dataset (train: {train_ratio*100:.0f}%, val: {(1-train_ratio)*100:.0f}%)")
    
    # Get all image files
    images_dir = os.path.join(data_dir, "images")
    image_files = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    
    # Shuffle and split
    import random
    random.seed(42)  # For reproducibility
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"   Train images: {len(train_files)}")
    print(f"   Validation images: {len(val_files)}")
    
    # Create train and val directories
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)
    
    # Move files to train directory
    for img_file in train_files:
        # Copy image
        dst_img = os.path.join(train_dir, "images", img_file.name)
        shutil.copy2(img_file, dst_img)
        
        # Copy corresponding label from the main labels directory
        label_file = os.path.join(data_dir, "labels", img_file.with_suffix('.txt').name)
        if os.path.exists(label_file):
            dst_label = os.path.join(train_dir, "labels", img_file.with_suffix('.txt').name)
            shutil.copy2(label_file, dst_label)
    
    # Move files to val directory
    for img_file in val_files:
        # Copy image
        dst_img = os.path.join(val_dir, "images", img_file.name)
        shutil.copy2(img_file, dst_img)
        
        # Copy corresponding label from the main labels directory
        label_file = os.path.join(data_dir, "labels", img_file.with_suffix('.txt').name)
        if os.path.exists(label_file):
            dst_label = os.path.join(val_dir, "labels", img_file.with_suffix('.txt').name)
            shutil.copy2(label_file, dst_label)
    
    # Update config file
    config_content = f"""# YOLOv8 configuration for branch detection
path: {os.path.abspath(data_dir)}
train: train/images
val: val/images

# Classes
nc: 1  # number of classes
names: ['branch']  # class names
"""
    
    config_path = os.path.join(data_dir, "data.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Updated config: {config_path}")
    return config_path

def main():
    """Main function to prepare YOLO dataset."""
    print("Branch Detection Dataset Preparation (YOLO Format)")
    print("=" * 60)
    
    try:
        # Define paths
        train_json = "train/_annotations.coco.json"
        test_json = "test/_annotations.coco.json"
        train_dir = "train"
        test_dir = "test"
        output_dir = "yolo_dataset"
        
        # Check if files exist
        if not os.path.exists(train_json):
            raise FileNotFoundError(f"Training annotations not found: {train_json}")
        if not os.path.exists(test_json):
            raise FileNotFoundError(f"Test annotations not found: {test_json}")
        
        # Convert training data
        print("\nConverting training data...")
        train_count = convert_coco_to_yolo(train_json, train_dir, output_dir)
        
        # Convert test data (append to training data for now)
        print("\nConverting test data...")
        test_count = convert_coco_to_yolo(test_json, test_dir, output_dir)
        
        # Split dataset
        config_path = split_dataset(output_dir, train_ratio=0.8)
        
        print("\n" + "=" * 60)
        print("YOLO dataset preparation completed successfully!")
        print(f"   Total images: {train_count + test_count}")
        print(f"   Dataset directory: {output_dir}")
        print(f"   Config file: {config_path}")
        print("\nNext steps:")
        print("1. Run: python train_model_yolo.py")
        
        return True
        
    except Exception as e:
        print(f"\nError during dataset preparation: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
