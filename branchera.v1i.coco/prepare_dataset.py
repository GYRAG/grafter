#!/usr/bin/env python3
"""
Dataset preparation script for branch detection.
Converts COCO format annotations and registers datasets with Detectron2.
"""

import json
import os
import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from pathlib import Path

def load_coco_annotations(json_path, image_dir):
    """
    Load COCO annotations and convert to Detectron2 format.
    Consolidates all categories into a single 'branch' class.
    """
    print(f"📂 Loading COCO annotations from {json_path}")
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image id to filename mapping
    image_id_to_filename = {}
    for img in coco_data['images']:
        image_id_to_filename[img['id']] = img['file_name']
    
    # Create category mapping - consolidate all categories into 'branch'
    category_mapping = {}
    for cat in coco_data['categories']:
        category_mapping[cat['id']] = 0  # Map all to class 0 (branch)
    
    print(f"📊 Found {len(coco_data['images'])} images")
    print(f"📊 Found {len(coco_data['annotations'])} annotations")
    print(f"📊 Original categories: {[cat['name'] for cat in coco_data['categories']]}")
    print("🔄 Consolidating all categories into single 'branch' class")
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Convert to Detectron2 format
    dataset_dicts = []
    skipped_images = 0
    
    for img_info in coco_data['images']:
        image_id = img_info['id']
        filename = img_info['file_name']
        image_path = os.path.join(image_dir, filename)
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            skipped_images += 1
            continue
        
        # Get image dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️  Could not load image: {image_path}")
            skipped_images += 1
            continue
        
        height, width = img.shape[:2]
        
        # Create record
        record = {
            "file_name": image_path,
            "image_id": image_id,
            "height": height,
            "width": width,
        }
        
        # Add annotations
        objs = []
        if image_id in image_annotations:
            for ann in image_annotations[image_id]:
                # Convert segmentation to bounding box
                if 'segmentation' in ann and ann['segmentation']:
                    # Use segmentation to get more accurate bbox
                    seg = ann['segmentation'][0]  # First segmentation polygon
                    if len(seg) >= 6:  # At least 3 points (x,y pairs)
                        # Convert to numpy array and reshape
                        seg_array = np.array(seg).reshape(-1, 2)
                        x_coords = seg_array[:, 0]
                        y_coords = seg_array[:, 1]
                        
                        x_min, x_max = x_coords.min(), x_coords.max()
                        y_min, y_max = y_coords.min(), y_coords.max()
                        
                        # Ensure bbox is within image bounds
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(width, x_max)
                        y_max = min(height, y_max)
                        
                        # Skip invalid bboxes
                        if x_max <= x_min or y_max <= y_min:
                            continue
                            
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    else:
                        # Fallback to COCO bbox
                        bbox = ann['bbox']
                else:
                    # Use COCO bbox directly
                    bbox = ann['bbox']
                
                # Convert COCO bbox format [x, y, w, h] to Detectron2 format [x1, y1, x2, y2]
                x, y, w, h = bbox
                x2 = x + w
                y2 = y + h
                
                # Ensure bbox is within image bounds
                x = max(0, x)
                y = max(0, y)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # Skip invalid bboxes
                if x2 <= x or y2 <= y:
                    continue
                
                obj = {
                    "bbox": [x, y, x2, y2],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0,  # All objects are 'branch' class
                }
                objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    print(f"✅ Successfully processed {len(dataset_dicts)} images")
    if skipped_images > 0:
        print(f"⚠️  Skipped {skipped_images} images (file not found or invalid)")
    
    return dataset_dicts

def register_datasets():
    """Register train and test datasets with Detectron2."""
    print("\n📝 Registering datasets with Detectron2...")
    
    # Define dataset paths
    train_json = "train/_annotations.coco.json"
    test_json = "test/_annotations.coco.json"
    train_dir = "train"
    test_dir = "test"
    
    # Check if files exist
    if not os.path.exists(train_json):
        raise FileNotFoundError(f"Training annotations not found: {train_json}")
    if not os.path.exists(test_json):
        raise FileNotFoundError(f"Test annotations not found: {test_json}")
    
    # Load datasets
    train_dataset = load_coco_annotations(train_json, train_dir)
    test_dataset = load_coco_annotations(test_json, test_dir)
    
    # Register datasets
    DatasetCatalog.register("branch_train", lambda: train_dataset)
    DatasetCatalog.register("branch_test", lambda: test_dataset)
    
    # Register metadata
    MetadataCatalog.get("branch_train").set(thing_classes=["branch"])
    MetadataCatalog.get("branch_test").set(thing_classes=["branch"])
    
    print("✅ Datasets registered successfully!")
    print(f"   - branch_train: {len(train_dataset)} images")
    print(f"   - branch_test: {len(test_dataset)} images")
    
    return train_dataset, test_dataset

def visualize_dataset(dataset_dicts, num_samples=5, output_dir="dataset_visualization"):
    """Visualize sample images with annotations."""
    print(f"\n🎨 Creating visualizations of {num_samples} sample images...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, d in enumerate(dataset_dicts[:num_samples]):
        img = cv2.imread(d["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        
        # Draw bounding boxes
        for obj in d["annotations"]:
            bbox = obj["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Create rectangle
            from matplotlib.patches import Rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
            
            # Add label
            plt.text(x1, y1-5, "branch", color='red', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.title(f"Sample {i+1}: {len(d['annotations'])} branches detected")
        plt.axis('off')
        
        # Save visualization
        output_path = os.path.join(output_dir, f"sample_{i+1}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"   Saved: {output_path}")
    
    print(f"✅ Visualizations saved to {output_dir}/")

def print_dataset_stats(dataset_dicts, dataset_name):
    """Print statistics about the dataset."""
    print(f"\n📊 {dataset_name} Statistics:")
    
    total_images = len(dataset_dicts)
    total_annotations = sum(len(d["annotations"]) for d in dataset_dicts)
    
    # Count images with different numbers of annotations
    annotation_counts = {}
    for d in dataset_dicts:
        count = len(d["annotations"])
        annotation_counts[count] = annotation_counts.get(count, 0) + 1
    
    print(f"   Total images: {total_images}")
    print(f"   Total annotations: {total_annotations}")
    print(f"   Average annotations per image: {total_annotations/total_images:.2f}")
    print(f"   Images with annotations: {sum(1 for d in dataset_dicts if len(d['annotations']) > 0)}")
    print(f"   Images without annotations: {sum(1 for d in dataset_dicts if len(d['annotations']) == 0)}")
    
    print("   Annotation distribution:")
    for count in sorted(annotation_counts.keys()):
        print(f"     {count} annotations: {annotation_counts[count]} images")

def main():
    """Main function to prepare the dataset."""
    print("🌿 Branch Detection Dataset Preparation")
    print("=" * 50)
    
    try:
        # Register datasets
        train_dataset, test_dataset = register_datasets()
        
        # Print statistics
        print_dataset_stats(train_dataset, "Training Dataset")
        print_dataset_stats(test_dataset, "Test Dataset")
        
        # Create visualizations
        visualize_dataset(train_dataset, num_samples=3, output_dir="train_visualization")
        visualize_dataset(test_dataset, num_samples=2, output_dir="test_visualization")
        
        print("\n" + "=" * 50)
        print("🎉 Dataset preparation completed successfully!")
        print("\nNext steps:")
        print("1. Review the visualizations in train_visualization/ and test_visualization/")
        print("2. Run: python train_model.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during dataset preparation: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
