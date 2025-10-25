#!/usr/bin/env python3
"""
Evaluation script for the trained branch detection model.
Provides detailed metrics and visualizations.
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time

# Import our dataset preparation
from prepare_dataset import register_datasets

class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model_path="./output/model_final.pth"):
        self.model_path = model_path
        self.cfg = None
        self.predictor = None
        self.test_dataset = None
        
    def setup_model(self):
        """Setup the trained model for evaluation."""
        print("🔧 Setting up model for evaluation...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
    
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = self.model_path
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.DATASETS.TEST = ("branch_test",)
        
        
        self.predictor = DefaultPredictor(self.cfg)
        
        
        self.test_dataset = DatasetCatalog.get("branch_test")
        
        print(f"✅ Model loaded from: {self.model_path}")
        print(f"✅ Test dataset: {len(self.test_dataset)} images")
        
    def run_coco_evaluation(self):
        """Run official COCO evaluation metrics."""
        print("\n📊 Running COCO evaluation metrics...")
        
        # Create evaluator
        evaluator = COCOEvaluator("branch_test", self.cfg, True, "./output/evaluation")
        
        # Build test loader
        test_loader = build_detection_test_loader(self.cfg, "branch_test")
        
        # Run evaluation
        results = inference_on_dataset(self.predictor.model, test_loader, evaluator)
        
        print("✅ COCO evaluation completed")
        return results
    
    def calculate_detailed_metrics(self):
        """Calculate detailed detection metrics."""
        print("\n📈 Calculating detailed metrics...")
        
        total_images = len(self.test_dataset)
        total_gt_boxes = 0
        total_pred_boxes = 0
        correct_detections = 0
        false_positives = 0
        false_negatives = 0
        
        # IoU threshold for matching
        iou_threshold = 0.5
        
        for i, sample in enumerate(self.test_dataset):
            # Load image
            img = cv2.imread(sample["file_name"])
            
            # Run prediction
            outputs = self.predictor(img)
            instances = outputs["instances"]
            
            # Get ground truth boxes
            gt_boxes = []
            for ann in sample["annotations"]:
                gt_boxes.append(ann["bbox"])  # [x1, y1, x2, y2]
            
            # Get predicted boxes
            pred_boxes = instances.pred_boxes.tensor.cpu().numpy()  # [x1, y1, x2, y2]
            pred_scores = instances.scores.cpu().numpy()
            
            # Filter predictions by confidence
            confident_preds = pred_boxes[pred_scores > 0.5]
            
            total_gt_boxes += len(gt_boxes)
            total_pred_boxes += len(confident_preds)
            
            # Calculate IoU between predictions and ground truth
            if len(gt_boxes) > 0 and len(confident_preds) > 0:
                # Convert to numpy arrays
                gt_boxes = np.array(gt_boxes)
                confident_preds = np.array(confident_preds)
                
                # Calculate IoU matrix
                ious = self.calculate_iou_matrix(gt_boxes, confident_preds)
                
                # Find matches
                matched_gt = set()
                matched_pred = set()
                
                for gt_idx in range(len(gt_boxes)):
                    for pred_idx in range(len(confident_preds)):
                        if ious[gt_idx, pred_idx] > iou_threshold:
                            if gt_idx not in matched_gt and pred_idx not in matched_pred:
                                matched_gt.add(gt_idx)
                                matched_pred.add(pred_idx)
                                correct_detections += 1
                
                # Count false positives and false negatives
                false_positives += len(confident_preds) - len(matched_pred)
                false_negatives += len(gt_boxes) - len(matched_gt)
            else:
                # All predictions are false positives or all ground truth are false negatives
                false_positives += len(confident_preds)
                false_negatives += len(gt_boxes)
        
        # Calculate metrics
        precision = correct_detections / (correct_detections + false_positives) if (correct_detections + false_positives) > 0 else 0
        recall = correct_detections / (correct_detections + false_negatives) if (correct_detections + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "total_images": total_images,
            "total_gt_boxes": total_gt_boxes,
            "total_pred_boxes": total_pred_boxes,
            "correct_detections": correct_detections,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "average_gt_per_image": total_gt_boxes / total_images,
            "average_pred_per_image": total_pred_boxes / total_images
        }
        
        print("✅ Detailed metrics calculated")
        return metrics
    
    def calculate_iou_matrix(self, boxes1, boxes2):
        """Calculate IoU matrix between two sets of boxes."""
        # boxes1: [N, 4] (x1, y1, x2, y2)
        # boxes2: [M, 4] (x1, y1, x2, y2)
        # Returns: [N, M] IoU matrix
        
        # Calculate intersection areas
        x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
        y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
        x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
        y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate union
        union = area1[:, np.newaxis] + area2 - intersection
        
        # Calculate IoU
        iou = intersection / union
        
        return iou
    
    def visualize_predictions(self, num_samples=10, output_dir="./output/predictions"):
        """Visualize predictions on test images."""
        print(f"\n🎨 Creating visualizations of {num_samples} test images...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, sample in enumerate(self.test_dataset[:num_samples]):
            # Load image
            img = cv2.imread(sample["file_name"])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run prediction
            outputs = self.predictor(img)
            
            # Create visualizer
            visualizer = Visualizer(img_rgb, 
                                  metadata=MetadataCatalog.get("branch_test"),
                                  scale=1.0,
                                  instance_mode=ColorMode.IMAGE_BW)
            
            # Draw predictions
            vis_output = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Original image with ground truth
            ax1.imshow(img_rgb)
            ax1.set_title(f"Ground Truth - {len(sample['annotations'])} branches")
            ax1.axis('off')
            
            # Draw ground truth boxes
            for ann in sample["annotations"]:
                bbox = ann["bbox"]
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='green', facecolor='none')
                ax1.add_patch(rect)
            
            # Predicted image
            ax2.imshow(vis_output.get_image())
            ax2.set_title(f"Predictions - {len(outputs['instances'])} branches")
            ax2.axis('off')
            
            # Add confidence scores
            instances = outputs["instances"]
            if len(instances) > 0:
                scores = instances.scores.cpu().numpy()
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                
                for j, (box, score) in enumerate(zip(boxes, scores)):
                    x1, y1, x2, y2 = box
                    ax2.text(x1, y1-5, f"{score:.2f}", 
                            color='red', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save visualization
            output_path = os.path.join(output_dir, f"test_sample_{i+1}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"   Saved: {output_path}")
        
        print(f"✅ Visualizations saved to {output_dir}/")
    
    def save_metrics(self, metrics, output_path="./output/evaluation_metrics.json"):
        """Save evaluation metrics to JSON file."""
        print(f"\n💾 Saving metrics to {output_path}...")
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print("✅ Metrics saved")
    
    def print_metrics_summary(self, metrics):
        """Print a summary of evaluation metrics."""
        print("\n" + "=" * 50)
        print("📊 EVALUATION RESULTS SUMMARY")
        print("=" * 50)
        
        print(f"Dataset: {metrics['total_images']} test images")
        print(f"Ground Truth: {metrics['total_gt_boxes']} branches")
        print(f"Predictions: {metrics['total_pred_boxes']} branches")
        print()
        print(f"Correct Detections: {metrics['correct_detections']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print()
        print(f"Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
        print(f"Recall: {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
        print(f"F1-Score: {metrics['f1_score']:.3f} ({metrics['f1_score']*100:.1f}%)")
        print()
        print(f"Average GT per image: {metrics['average_gt_per_image']:.2f}")
        print(f"Average Pred per image: {metrics['average_pred_per_image']:.2f}")
        
        # Performance assessment
        print("\n🎯 Performance Assessment:")
        if metrics['f1_score'] > 0.7:
            print("   Excellent performance! 🎉")
        elif metrics['f1_score'] > 0.5:
            print("   Good performance! 👍")
        elif metrics['f1_score'] > 0.3:
            print("   Fair performance. Consider more training. 🤔")
        else:
            print("   Poor performance. Check dataset quality or training parameters. 😞")

def main():
    """Main evaluation function."""
    print("🌿 Branch Detection Model Evaluation")
    print("=" * 50)
    
    try:
        # Register datasets
        register_datasets()
        
        # Create evaluator
        evaluator = ModelEvaluator()
        
        # Setup model
        evaluator.setup_model()
        
        # Run COCO evaluation
        coco_results = evaluator.run_coco_evaluation()
        
        # Calculate detailed metrics
        detailed_metrics = evaluator.calculate_detailed_metrics()
        
        # Print summary
        evaluator.print_metrics_summary(detailed_metrics)
        
        # Save metrics
        evaluator.save_metrics(detailed_metrics)
        
        # Create visualizations
        evaluator.visualize_predictions(num_samples=10)
        
        print("\n" + "=" * 50)
        print("🎉 Evaluation completed successfully!")
        print("\nFiles created:")
        print("- output/evaluation_metrics.json (detailed metrics)")
        print("- output/predictions/ (visualization images)")
        print("- output/evaluation/ (COCO evaluation results)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
