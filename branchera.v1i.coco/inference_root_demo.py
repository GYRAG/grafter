#!/usr/bin/env python3
"""
Inference demo script for the trained YOLOv8 root detection model.
Test the model on new images and visualize results.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from ultralytics import YOLO
import argparse
import time
import json

class RootDetectorYOLO:
    """Root detection inference class using YOLOv8."""
    
    def __init__(self, model_path="output/root_detection/weights/best.pt"):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """Load the trained YOLOv8 model."""
        print("Loading trained YOLOv8 model...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load model
        self.model = YOLO(self.model_path)
        
        print(f"Model loaded from: {self.model_path}")
        
    def detect_roots(self, image_path, confidence_threshold=0.5):
        """Detect roots in an image."""
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            img = image_path
        
        # Run inference
        start_time = time.time()
        results = self.model(img, conf=confidence_threshold)
        inference_time = time.time() - start_time
        
        # Extract results
        result = results[0]  # Get first (and only) result
        
        # Get bounding boxes, confidence scores, and class IDs
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
        class_ids = result.boxes.cls.cpu().numpy() if result.boxes is not None else np.array([])
        
        num_roots = len(boxes)
        
        results_dict = {
            "num_roots": num_roots,
            "boxes": boxes.tolist() if len(boxes) > 0 else [],
            "scores": scores.tolist() if len(scores) > 0 else [],
            "class_ids": class_ids.tolist() if len(class_ids) > 0 else [],
            "inference_time": inference_time,
            "image_shape": img.shape
        }
        
        return results_dict, result
    
    def visualize_detection(self, image_path, results, result, save_path=None, show_confidence=True):
        """Visualize detection results."""
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display image
        ax.imshow(img_rgb)
        
        # Draw bounding boxes
        if len(results['boxes']) > 0:
            for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
                x1, y1, x2, y2 = box
                
                # Draw rectangle
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                # Draw center line (vertical line through the center)
                center_x = (x1 + x2) / 2
                ax.axvline(x=center_x, ymin=y1/img_rgb.shape[0], ymax=y2/img_rgb.shape[0], 
                          color='blue', linewidth=3, alpha=0.8)
                
                # Draw center point
                center_y = (y1 + y2) / 2
                ax.plot(center_x, center_y, 'bo', markersize=8, markeredgecolor='white', markeredgewidth=2)
                
                # Add confidence score
                if show_confidence:
                    ax.text(x1, y1-5, f"Root {i+1}: {score:.2f}", 
                           color='red', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add title
        title = f"Root Detection - {results['num_roots']} roots found"
        if results['inference_time'] > 0:
            title += f" (inference: {results['inference_time']*1000:.1f}ms)"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"   Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def process_image(self, image_path, confidence_threshold=0.5, save_output=True, output_dir="./output/inference"):
        """Process a single image and return results."""
        print(f"\nProcessing image: {image_path}")
        
        # Detect roots
        results, result = self.detect_roots(image_path, confidence_threshold)
        
        # Print results
        print(f"   Roots detected: {results['num_roots']}")
        print(f"   Inference time: {results['inference_time']*1000:.1f}ms")
        
        if results['num_roots'] > 0:
            print("   Detection details:")
            for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
                x1, y1, x2, y2 = box
                print(f"     Root {i+1}: confidence={score:.3f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        
        # Create visualization
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            image_name = Path(image_path).stem
            save_path = os.path.join(output_dir, f"{image_name}_root_detection.png")
            self.visualize_detection(image_path, results, result, save_path)
        
        return results
    
    def process_batch(self, image_dir, confidence_threshold=0.5, output_dir="./output/inference"):
        """Process a batch of images."""
        print(f"\nProcessing batch from: {image_dir}")
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"   No image files found in {image_dir}")
            return []
        
        print(f"   Found {len(image_files)} images")
        
        # Process each image
        all_results = []
        for image_file in image_files:
            try:
                results = self.process_image(str(image_file), confidence_threshold, True, output_dir)
                all_results.append({
                    "image": str(image_file),
                    "results": results
                })
            except Exception as e:
                print(f"   Error processing {image_file}: {e}")
        
        # Save batch results
        batch_results_path = os.path.join(output_dir, "root_batch_results.json")
        with open(batch_results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"   Batch results saved to: {batch_results_path}")
        return all_results
    
    def benchmark_performance(self, image_path, num_runs=10):
        """Benchmark inference performance."""
        print(f"\nBenchmarking performance on {num_runs} runs...")
        
        times = []
        for i in range(num_runs):
            results, _ = self.detect_roots(image_path)
            times.append(results['inference_time'])
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time
        
        print(f"   Average inference time: {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms")
        print(f"   FPS: {fps:.1f}")
        
        return {
            "average_time": avg_time,
            "std_time": std_time,
            "fps": fps,
            "times": times
        }

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="YOLOv8 Root Detection Inference Demo")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--batch", type=str, help="Path to directory of images")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="./output/inference", help="Output directory")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--model", type=str, default="output/root_detection/weights/best.pt", help="Path to trained model")
    
    args = parser.parse_args()
    
    print("YOLOv8 Root Detection Inference Demo")
    print("=" * 50)
    
    try:
        # Create detector
        detector = RootDetectorYOLO(args.model)
        detector.load_model()
        
        if args.benchmark and args.image:
            # Run benchmark
            detector.benchmark_performance(args.image)
        
        elif args.image:
            # Process single image
            detector.process_image(args.image, args.confidence, True, args.output)
        
        elif args.batch:
            # Process batch
            detector.process_batch(args.batch, args.confidence, args.output)
        
        else:
            # Demo with test images
            print("\nRunning demo on test images...")
            test_images = []
            
            # Find test images
            test_dir = "test"
            if os.path.exists(test_dir):
                for ext in ['.jpg', '.jpeg', '.png']:
                    test_images.extend(Path(test_dir).glob(f"*{ext}"))
            
            if test_images:
                # Process first 3 test images
                for i, img_path in enumerate(test_images[:3]):
                    detector.process_image(str(img_path), args.confidence, True, args.output)
            else:
                print("   No test images found. Use --image or --batch to specify images.")
        
        print("\n" + "=" * 50)
        print("Inference completed successfully!")
        print(f"   Results saved to: {args.output}")
        
        return True
        
    except Exception as e:
        print(f"\nInference failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
