#!/usr/bin/env python3
"""
Live video detection script for root detection using YOLOv8.
Processes video from webcam or video file in real-time.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import argparse
import time
from pathlib import Path

class LiveRootDetector:
    """Real-time root detection for video streams."""
    
    def __init__(self, model_path="output/root_detection/weights/best.pt", mirror_camera=True):
        self.model_path = model_path
        self.model = None
        self.confidence_threshold = 0.5
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.mirror_camera = mirror_camera
        
    def load_model(self):
        """Load the trained YOLOv8 model."""
        print("Loading trained YOLOv8 model...")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load model
        self.model = YOLO(self.model_path)
        
        print(f"Model loaded from: {self.model_path}")
        
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_camera_center_lines(self, frame):
        """Draw center lines on the camera view."""
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2
        
        # Draw vertical center line only
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 255, 255), 2)
        
        # Draw center cross (vertical line only)
        cross_size = 20
        cv2.line(frame, (center_x, center_y - cross_size), (center_x, center_y + cross_size), (255, 255, 255), 3)
        
        return center_x, center_y

    def draw_detections(self, frame, results):
        """Draw detection boxes and labels on frame."""
        # Draw camera center lines first
        camera_center_x, camera_center_y = self.draw_camera_center_lines(frame)
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            
            for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                if score >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Calculate object center
                    obj_center_x = (x1 + x2) // 2
                    obj_center_y = (y1 + y2) // 2
                    
                    # Check if object center aligns with camera center (within tolerance)
                    tolerance = 30  # pixels tolerance for alignment
                    x_aligned = abs(obj_center_x - camera_center_x) <= tolerance
                    y_aligned = abs(obj_center_y - camera_center_y) <= tolerance
                    is_aligned = x_aligned and y_aligned
                    
                    # Choose colors based on alignment
                    if is_aligned:
                        box_color = (0, 255, 0)  # Green for aligned
                        line_color = (0, 255, 0)  # Green for aligned
                        point_color = (0, 255, 0)  # Green for aligned
                        alignment_text = "ALIGNED!"
                        text_color = (0, 255, 0)
                    else:
                        box_color = (0, 0, 255)  # Red for not aligned
                        line_color = (255, 0, 0)  # Blue for not aligned
                        point_color = (255, 0, 0)  # Blue for not aligned
                        alignment_text = f"X:{obj_center_x-camera_center_x:+d} Y:{obj_center_y-camera_center_y:+d}"
                        text_color = (0, 0, 255)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                    
                    # Draw center line (vertical line through the center of the bounding box)
                    cv2.line(frame, (obj_center_x, y1), (obj_center_x, y2), line_color, 3)
                    
                    # Draw center point
                    cv2.circle(frame, (obj_center_x, obj_center_y), 8, point_color, -1)
                    cv2.circle(frame, (obj_center_x, obj_center_y), 8, (255, 255, 255), 2)
                    
                    # Draw alignment line from camera center to object center
                    cv2.line(frame, (camera_center_x, camera_center_y), (obj_center_x, obj_center_y), (255, 255, 0), 2)
                    
                    # Draw label with confidence and alignment info
                    label = f"Root {i+1}: {score:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), box_color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Draw alignment status
                    alignment_size = cv2.getTextSize(alignment_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.putText(frame, alignment_text, (x1, y2 + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        return frame
    
    def draw_info_panel(self, frame, num_detections, aligned_count=0):
        """Draw information panel on frame."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text information
        cv2.putText(frame, f"Root Detection Live", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Roots Found: {num_detections}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Aligned: {aligned_count}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {self.current_fps}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {self.confidence_threshold:.1f}", (20, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_webcam(self, camera_index=0):
        """Process live video from webcam."""
        print(f"Starting webcam detection (camera {camera_index})...")
        print("Press 'q' to quit, '+' to increase confidence, '-' to decrease confidence")
        print("Press 'm' to toggle mirror mode")
        
        # Initialize webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Webcam initialized successfully!")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Flip frame horizontally if mirror mode is enabled
                if self.mirror_camera:
                    frame = cv2.flip(frame, 1)
                
                # Run inference
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Count detections and aligned objects
                num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
                aligned_count = 0
                
                if results[0].boxes is not None:
                    height, width = frame.shape[:2]
                    camera_center_x = width // 2
                    camera_center_y = height // 2
                    tolerance = 30
                    
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    scores = results[0].boxes.conf.cpu().numpy()
                    
                    for box, score in zip(boxes, scores):
                        if score >= self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box)
                            obj_center_x = (x1 + x2) // 2
                            obj_center_y = (y1 + y2) // 2
                            
                            x_aligned = abs(obj_center_x - camera_center_x) <= tolerance
                            y_aligned = abs(obj_center_y - camera_center_y) <= tolerance
                            if x_aligned and y_aligned:
                                aligned_count += 1
                
                # Draw detections
                frame = self.draw_detections(frame, results)
                
                # Draw info panel
                frame = self.draw_info_panel(frame, num_detections, aligned_count)
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow('Root Detection - Live', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+') or key == ord('='):
                    self.confidence_threshold = min(0.9, self.confidence_threshold + 0.1)
                    print(f"Confidence threshold: {self.confidence_threshold:.1f}")
                elif key == ord('-'):
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.1)
                    print(f"Confidence threshold: {self.confidence_threshold:.1f}")
                elif key == ord('m'):
                    self.mirror_camera = not self.mirror_camera
                    print(f"Mirror mode: {'ON' if self.mirror_camera else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\nStopping detection...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return True
    
    def process_video_file(self, video_path, output_path=None):
        """Process video file and optionally save output."""
        print(f"Processing video file: {video_path}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Initialize video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run inference
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Count detections
                num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
                
                # Draw detections
                frame = self.draw_detections(frame, results)
                
                # Draw info panel
                frame = self.draw_info_panel(frame, num_detections)
                
                # Add progress bar
                progress = frame_count / total_frames
                cv2.rectangle(frame, (10, height - 30), (width - 10, height - 10), (100, 100, 100), -1)
                cv2.rectangle(frame, (10, height - 30), (int(10 + (width - 20) * progress), height - 10), (0, 255, 0), -1)
                cv2.putText(frame, f"Progress: {progress*100:.1f}%", (20, height - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Write frame to output video
                if out:
                    out.write(frame)
                
                # Display frame (optional)
                cv2.imshow('Root Detection - Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
        
        except KeyboardInterrupt:
            print("\nStopping video processing...")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
        
        print(f"Video processing completed! Processed {frame_count} frames.")
        return True
    
    def benchmark_live_performance(self, duration=30):
        """Benchmark live detection performance."""
        print(f"Benchmarking live performance for {duration} seconds...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera for benchmarking")
            return
        
        start_time = time.time()
        frame_count = 0
        total_inference_time = 0
        
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference and measure time
                inference_start = time.time()
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                inference_time = time.time() - inference_start
                
                total_inference_time += inference_time
                frame_count += 1
                
                # Display frame with FPS
                cv2.putText(frame, f"Benchmarking... {int(duration - (time.time() - start_time))}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Benchmark', frame)
                cv2.waitKey(1)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Calculate performance metrics
        avg_fps = frame_count / duration
        avg_inference_time = total_inference_time / frame_count
        
        print(f"\nBenchmark Results:")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Average inference time: {avg_inference_time*1000:.1f}ms")
        print(f"  Total frames processed: {frame_count}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Live Root Detection with YOLOv8")
    parser.add_argument("--model", type=str, default="output/root_detection/weights/best.pt", 
                       help="Path to trained model")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera index (0 for default webcam)")
    parser.add_argument("--video", type=str, 
                       help="Path to input video file")
    parser.add_argument("--output", type=str, 
                       help="Path to save output video")
    parser.add_argument("--confidence", type=float, default=0.5, 
                       help="Confidence threshold")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Run performance benchmark")
    parser.add_argument("--no-mirror", action="store_true", 
                       help="Disable mirror mode (show true camera view)")
    
    args = parser.parse_args()
    
    print("Live Root Detection with YOLOv8")
    print("=" * 50)
    
    try:
        # Create detector
        mirror_mode = not args.no_mirror  # Default to mirror mode unless --no-mirror is specified
        detector = LiveRootDetector(args.model, mirror_camera=mirror_mode)
        detector.confidence_threshold = args.confidence
        detector.load_model()
        
        if args.benchmark:
            # Run benchmark
            detector.benchmark_live_performance()
        
        elif args.video:
            # Process video file
            detector.process_video_file(args.video, args.output)
        
        else:
            # Process webcam
            detector.process_webcam(args.camera)
        
        print("Live detection completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
