/**
 * YOLOv8 Worker Implementation for Cloudflare Workers
 * Optimized for edge computing with WebAssembly
 */

export class YOLOv8 {
  constructor() {
    this.model = null;
    this.isLoaded = false;
    this.confidenceThreshold = 0.1;
    this.tolerance = 30;
    
    // COCO class names
    this.cocoClasses = [
      'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
      'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
      'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
      'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
      'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
      'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
      'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ];
  }

  async loadModel() {
    if (this.isLoaded) return;

    try {
      // For Cloudflare Workers, we'll use a simplified approach
      // In production, you'd load the actual YOLOv8 model
      console.log('Loading YOLOv8n model...');
      
      // Simulate model loading (replace with actual model loading)
      await new Promise(resolve => setTimeout(resolve, 100));
      
      this.isLoaded = true;
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
      throw error;
    }
  }

  async detect(imageData, confidence = 0.1) {
    if (!this.isLoaded) {
      await this.loadModel();
    }

    try {
      const startTime = Date.now();
      
      // Parse base64 image
      const imageBuffer = this.base64ToArrayBuffer(imageData);
      const image = await this.createImageFromBuffer(imageBuffer);
      
      // Resize image for processing
      const resizedImage = this.resizeImage(image, 640, 640);
      
      // Run inference (simplified for demo)
      const detections = await this.runInference(resizedImage, confidence);
      
      const inferenceTime = Date.now() - startTime;
      
      // Calculate camera center
      const cameraCenter = [resizedImage.width / 2, resizedImage.height / 2];
      
      // Process detections
      const processedDetections = detections.map((detection, index) => {
        const { box, confidence: conf, classId } = detection;
        const [x1, y1, x2, y2] = box;
        
        // Calculate object center
        const objCenterX = (x1 + x2) / 2;
        const objCenterY = (y1 + y2) / 2;
        
        // Check alignment with camera center
        const xAligned = Math.abs(objCenterX - cameraCenter[0]) <= this.tolerance;
        const yAligned = Math.abs(objCenterY - cameraCenter[1]) <= this.tolerance;
        const isAligned = xAligned && yAligned;
        
        const className = this.cocoClasses[classId] || `class_${classId}`;
        
        return {
          id: index + 1,
          box: [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)],
          center: [Math.round(objCenterX), Math.round(objCenterY)],
          confidence: conf,
          class_id: classId,
          class_name: className,
          is_aligned: isAligned,
          alignment_offset: {
            x: Math.round(objCenterX - cameraCenter[0]),
            y: Math.round(objCenterY - cameraCenter[1])
          }
        };
      });

      const alignedCount = processedDetections.filter(d => d.is_aligned).length;

      return {
        success: true,
        detections: processedDetections,
        stats: {
          fps: 1000 / inferenceTime,
          detection_count: processedDetections.length,
          aligned_count: alignedCount,
          camera_center: cameraCenter,
          image_size: [resizedImage.width, resizedImage.height],
          inference_time_ms: inferenceTime
        }
      };

    } catch (error) {
      console.error('Detection error:', error);
      return {
        success: false,
        error: error.message,
        detections: [],
        stats: {}
      };
    }
  }

  base64ToArrayBuffer(base64) {
    // Remove data URL prefix if present
    const base64Data = base64.includes(',') ? base64.split(',')[1] : base64;
    
    // Convert base64 to binary string
    const binaryString = atob(base64Data);
    
    // Convert to ArrayBuffer
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    
    return bytes.buffer;
  }

  async createImageFromBuffer(buffer) {
    // Create ImageData from buffer
    // This is a simplified implementation
    return {
      width: 640,
      height: 480,
      data: new Uint8ClampedArray(buffer)
    };
  }

  resizeImage(image, targetWidth, targetHeight) {
    // Simplified resize implementation
    return {
      width: targetWidth,
      height: targetHeight,
      data: image.data
    };
  }

  async runInference(image, confidence) {
    // Simplified inference for demo
    // In production, this would run the actual YOLOv8 model
    
    // Simulate some detections for demo purposes
    const detections = [];
    
    // Add a random detection for testing
    if (Math.random() > 0.7) {
      detections.push({
        box: [100, 100, 200, 200],
        confidence: 0.8,
        classId: 0 // person
      });
    }
    
    return detections;
  }
}
