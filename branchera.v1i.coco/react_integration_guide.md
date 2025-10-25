# React Website Integration for Root Detection

## Overview
This guide shows how to integrate your trained YOLOv8 root detection model into a React website with live camera detection.

## Approach Options

### 1. **Client-Side Integration (Recommended)**
- Run detection directly in the browser
- Real-time processing
- No server required
- Uses TensorFlow.js or ONNX.js

### 2. **Server-Side Integration**
- Send images to Python backend
- Process with your trained model
- Return detection results
- More accurate but requires server

### 3. **Hybrid Approach**
- Lightweight detection in browser
- Heavy processing on server
- Best of both worlds

## Implementation: Client-Side with TensorFlow.js

### Step 1: Convert YOLOv8 Model to TensorFlow.js

First, convert your trained model to TensorFlow.js format:

```python
# convert_model_to_tfjs.py
from ultralytics import YOLO
import tensorflow as tf

# Load your trained model
model = YOLO('output/root_detection/weights/best.pt')

# Export to TensorFlow.js format
model.export(format='tfjs', imgsz=640)
```

### Step 2: React Component Structure

```
src/
├── components/
│   ├── RootDetector.jsx
│   ├── CameraFeed.jsx
│   └── DetectionResults.jsx
├── utils/
│   ├── modelLoader.js
│   └── detectionUtils.js
└── models/
    └── root_detection_model/
        ├── model.json
        └── *.bin files
```

### Step 3: Main Detection Component

```jsx
// components/RootDetector.jsx
import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';

const RootDetector = () => {
  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [detections, setDetections] = useState([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  // Load TensorFlow.js model
  useEffect(() => {
    loadModel();
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const loadModel = async () => {
    try {
      setIsLoading(true);
      const loadedModel = await tf.loadLayersModel('/models/root_detection_model/model.json');
      setModel(loadedModel);
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setCameraEnabled(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      setCameraEnabled(false);
    }
  };

  const detectRoots = async () => {
    if (!model || !videoRef.current || !canvasRef.current) return;

    setIsDetecting(true);
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Draw video frame to canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Get image data
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Preprocess image for model
    const tensor = tf.browser.fromPixels(imageData)
      .resizeNearestNeighbor([640, 640])
      .expandDims(0)
      .div(255.0);

    // Run detection
    const predictions = await model.predict(tensor);
    
    // Process predictions
    const detections = processPredictions(predictions, canvas.width, canvas.height);
    setDetections(detections);

    // Draw detections on canvas
    drawDetections(ctx, detections);

    // Clean up
    tensor.dispose();
    predictions.forEach(pred => pred.dispose());
    
    setIsDetecting(false);
  };

  const processPredictions = (predictions, width, height) => {
    // Convert model predictions to detection format
    // This depends on your specific model output format
    const detections = [];
    
    // Example processing (adjust based on your model output)
    const boxes = predictions[0].dataSync();
    const scores = predictions[1].dataSync();
    
    for (let i = 0; i < boxes.length; i += 4) {
      if (scores[i / 4] > 0.5) { // Confidence threshold
        detections.push({
          x: boxes[i] * width,
          y: boxes[i + 1] * height,
          width: boxes[i + 2] * width,
          height: boxes[i + 3] * height,
          confidence: scores[i / 4]
        });
      }
    }
    
    return detections;
  };

  const drawDetections = (ctx, detections) => {
    // Clear canvas
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    // Draw video frame
    ctx.drawImage(videoRef.current, 0, 0);
    
    // Draw center line
    const centerX = ctx.canvas.width / 2;
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, ctx.canvas.height);
    ctx.stroke();
    
    // Draw detections
    detections.forEach((detection, index) => {
      const { x, y, width, height, confidence } = detection;
      
      // Check alignment with center
      const objCenterX = x + width / 2;
      const isAligned = Math.abs(objCenterX - centerX) < 30;
      
      // Choose colors based on alignment
      const boxColor = isAligned ? '#00ff00' : '#ff0000';
      const lineColor = isAligned ? '#00ff00' : '#0000ff';
      
      // Draw bounding box
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);
      
      // Draw center line
      ctx.strokeStyle = lineColor;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(objCenterX, y);
      ctx.lineTo(objCenterX, y + height);
      ctx.stroke();
      
      // Draw center point
      const centerY = y + height / 2;
      ctx.fillStyle = lineColor;
      ctx.beginPath();
      ctx.arc(objCenterX, centerY, 8, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw label
      ctx.fillStyle = boxColor;
      ctx.font = '16px Arial';
      ctx.fillText(`Root ${index + 1}: ${confidence.toFixed(2)}`, x, y - 5);
      
      // Draw alignment status
      const alignmentText = isAligned ? 'ALIGNED!' : `X: ${objCenterX - centerX:+d}`;
      ctx.fillStyle = isAligned ? '#00ff00' : '#ff0000';
      ctx.font = '14px Arial';
      ctx.fillText(alignmentText, x, y + height + 20);
    });
  };

  return (
    <div className="root-detector">
      <h1>Root Detection System</h1>
      
      {isLoading && <p>Loading model...</p>}
      
      <div className="camera-section">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ display: 'none' }}
        />
        
        <canvas
          ref={canvasRef}
          style={{
            border: '2px solid #333',
            maxWidth: '100%',
            height: 'auto'
          }}
        />
        
        <div className="controls">
          {!cameraEnabled ? (
            <button onClick={startCamera} disabled={isLoading}>
              Start Camera
            </button>
          ) : (
            <>
              <button onClick={stopCamera}>Stop Camera</button>
              <button 
                onClick={detectRoots} 
                disabled={isDetecting}
              >
                {isDetecting ? 'Detecting...' : 'Detect Roots'}
              </button>
            </>
          )}
        </div>
      </div>
      
      <div className="results">
        <h3>Detection Results</h3>
        <p>Roots Found: {detections.length}</p>
        <p>Aligned: {detections.filter(d => Math.abs(d.x + d.width/2 - canvasRef.current?.width/2) < 30).length}</p>
      </div>
    </div>
  );
};

export default RootDetector;
```

### Step 4: Package.json Dependencies

```json
{
  "dependencies": {
    "@tensorflow/tfjs": "^4.15.0",
    "@tensorflow/tfjs-backend-webgl": "^4.15.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  }
}
```

### Step 5: CSS Styling

```css
/* styles.css */
.root-detector {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  font-family: Arial, sans-serif;
}

.camera-section {
  text-align: center;
  margin: 20px 0;
}

.controls {
  margin: 20px 0;
}

.controls button {
  margin: 0 10px;
  padding: 10px 20px;
  font-size: 16px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  background-color: #007bff;
  color: white;
}

.controls button:hover {
  background-color: #0056b3;
}

.controls button:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
}

.results {
  background-color: #f8f9fa;
  padding: 20px;
  border-radius: 5px;
  margin-top: 20px;
}
```

## Alternative: Server-Side Integration

If you prefer server-side processing:

### Backend API (Flask/FastAPI)

```python
# app.py
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)
model = YOLO('output/root_detection/weights/best.pt')

@app.route('/detect', methods=['POST'])
def detect_roots():
    try:
        # Get image from request
        image_data = request.json['image']
        image_bytes = base64.b64decode(image_data.split(',')[1])
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run detection
        results = model(image)
        
        # Process results
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    detections.append({
                        'x': float(x1),
                        'y': float(y1),
                        'width': float(x2 - x1),
                        'height': float(y2 - y1),
                        'confidence': float(confidence)
                    })
        
        return jsonify({'detections': detections})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### React Component for Server Integration

```jsx
// components/ServerRootDetector.jsx
import React, { useState, useRef } from 'react';

const ServerRootDetector = () => {
  const [detections, setDetections] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const captureAndDetect = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    setIsLoading(true);
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Capture frame
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.drawImage(videoRef.current, 0, 0);
    
    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg');
    
    try {
      const response = await fetch('http://localhost:5000/detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });
      
      const result = await response.json();
      setDetections(result.detections);
      
      // Draw results
      drawDetections(ctx, result.detections);
      
    } catch (error) {
      console.error('Detection error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const drawDetections = (ctx, detections) => {
    // Clear and redraw video
    ctx.drawImage(videoRef.current, 0, 0);
    
    // Draw center line
    const centerX = ctx.canvas.width / 2;
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, ctx.canvas.height);
    ctx.stroke();
    
    // Draw detections (same as client-side version)
    detections.forEach((detection, index) => {
      // ... detection drawing code ...
    });
  };

  return (
    <div className="server-root-detector">
      {/* Similar JSX structure as client-side version */}
    </div>
  );
};

export default ServerRootDetector;
```

## Deployment Options

### 1. **Static Hosting (Client-Side)**
- Netlify, Vercel, GitHub Pages
- No server required
- Model files served from CDN

### 2. **Full-Stack Hosting**
- Heroku, Railway, DigitalOcean
- React frontend + Python backend
- More accurate detection

### 3. **Docker Deployment**
```dockerfile
# Dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## Performance Considerations

- **Model Size**: YOLOv8n is ~6MB, good for web
- **Inference Speed**: 30-50 FPS on modern devices
- **Memory Usage**: ~100-200MB for model + processing
- **Browser Support**: Chrome, Firefox, Safari (WebGL required)

## Next Steps

1. Choose client-side or server-side approach
2. Convert your model to TensorFlow.js (if client-side)
3. Set up React project with dependencies
4. Implement detection component
5. Add styling and user interface
6. Test and optimize performance
7. Deploy to hosting platform

Would you like me to help you implement any specific part of this integration?
