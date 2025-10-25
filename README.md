# Grafter - AI Root Detection System

A real-time root detection system using YOLOv8 and React with a glassmorphic design interface.

## Features

- **Real-time Detection**: Live camera feed with AI-powered root detection
- **Center Alignment**: Visual indicators showing when roots are aligned with camera center
- **High Accuracy**: 86.7% mAP50 using trained YOLOv8 model
- **Modern UI**: Glassmorphic design with dark theme and green accents
- **WebSocket Communication**: Real-time data streaming between frontend and backend
- **Adjustable Settings**: Confidence threshold, mirror mode, and detection controls

## Architecture

### Backend (Flask + YOLOv8)
- Flask API server with WebSocket support
- YOLOv8 model integration for root detection
- Real-time frame processing and analysis
- Center alignment calculation with tolerance settings

### Frontend (React + TypeScript)
- Modern React components with TypeScript
- Glassmorphic design system
- Real-time camera feed with HTML5 video
- Canvas overlay for detection visualization
- WebSocket client for live communication

## Quick Start

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start the Flask server
python app.py
```

The backend will start on `http://localhost:5000`

### 2. Frontend Setup

```bash
# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```

The frontend will start on `http://localhost:5173`

### 3. Usage

1. Open the web application in your browser
2. Click "Start Camera" to begin live detection
3. Position roots in the camera view
4. Green bounding boxes indicate aligned roots
5. Red bounding boxes indicate roots that need adjustment
6. Adjust confidence threshold as needed

## Model Information

- **Model**: YOLOv8n (nano)
- **Training Data**: 322 images (268 train / 54 test)
- **Performance**: 86.7% mAP50, 70.5% mAP50-95
- **Training Time**: ~19.6 minutes on RTX 4070 Mobile
- **Model Size**: ~6MB

## API Endpoints

### REST API
- `GET /api/health` - Health check
- `POST /api/detect` - Single image detection
- `GET /api/model/info` - Model information
- `POST /api/model/confidence` - Set confidence threshold

### WebSocket Events
- `detect_frame` - Send frame for detection
- `detection_result` - Receive detection results
- `set_confidence` - Update confidence threshold
- `ping/pong` - Connection health check

## File Structure

```
├── backend/
│   ├── app.py                 # Flask server with WebSocket
│   ├── root_detector.py       # YOLOv8 detection service
│   └── requirements.txt       # Python dependencies
├── src/
│   ├── components/
│   │   ├── LiveRootDetector.tsx    # Main component
│   │   ├── CameraFeed.tsx          # Camera handling
│   │   ├── DetectionOverlay.tsx    # Canvas visualization
│   │   └── ControlPanel.tsx        # UI controls
│   ├── hooks/
│   │   └── useWebSocket.ts         # WebSocket hook
│   ├── styles/
│   │   └── LiveRootDetector.css    # Glassmorphic styles
│   └── App.tsx                     # Main app component
└── branchera.v1i.coco/            # Trained model files
    └── output/root_detection/weights/best.pt
```

## Design System

The application uses a glassmorphic design system with:

- **Colors**: Dark theme with green accent (#73D700)
- **Effects**: Backdrop blur, glass morphism, subtle shadows
- **Typography**: Custom fonts (Dachi The Lynx, 123Wave)
- **Layout**: Responsive grid with glassmorphic panels

## Performance

- **Detection Speed**: 15-30 FPS on modern devices
- **Latency**: ~50-100ms inference time
- **Memory Usage**: ~100-200MB for model + processing
- **Browser Support**: Chrome, Firefox, Safari (WebGL required)

## Troubleshooting

### Camera Issues
- Ensure camera permissions are granted
- Try different browsers if camera doesn't work
- Check if camera is being used by another application

### Connection Issues
- Verify backend server is running on port 5000
- Check browser console for WebSocket errors
- Ensure no firewall blocking localhost connections

### Model Issues
- Verify model file exists at correct path
- Check Python dependencies are installed
- Review backend logs for model loading errors

## Development

### Adding New Features
1. Backend: Add new endpoints in `app.py`
2. Frontend: Create new components in `src/components/`
3. Styling: Use existing CSS variables in `LiveRootDetector.css`

### Testing
- Backend: Test API endpoints with curl or Postman
- Frontend: Use browser dev tools for debugging
- Integration: Test WebSocket connection and real-time updates

## License

MIT License - see LICENSE file for details.