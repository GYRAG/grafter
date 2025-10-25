#!/usr/bin/env python3
"""
Flask API server for live root detection.
Provides WebSocket and REST endpoints for real-time detection.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import base64
import time
import json
import os
from root_detector import RootDetectorAPI

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'root_detection_secret_key'
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# Initialize root detector
detector = RootDetectorAPI()

# Load model on startup
def load_model():
    """Load the YOLOv8 model when the server starts."""
    success = detector.load_model()
    if not success:
        print("Warning: Failed to load model. Some endpoints may not work.")

# REST API Endpoints

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_info = detector.get_model_info()
    return jsonify({
        "status": "healthy",
        "model": model_info,
        "timestamp": time.time()
    })

@app.route('/api/detect', methods=['POST'])
def detect_roots():
    """Single image detection endpoint."""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No image data provided"
            }), 400
        
        # Set confidence threshold if provided
        if 'confidence' in data:
            detector.set_confidence_threshold(data['confidence'])
        
        # Run detection
        result = detector.detect_roots_in_image(data['image'])
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    return jsonify(detector.get_model_info())

@app.route('/api/model/confidence', methods=['POST'])
def set_confidence():
    """Set confidence threshold."""
    try:
        data = request.get_json()
        if 'threshold' not in data:
            return jsonify({"error": "No threshold provided"}), 400
        
        detector.set_confidence_threshold(data['threshold'])
        return jsonify({
            "success": True,
            "confidence_threshold": detector.confidence_threshold
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WebSocket Events

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected: {request.sid}")
    emit('connected', {
        'message': 'Connected to root detection server',
        'model_info': detector.get_model_info()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"Client disconnected: {request.sid}")

@socketio.on('detect_frame')
def handle_detect_frame(data):
    """Handle real-time frame detection via WebSocket."""
    try:
        # Set confidence threshold if provided
        if 'confidence' in data:
            detector.set_confidence_threshold(data['confidence'])
        
        # Run detection
        result = detector.detect_roots_in_image(data['image'])
        
        # Add client session info
        result['session_id'] = request.sid
        
        # Emit results back to client
        emit('detection_result', result)
        
    except Exception as e:
        emit('detection_error', {
            'error': str(e),
            'session_id': request.sid
        })

@socketio.on('set_confidence')
def handle_set_confidence(data):
    """Handle confidence threshold change via WebSocket."""
    try:
        if 'threshold' in data:
            detector.set_confidence_threshold(data['threshold'])
            emit('confidence_updated', {
                'confidence_threshold': detector.confidence_threshold,
                'session_id': request.sid
            })
    except Exception as e:
        emit('error', {'error': str(e)})

@socketio.on('ping')
def handle_ping():
    """Handle ping for connection testing."""
    emit('pong', {'timestamp': time.time()})

# Error handlers

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model before starting server
    print("Starting Root Detection API Server...")
    print("Loading YOLOv8 model...")
    
    success = load_model()
    if success:
        print("✅ Model loaded successfully!")
        print("🚀 Starting server on http://localhost:5000")
        print("📡 WebSocket available at ws://localhost:5000")
    else:
        print("❌ Failed to load model. Server starting anyway...")
    
    # Start server
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=True,
        allow_unsafe_werkzeug=True,
        async_mode='gevent'
    )
