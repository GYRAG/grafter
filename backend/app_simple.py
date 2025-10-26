#!/usr/bin/env python3
"""
Simple Flask API server for root detection (without WebSocket).
Use this if the WebSocket version has compatibility issues.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import time
import json
import os
from root_detector import RootDetectorAPI

# Initialize Flask app
app = Flask(__name__)
# CORS configuration for production
CORS(app, origins=[
    "http://localhost:3000",
    "http://localhost:5173", 
    "https://*.vercel.app",
    "https://*.netlify.app",
    "https://*.github.io"
])

# Initialize root detector
detector = RootDetectorAPI()

# Load model on startup
def load_model():
    """Load the YOLOv8 model when the server starts."""
    success = detector.load_model()
    if not success:
        print("Warning: Failed to load model. Some endpoints may not work.")
    return success

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

# Error handlers

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model before starting server
    print("Starting Simple Root Detection API Server...")
    print("Loading YOLOv8 model...")
    
    success = load_model()
    if success:
        print("SUCCESS: Model loaded successfully!")
        print("Starting server on http://localhost:5000")
        print("REST API available at http://localhost:5000/api/")
    else:
        print("WARNING: Failed to load model. Server starting anyway...")
    
    # Start server
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=True,
        threaded=True
    )
