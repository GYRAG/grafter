import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useRestAPI } from '../hooks/useRestAPI';
import CameraFeed from './CameraFeed';
import DetectionOverlay from './DetectionOverlay';
import '../styles/LiveRootDetector.css';

interface Detection {
  id: number;
  box: [number, number, number, number];
  center: [number, number];
  confidence: number;
  is_aligned: boolean;
  alignment_offset: {
    x: number;
    y: number;
  };
}

const LiveRootDetectorSimple: React.FC = () => {
  // State management
  const [isStreaming, setIsStreaming] = useState(false);
  const [confidence] = useState(0.1);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [videoDimensions, setVideoDimensions] = useState({ width: 640, height: 480 });
  const [cameraCenter, setCameraCenter] = useState<[number, number]>([320, 240]);
  const [tolerance] = useState(30);

  // REST API connection
  const { isConnected, sendFrame, lastResult, error, isLoading } = useRestAPI();

  // Refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const frameIntervalRef = useRef<number | null>(null);
  const isProcessingRef = useRef(false);

  // Handle video ready
  const handleVideoReady = useCallback((video: HTMLVideoElement) => {
    videoRef.current = video;
  }, []);

  // Handle frame capture
  const handleFrameCapture = useCallback(async (imageData: string) => {
    if (isConnected && isStreaming && !isProcessingRef.current) {
      isProcessingRef.current = true;
      try {
        await sendFrame(imageData, confidence);
      } finally {
        isProcessingRef.current = false;
      }
    }
  }, [isConnected, isStreaming, sendFrame, confidence]);

  // Update detections when new results arrive
  useEffect(() => {
    if (lastResult && lastResult.success) {
      setDetections(lastResult.detections);
      
      // Update camera center from server response
      if (lastResult.stats.camera_center) {
        setCameraCenter(lastResult.stats.camera_center);
      }
      
      // Update video dimensions
      if (lastResult.stats.image_size) {
        setVideoDimensions({
          width: lastResult.stats.image_size[0],
          height: lastResult.stats.image_size[1]
        });
      }
    }
  }, [lastResult]);

  // Manual camera control
  const handleStartCamera = () => {
    setIsStreaming(true);
  };

  const handleStopCamera = () => {
    setIsStreaming(false);
  };

  // Auto-capture frames when streaming
  useEffect(() => {
    if (isStreaming && videoRef.current) {
      const captureFrame = () => {
        if (videoRef.current && isStreaming && !isProcessingRef.current) {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          
          if (!ctx) return;

          const video = videoRef.current;
          // Resize canvas for faster processing
          const maxWidth = 640;
          const scale = Math.min(maxWidth / video.videoWidth, 1);
          canvas.width = video.videoWidth * scale;
          canvas.height = video.videoHeight * scale;

          // Draw video frame to canvas
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          // Convert to base64 with lower quality for faster processing
          const imageData = canvas.toDataURL('image/jpeg', 0.6);
          handleFrameCapture(imageData);
        }
      };

      // Capture frames at ~3 FPS (every 333ms) for REST API to reduce lag
      frameIntervalRef.current = setInterval(captureFrame, 333);
    } else {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
    }

    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
      }
    };
  }, [isStreaming, handleFrameCapture]);

  // Removed unused functions



  return (
    <div className="live-root-detector">
      {/* Header */}
      <header className="detector-header">
        <div className="glow"></div>
        <h1>AI Root Detection</h1>
      </header>

      {/* Main Content */}
      <main className="detector-body">
        <div className="camera-section">
          {/* Camera Controls */}
          {!isStreaming && (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '400px',
              gap: '20px'
            }}>
              <div style={{
                fontSize: '48px',
                color: '#73D700'
              }}>📷</div>
              <div style={{
                fontSize: '24px',
                fontWeight: 'bold',
                color: 'white',
                textAlign: 'center'
              }}>
                Start Camera for Root Detection
              </div>
              <div style={{
                fontSize: '16px',
                color: '#ccc',
                textAlign: 'center',
                maxWidth: '400px',
                lineHeight: '1.5'
              }}>
                Click the button below to start your camera and begin detecting plant roots in real-time.
              </div>
              <button
                onClick={handleStartCamera}
                style={{
                  background: 'linear-gradient(135deg, #73D700, #5bb300)',
                  color: 'white',
                  border: 'none',
                  padding: '16px 32px',
                  borderRadius: '12px',
                  fontSize: '18px',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  boxShadow: '0 6px 20px rgba(115, 215, 0, 0.3)',
                  transition: 'all 0.3s ease',
                  minWidth: '200px'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.transform = 'translateY(-3px)';
                  e.currentTarget.style.boxShadow = '0 8px 25px rgba(115, 215, 0, 0.4)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 6px 20px rgba(115, 215, 0, 0.3)';
                }}
              >
                Start Camera
              </button>
            </div>
          )}

          {/* Camera Feed */}
          <div className="camera-wrapper">
            <CameraFeed
              onFrameCapture={() => {}} // We handle frame capture internally
              isStreaming={isStreaming}
              mirrorMode={true}
              onVideoReady={handleVideoReady}
            />
            
            {/* Detection Overlay */}
            {isStreaming && (
              <DetectionOverlay
                detections={detections}
                videoWidth={videoDimensions.width}
                videoHeight={videoDimensions.height}
                cameraCenter={cameraCenter}
                tolerance={tolerance}
                mirrorMode={true}
              />
            )}

            {/* Camera Control Overlay */}
            {isStreaming && (
              <div style={{
                position: 'absolute',
                top: '20px',
                right: '20px',
                zIndex: 20
              }}>
                <button
                  onClick={handleStopCamera}
                  style={{
                    background: 'rgba(0, 0, 0, 0.7)',
                    color: 'white',
                    border: 'none',
                    padding: '12px 16px',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: 'bold',
                    cursor: 'pointer',
                    backdropFilter: 'blur(10px)',
                    transition: 'all 0.3s ease'
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.background = 'rgba(255, 68, 68, 0.8)';
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.background = 'rgba(0, 0, 0, 0.7)';
                  }}
                >
                  Stop Camera
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Loading Indicator */}
        {isLoading && (
          <div className="loading-indicator">
            🔍 Processing detection...
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="error-display">
            <div className="error-title">Connection Error</div>
            <div className="error-message">{error}</div>
          </div>
        )}
      </main>
    </div>
  );
};

export default LiveRootDetectorSimple;
