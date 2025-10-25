import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import CameraFeed from './CameraFeed';
import DetectionOverlay from './DetectionOverlay';
import ControlPanel from './ControlPanel';
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

const LiveRootDetector: React.FC = () => {
  // State management
  const [isStreaming, setIsStreaming] = useState(false);
  const [confidence, setConfidence] = useState(0.5);
  const [mirrorMode, setMirrorMode] = useState(true);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState(0);
  const [videoDimensions, setVideoDimensions] = useState({ width: 640, height: 480 });
  const [cameraCenter, setCameraCenter] = useState<[number, number]>([320, 240]);
  const [tolerance] = useState(30);

  // WebSocket connection
  const { isConnected, sendFrame, setConfidence: setServerConfidence, lastResult, error } = useWebSocket();

  // Refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fpsCounterRef = useRef(0);
  const fpsStartTimeRef = useRef(Date.now());

  // Handle video ready
  const handleVideoReady = useCallback((video: HTMLVideoElement) => {
    videoRef.current = video;
  }, []);

  // Handle frame capture
  const handleFrameCapture = useCallback((imageData: string) => {
    if (isConnected && isStreaming) {
      sendFrame(imageData, confidence);
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

  // FPS calculation
  useEffect(() => {
    if (isStreaming) {
      const interval = setInterval(() => {
        const now = Date.now();
        const elapsed = (now - fpsStartTimeRef.current) / 1000;
        
        if (elapsed >= 1.0) {
          setFps(fpsCounterRef.current / elapsed);
          fpsCounterRef.current = 0;
          fpsStartTimeRef.current = now;
        }
      }, 1000);

      return () => clearInterval(interval);
    } else {
      setFps(0);
      fpsCounterRef.current = 0;
    }
  }, [isStreaming]);

  // Increment FPS counter when detections are processed
  useEffect(() => {
    if (lastResult && lastResult.success) {
      fpsCounterRef.current++;
    }
  }, [lastResult]);

  // Handle start/stop camera
  const handleStartStop = () => {
    if (isStreaming) {
      setIsStreaming(false);
    } else {
      setIsStreaming(true);
    }
  };

  // Handle confidence change
  const handleConfidenceChange = (newConfidence: number) => {
    setConfidence(newConfidence);
    setServerConfidence(newConfidence);
  };

  // Handle mirror mode toggle
  const handleMirrorToggle = () => {
    setMirrorMode(!mirrorMode);
  };

  // Calculate aligned count
  const alignedCount = detections.filter(detection => detection.is_aligned).length;

  return (
    <div className="live-root-detector">
      {/* Header */}
      <header className="detector-header">
        <div className="glow"></div>
        <h1>AI Root Detection System</h1>
      </header>

      {/* Main Content */}
      <main className="detector-body">
        <div className="camera-section">
          {/* Camera Feed */}
          <div style={{ position: 'relative' }}>
            <CameraFeed
              onFrameCapture={handleFrameCapture}
              isStreaming={isStreaming}
              mirrorMode={mirrorMode}
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
                mirrorMode={mirrorMode}
              />
            )}
          </div>

          {/* Control Panel */}
          <ControlPanel
            isStreaming={isStreaming}
            confidence={confidence}
            mirrorMode={mirrorMode}
            detections={detections}
            alignedCount={alignedCount}
            fps={fps}
            isConnected={isConnected}
            onStartStop={handleStartStop}
            onConfidenceChange={handleConfidenceChange}
            onMirrorToggle={handleMirrorToggle}
          />
        </div>

        {/* Error Display */}
        {error && (
          <div style={{
            position: 'fixed',
            top: '20px',
            right: '20px',
            background: 'rgba(255, 68, 68, 0.9)',
            color: 'white',
            padding: '12px 16px',
            borderRadius: '8px',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 68, 68, 0.3)',
            zIndex: 1000,
            maxWidth: '300px'
          }}>
            <div style={{ fontWeight: '600', marginBottom: '4px' }}>Connection Error</div>
            <div style={{ fontSize: '13px' }}>{error}</div>
          </div>
        )}

        {/* Status Info */}
        {isStreaming && (
          <div style={{
            position: 'fixed',
            bottom: '20px',
            left: '20px',
            background: 'rgba(115, 215, 0, 0.1)',
            border: '1px solid rgba(115, 215, 0, 0.3)',
            borderRadius: '12px',
            padding: '12px 16px',
            backdropFilter: 'blur(10px)',
            color: 'var(--text)',
            fontSize: '13px',
            zIndex: 1000
          }}>
            <div style={{ fontWeight: '600', marginBottom: '4px' }}>Live Detection Active</div>
            <div>Model: YOLOv8n • Accuracy: 86.7% mAP50</div>
          </div>
        )}
      </main>
    </div>
  );
};

export default LiveRootDetector;
