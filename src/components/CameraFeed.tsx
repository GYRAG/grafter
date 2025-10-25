import React, { useRef, useEffect, useState } from 'react';

interface CameraFeedProps {
  onFrameCapture: (imageData: string) => void;
  isStreaming: boolean;
  mirrorMode: boolean;
  onVideoReady: (video: HTMLVideoElement) => void;
}

const CameraFeed: React.FC<CameraFeedProps> = ({
  onFrameCapture,
  isStreaming,
  mirrorMode,
  onVideoReady
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [, setVideoDimensions] = useState<{ width: number; height: number }>({ width: 640, height: 480 });
  const [error, setError] = useState<string | null>(null);
  const [permissionStatus, setPermissionStatus] = useState<'checking' | 'granted' | 'denied' | 'prompt'>('checking');

  useEffect(() => {
    if (isStreaming) {
      startCamera();
    } else {
      stopCamera();
    }

    return () => {
      stopCamera();
    };
  }, [isStreaming]);

  useEffect(() => {
    if (videoRef.current) {
      onVideoReady(videoRef.current);
    }
  }, [onVideoReady]);

  const checkCameraPermission = async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError('Camera access not supported in this browser.');
        setPermissionStatus('denied');
        return false;
      }

      // Check if we're on HTTPS or localhost
      const isSecure = window.location.protocol === 'https:' || 
                      window.location.hostname === 'localhost' || 
                      window.location.hostname === '127.0.0.1';
      
      if (!isSecure) {
        setError('Camera access requires HTTPS. Please use https:// or localhost.');
        setPermissionStatus('denied');
        return false;
      }

      // Check permission status
      if (navigator.permissions) {
        const permission = await navigator.permissions.query({ name: 'camera' as PermissionName });
        setPermissionStatus(permission.state);
        
        if (permission.state === 'denied') {
          setError('Camera permission denied. Please enable camera access in your browser settings.');
          return false;
        }
      }

      return true;
    } catch (err) {
      console.error('Error checking camera permission:', err);
      setPermissionStatus('prompt');
      return true; // Try anyway
    }
  };

  const startCamera = async () => {
    try {
      setError(null);
      setPermissionStatus('checking');
      
      // Check permissions first
      const canAccess = await checkCameraPermission();
      if (!canAccess) return;
      
      setPermissionStatus('prompt');
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280, max: 1920 },
          height: { ideal: 720, max: 1080 },
          facingMode: 'user',
          frameRate: { ideal: 30, max: 60 }
        },
        audio: false
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setPermissionStatus('granted');

        // Wait for video to load and get dimensions
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            const { videoWidth, videoHeight } = videoRef.current;
            setVideoDimensions({ width: videoWidth, height: videoHeight });
          }
        };
      }
    } catch (err: any) {
      console.error('Error accessing camera:', err);
      setPermissionStatus('denied');
      
      if (err.name === 'NotAllowedError') {
        setError('Camera permission denied. Please allow camera access and refresh the page.');
      } else if (err.name === 'NotFoundError') {
        setError('No camera found. Please connect a camera and try again.');
      } else if (err.name === 'NotReadableError') {
        setError('Camera is already in use by another application.');
      } else {
        setError(`Failed to access camera: ${err.message}`);
      }
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const captureFrame = () => {
    if (!videoRef.current || !isStreaming) return;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;

    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    onFrameCapture(imageData);
  };

  // Auto-capture frames when streaming
  useEffect(() => {
    let intervalId: number;
    
    if (isStreaming) {
      // Capture frames at ~15 FPS (every 67ms)
      intervalId = setInterval(captureFrame, 67);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isStreaming]);

  const handleRetry = () => {
    setError(null);
    setPermissionStatus('checking');
    startCamera();
  };

  return (
    <div className="camera-container">
      <div className="camera-feed">
        {error ? (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '400px',
            color: '#ff4444',
            textAlign: 'center',
            padding: '20px',
            background: 'rgba(255, 68, 68, 0.1)',
            borderRadius: '12px',
            border: '1px solid rgba(255, 68, 68, 0.3)'
          }}>
            <div>
              <div style={{ fontSize: '48px', marginBottom: '20px' }}>📷</div>
              <div style={{ fontSize: '18px', marginBottom: '10px', fontWeight: 'bold' }}>
                Camera Access Required
              </div>
              <div style={{ marginBottom: '20px', lineHeight: '1.5' }}>
                {error}
              </div>
              {permissionStatus === 'denied' && (
                <div style={{ marginBottom: '20px', fontSize: '14px', color: '#ccc' }}>
                  <div>To fix this:</div>
                  <div>1. Click the camera icon in your browser's address bar</div>
                  <div>2. Allow camera access</div>
                  <div>3. Refresh the page</div>
                </div>
              )}
              <button
                onClick={handleRetry}
                style={{
                  background: 'linear-gradient(135deg, #73D700, #5bb300)',
                  color: 'white',
                  border: 'none',
                  padding: '12px 24px',
                  borderRadius: '8px',
                  fontSize: '16px',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  boxShadow: '0 4px 12px rgba(115, 215, 0, 0.3)',
                  transition: 'all 0.3s ease'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = '0 6px 16px rgba(115, 215, 0, 0.4)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(115, 215, 0, 0.3)';
                }}
              >
                Try Again
              </button>
            </div>
          </div>
        ) : permissionStatus === 'checking' ? (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '400px',
            color: '#73D700',
            textAlign: 'center',
            padding: '20px'
          }}>
            <div>
              <div style={{ fontSize: '48px', marginBottom: '20px' }}>⏳</div>
              <div style={{ fontSize: '18px' }}>Checking camera permissions...</div>
            </div>
          </div>
        ) : (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="camera-video"
            style={{
              transform: mirrorMode ? 'scaleX(-1)' : 'scaleX(1)',
              maxWidth: '100%',
              maxHeight: '100%',
              width: 'auto',
              height: 'auto',
              display: 'block',
              objectFit: 'contain'
            }}
          />
        )}
      </div>
    </div>
  );
};

export default CameraFeed;
