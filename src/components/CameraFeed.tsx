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
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number }>({ width: 640, height: 480 });
  const [error, setError] = useState<string | null>(null);

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

  const startCamera = async () => {
    try {
      setError(null);
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;

        // Wait for video to load and get dimensions
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            const { videoWidth, videoHeight } = videoRef.current;
            setVideoDimensions({ width: videoWidth, height: videoHeight });
          }
        };
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      setError('Failed to access camera. Please check permissions.');
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
    let intervalId: NodeJS.Timeout;
    
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

  return (
    <div className="camera-container">
      <div className="camera-feed">
        {error ? (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '300px',
            color: '#ff4444',
            textAlign: 'center',
            padding: '20px'
          }}>
            <div>
              <div style={{ fontSize: '24px', marginBottom: '10px' }}>📷</div>
              <div>{error}</div>
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
