import React, { useRef, useEffect } from 'react';

interface Detection {
  id: number;
  box: [number, number, number, number]; // [x1, y1, x2, y2]
  center: [number, number];
  confidence: number;
  is_aligned: boolean;
  alignment_offset: {
    x: number;
    y: number;
  };
}

interface DetectionOverlayProps {
  detections: Detection[];
  videoWidth: number;
  videoHeight: number;
  cameraCenter: [number, number];
  tolerance: number;
  mirrorMode: boolean;
}

const DetectionOverlay: React.FC<DetectionOverlayProps> = ({
  detections,
  videoWidth,
  videoHeight,
  cameraCenter,
  tolerance,
  mirrorMode
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const updateCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Get the video element to find its actual display size and position
    const video = canvas.parentElement?.querySelector('video');
    if (!video) return;

    const videoRect = video.getBoundingClientRect();
    const containerRect = canvas.parentElement?.getBoundingClientRect();
    if (!containerRect) return;

    // Calculate video position relative to container
    const videoX = videoRect.left - containerRect.left;
    const videoY = videoRect.top - containerRect.top;
    const videoDisplayWidth = videoRect.width;
    const videoDisplayHeight = videoRect.height;

    // Set canvas size to match container
    canvas.width = containerRect.width;
    canvas.height = containerRect.height;
    canvas.style.width = `${containerRect.width}px`;
    canvas.style.height = `${containerRect.height}px`;

    // Clear canvas
    ctx.clearRect(0, 0, containerRect.width, containerRect.height);

    // Calculate scale factors for video display size
    const scaleX = videoDisplayWidth / videoWidth;
    const scaleY = videoDisplayHeight / videoHeight;

    // Draw only within video bounds
    ctx.save();
    ctx.translate(videoX, videoY);
    ctx.scale(scaleX, scaleY);

    // Draw camera center crosshair
    drawCameraCenter(ctx, videoWidth, videoHeight, cameraCenter);

    // Draw detections
    detections.forEach((detection, index) => {
      drawDetection(ctx, detection, index, mirrorMode);
    });

    ctx.restore();
  };

  useEffect(() => {
    updateCanvas();

    // Add resize observer to handle window resizing
    const resizeObserver = new ResizeObserver(() => {
      updateCanvas();
    });

    const container = canvasRef.current?.parentElement;
    if (container) {
      resizeObserver.observe(container);
      
      // Also observe the video element for size changes
      const video = container.querySelector('video');
      if (video) {
        resizeObserver.observe(video);
      }
    }

    return () => {
      resizeObserver.disconnect();
    };
  }, [detections, videoWidth, videoHeight, cameraCenter, tolerance, mirrorMode]);

  const drawCameraCenter = (
    ctx: CanvasRenderingContext2D,
    _width: number,
    height: number,
    center: [number, number]
  ) => {
    const [centerX, centerY] = center;
    
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.lineWidth = 2;
    ctx.setLineDash([]);

    // Draw vertical center line
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.stroke();

    // Draw center cross
    const crossSize = 20;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - crossSize);
    ctx.lineTo(centerX, centerY + crossSize);
    ctx.stroke();
  };

  const drawDetection = (
    ctx: CanvasRenderingContext2D,
    detection: Detection,
    index: number,
    mirrorMode: boolean
  ) => {
    const [x1, y1, x2, y2] = detection.box;
    const [objCenterX, objCenterY] = detection.center;
    const { confidence, is_aligned, alignment_offset } = detection;

    // Adjust coordinates for mirror mode
    const adjustedX1 = mirrorMode ? videoWidth - x2 : x1;
    const adjustedX2 = mirrorMode ? videoWidth - x1 : x2;
    const adjustedCenterX = mirrorMode ? videoWidth - objCenterX : objCenterX;
    const adjustedOffsetX = mirrorMode ? -alignment_offset.x : alignment_offset.x;

    // Choose colors based on alignment
    const boxColor = is_aligned ? '#73D700' : '#ff4444';
    const lineColor = is_aligned ? '#73D700' : '#4488ff';
    const pointColor = is_aligned ? '#73D700' : '#4488ff';

    // Draw bounding box
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.strokeRect(adjustedX1, y1, adjustedX2 - adjustedX1, y2 - y1);

    // Draw center line (vertical line through the center of the bounding box)
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(adjustedCenterX, y1);
    ctx.lineTo(adjustedCenterX, y2);
    ctx.stroke();

    // Draw center point
    ctx.fillStyle = pointColor;
    ctx.beginPath();
    ctx.arc(adjustedCenterX, objCenterY, 8, 0, 2 * Math.PI);
    ctx.fill();
    
    // Draw white border around center point
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw alignment line from camera center to object center
    ctx.strokeStyle = 'rgba(255, 255, 0, 0.8)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(cameraCenter[0], cameraCenter[1]);
    ctx.lineTo(adjustedCenterX, objCenterY);
    ctx.stroke();

    // Draw label background
    const label = `Root ${index + 1}: ${confidence.toFixed(2)}`;
    const labelMetrics = ctx.measureText(label);
    const labelWidth = labelMetrics.width + 16;
    const labelHeight = 20;
    
    ctx.fillStyle = boxColor;
    ctx.fillRect(adjustedX1, y1 - labelHeight - 5, labelWidth, labelHeight);

    // Draw label text
    ctx.fillStyle = 'white';
    ctx.font = '12px Arial';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, adjustedX1 + 8, y1 - labelHeight/2 - 5);

    // Draw alignment status
    const alignmentText = is_aligned 
      ? 'ALIGNED!' 
      : `X: ${adjustedOffsetX > 0 ? '+' : ''}${adjustedOffsetX} Y: ${alignment_offset.y > 0 ? '+' : ''}${alignment_offset.y}`;
    
    ctx.fillStyle = is_aligned ? '#73D700' : '#ff4444';
    ctx.font = '11px Arial';
    ctx.textBaseline = 'top';
    ctx.fillText(alignmentText, adjustedX1, y2 + 8);
  };

  return (
    <canvas
      ref={canvasRef}
      className="camera-canvas"
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none'
      }}
    />
  );
};

export default DetectionOverlay;
