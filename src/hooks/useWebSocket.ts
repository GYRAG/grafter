import { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';

interface DetectionResult {
  success: boolean;
  detections: any[];
  stats: {
    total_roots: number;
    aligned_roots: number;
    inference_time_ms: number;
    image_size: [number, number];
    camera_center: [number, number];
    confidence_threshold: number;
    tolerance: number;
  };
  timestamp: number;
}

interface UseWebSocketReturn {
  socket: Socket | null;
  isConnected: boolean;
  sendFrame: (imageData: string, confidence?: number) => void;
  setConfidence: (confidence: number) => void;
  lastResult: DetectionResult | null;
  error: string | null;
}

export const useWebSocket = (url: string = 'http://localhost:5000'): UseWebSocketReturn => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastResult, setLastResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    // Initialize socket connection
    const newSocket = io(url, {
      transports: ['websocket', 'polling'],
      timeout: 10000,
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
      maxReconnectionAttempts: 5
    });

    // Connection event handlers
    newSocket.on('connect', () => {
      console.log('Connected to root detection server');
      setIsConnected(true);
      setError(null);
      
      // Clear any pending reconnection timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    });

    newSocket.on('disconnect', (reason) => {
      console.log('Disconnected from server:', reason);
      setIsConnected(false);
      
      // Attempt reconnection after a delay
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, don't reconnect automatically
        setError('Server disconnected');
      } else {
        // Client-side disconnect, attempt reconnection
        reconnectTimeoutRef.current = setTimeout(() => {
          if (!newSocket.connected) {
            newSocket.connect();
          }
        }, 2000);
      }
    });

    newSocket.on('connect_error', (err) => {
      console.error('Connection error:', err);
      setError(`Connection failed: ${err.message}`);
      setIsConnected(false);
    });

    // Detection result handlers
    newSocket.on('detection_result', (result: DetectionResult) => {
      setLastResult(result);
      setError(null);
    });

    newSocket.on('detection_error', (errorData: { error: string }) => {
      setError(`Detection error: ${errorData.error}`);
    });

    newSocket.on('confidence_updated', (data: { confidence_threshold: number }) => {
      console.log('Confidence threshold updated:', data.confidence_threshold);
    });

    // Ping/pong for connection health
    newSocket.on('pong', (data: { timestamp: number }) => {
      const latency = Date.now() - data.timestamp;
      console.log(`Ping latency: ${latency}ms`);
    });

    setSocket(newSocket);

    // Cleanup function
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      newSocket.disconnect();
    };
  }, [url]);

  const sendFrame = (imageData: string, confidence?: number) => {
    if (socket && isConnected) {
      const data: any = { image: imageData };
      if (confidence !== undefined) {
        data.confidence = confidence;
      }
      socket.emit('detect_frame', data);
    } else {
      console.warn('Socket not connected, cannot send frame');
    }
  };

  const setConfidence = (confidence: number) => {
    if (socket && isConnected) {
      socket.emit('set_confidence', { threshold: confidence });
    }
  };

  // Ping the server periodically to check connection health
  useEffect(() => {
    if (!socket || !isConnected) return;

    const pingInterval = setInterval(() => {
      socket.emit('ping', { timestamp: Date.now() });
    }, 10000); // Ping every 10 seconds

    return () => clearInterval(pingInterval);
  }, [socket, isConnected]);

  return {
    socket,
    isConnected,
    sendFrame,
    setConfidence,
    lastResult,
    error
  };
};
