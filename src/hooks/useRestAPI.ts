import React, { useState, useCallback } from 'react';

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

interface UseRestAPIReturn {
  isConnected: boolean;
  sendFrame: (imageData: string, confidence?: number) => Promise<DetectionResult | null>;
  setConfidence: (confidence: number) => Promise<boolean>;
  lastResult: DetectionResult | null;
  error: string | null;
  isLoading: boolean;
}

export const useRestAPI = (url: string = import.meta.env.VITE_API_URL || 'https://grafter-ai-detection.grafter.workers.dev'): UseRestAPIReturn => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastResult, setLastResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Check connection health
  const checkConnection = useCallback(async () => {
    try {
      const response = await fetch(`${url}/api/health`);
      await response.json();
      setIsConnected(response.ok);
      return response.ok;
    } catch (err) {
      setIsConnected(false);
      return false;
    }
  }, [url]);

  // Send frame for detection
  const sendFrame = useCallback(async (imageData: string, confidence?: number): Promise<DetectionResult | null> => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${url}/api/detect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData,
          confidence: confidence
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: DetectionResult = await response.json();
      setLastResult(result);
      setIsConnected(true);
      return result;

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      setIsConnected(false);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [url]);

  // Set confidence threshold
  const setConfidence = useCallback(async (confidence: number): Promise<boolean> => {
    try {
      const response = await fetch(`${url}/api/model/confidence`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ threshold: confidence }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setIsConnected(true);
      return true;

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      setIsConnected(false);
      return false;
    }
  }, [url]);

  // Check connection on mount
  React.useEffect(() => {
    checkConnection();
  }, [checkConnection]);

  return {
    isConnected,
    sendFrame,
    setConfidence,
    lastResult,
    error,
    isLoading
  };
};
