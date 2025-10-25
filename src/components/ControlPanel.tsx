import React from 'react';

interface ControlPanelProps {
  isStreaming: boolean;
  confidence: number;
  mirrorMode: boolean;
  detections: any[];
  alignedCount: number;
  fps: number;
  isConnected: boolean;
  isLoading?: boolean;
  onStartStop: () => void;
  onConfidenceChange: (value: number) => void;
  onMirrorToggle: () => void;
}

const ControlPanel: React.FC<ControlPanelProps> = ({
  isStreaming,
  confidence,
  mirrorMode,
  detections,
  alignedCount,
  fps,
  isConnected,
  isLoading = false,
  onStartStop,
  onConfidenceChange,
  onMirrorToggle
}) => {
  return (
    <div className="control-panel">
      {/* Connection Status */}
      <div className="control-tile">
        <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
          <div className="connection-indicator"></div>
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
        {isLoading && (
          <div className="processing-indicator">
            <div className="spinner"></div>
            Processing...
          </div>
        )}
      </div>

      {/* Camera Controls */}
      <div className="control-tile">
        <div className="control-title">Camera Control</div>
        <button
          className={`control-button ${isStreaming ? 'stop' : ''}`}
          onClick={onStartStop}
          disabled={!isConnected}
        >
          {isStreaming ? 'Stop Camera' : 'Start Camera'}
        </button>
      </div>

      {/* Mirror Mode Toggle */}
      <div className="control-tile">
        <div className="control-title">Mirror Mode</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={mirrorMode}
              onChange={onMirrorToggle}
            />
            <span className="toggle-slider"></span>
          </label>
          <span style={{ fontSize: '13px', color: 'var(--muted)' }}>
            {mirrorMode ? 'Mirror ON' : 'Mirror OFF'}
          </span>
        </div>
      </div>

      {/* Confidence Threshold */}
      <div className="control-tile">
        <div className="control-title">Confidence Threshold</div>
        <input
          type="range"
          min="0.1"
          max="0.9"
          step="0.05"
          value={confidence}
          onChange={(e) => onConfidenceChange(parseFloat(e.target.value))}
          className="confidence-slider"
        />
        <div className="confidence-value">{confidence.toFixed(2)}</div>
      </div>

      {/* Detection Statistics */}
      <div className="control-tile">
        <div className="control-title">Detection Stats</div>
        <div className="status-panel">
          <div className="status-item">
            <span className="status-label">Roots Found:</span>
            <span className="status-value">{detections.length}</span>
          </div>
          <div className="status-item">
            <span className="status-label">Aligned:</span>
            <span className="status-value aligned">{alignedCount}</span>
          </div>
          <div className="status-item">
            <span className="status-label">FPS:</span>
            <span className="status-value">{fps.toFixed(1)}</span>
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div className="control-tile">
        <div className="control-title">Instructions</div>
        <div style={{ fontSize: '12px', color: 'var(--muted)', lineHeight: '1.4' }}>
          <div>• Position root in camera view</div>
          <div>• Green box = aligned with center</div>
          <div>• Red box = needs adjustment</div>
          <div>• Adjust confidence as needed</div>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
