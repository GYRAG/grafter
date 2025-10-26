// API Configuration for Cloudflare Workers
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://grafter-ai-detection.grafter.workers.dev';

export const API_ENDPOINTS = {
  DETECT: `${API_BASE_URL}/api/detect`,
  HEALTH: `${API_BASE_URL}/api/health`,
  MODEL_INFO: `${API_BASE_URL}/api/model-info`
};

export default API_BASE_URL;
