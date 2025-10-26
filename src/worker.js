/**
 * Cloudflare Worker for AI Root Detection
 * Runs YOLOv8n model at the edge for fast inference
 */

import { YOLOv8 } from './yolo-worker.js';

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    // Handle preflight requests
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // Health check endpoint
    if (url.pathname === '/api/health') {
      return new Response(JSON.stringify({ 
        status: 'healthy', 
        timestamp: new Date().toISOString(),
        model: 'YOLOv8n'
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    // Model info endpoint
    if (url.pathname === '/api/model-info') {
      return new Response(JSON.stringify({
        model: 'YOLOv8n',
        version: '8.0.196',
        classes: 80,
        input_size: [640, 640],
        confidence_threshold: 0.1
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    // Detection endpoint
    if (url.pathname === '/api/detect' && request.method === 'POST') {
      try {
        const body = await request.json();
        const { image_data, confidence = 0.1 } = body;

        if (!image_data) {
          return new Response(JSON.stringify({ 
            success: false, 
            error: 'No image data provided' 
          }), {
            status: 400,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          });
        }

        // Initialize YOLO model (cached across requests)
        const yolo = new YOLOv8();
        await yolo.loadModel();

        // Run detection
        const result = await yolo.detect(image_data, confidence);

        return new Response(JSON.stringify(result), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });

      } catch (error) {
        console.error('Detection error:', error);
        return new Response(JSON.stringify({ 
          success: false, 
          error: error.message 
        }), {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      }
    }

    // 404 for other routes
    return new Response('Not Found', { 
      status: 404,
      headers: corsHeaders
    });
  },
};
