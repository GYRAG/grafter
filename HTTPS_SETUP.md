# HTTPS Setup for Camera Access

## Why HTTPS is Required

Modern browsers require HTTPS (or localhost) for camera access due to security policies. This is a browser security feature, not a limitation of our application.

## Solutions for Hosted Websites

### Option 1: Use a Hosting Service with HTTPS (Recommended)

**Vercel (Free):**
1. Push your code to GitHub
2. Connect GitHub repo to Vercel
3. Deploy automatically with HTTPS

**Netlify (Free):**
1. Push your code to GitHub
2. Connect GitHub repo to Netlify
3. Deploy automatically with HTTPS

**GitHub Pages (Free):**
1. Push your code to GitHub
2. Enable GitHub Pages in repository settings
3. Use HTTPS URL provided

### Option 2: Local Development with HTTPS

**Using Vite with HTTPS:**
```bash
# Install mkcert for local HTTPS certificates
npm install -g mkcert

# Create local CA
mkcert -install

# Generate certificate
mkcert localhost 127.0.0.1 ::1

# Update vite.config.ts
```

**Updated vite.config.ts:**
```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'

export default defineConfig({
  plugins: [react()],
  server: {
    https: {
      key: fs.readFileSync('localhost-key.pem'),
      cert: fs.readFileSync('localhost.pem'),
    },
    port: 5173
  }
})
```

### Option 3: Use ngrok for HTTPS Tunneling

```bash
# Install ngrok
npm install -g ngrok

# Start your development server
npm run dev

# In another terminal, create HTTPS tunnel
ngrok http 5173
```

## Testing Camera Access

1. **Local Development:** Use `https://localhost:5173` or `http://localhost:5173`
2. **Hosted Website:** Use the HTTPS URL provided by your hosting service
3. **Check Browser Console:** Look for camera permission prompts and errors

## Browser Permissions

### Chrome/Edge:
- Click the camera icon in the address bar
- Allow camera access
- Refresh the page if needed

### Firefox:
- Click the camera icon in the address bar
- Allow camera access
- Refresh the page if needed

### Safari:
- Go to Safari > Preferences > Websites > Camera
- Allow camera access for your domain

## Troubleshooting

### "Camera access not supported"
- Ensure you're using HTTPS or localhost
- Check if your browser supports getUserMedia

### "Camera permission denied"
- Click the camera icon in the address bar
- Allow camera access
- Refresh the page

### "No camera found"
- Check if a camera is connected
- Ensure no other application is using the camera
- Try refreshing the page

### "Camera is already in use"
- Close other applications using the camera
- Restart your browser
- Try refreshing the page
