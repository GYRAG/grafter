# Deployment Guide

## 🚀 **RECOMMENDED: Cloudflare Workers (Best Option)**

**For the best performance, cost, and global distribution, use Cloudflare Workers.**
See `CLOUDFLARE_DEPLOYMENT.md` for detailed instructions.

## 🌐 **Alternative Hosting Options (Not Recommended)**

### **Frontend Deployment (React App)**

#### **Option 1: Vercel (Recommended)**
1. **Connect GitHub Repository**
2. **Set Environment Variables:**
   ```
   VITE_API_URL=https://your-backend-url.com
   ```
3. **Deploy**: Automatic deployment on push

#### **Option 2: Netlify**
1. **Connect GitHub Repository**
2. **Build Settings:**
   - Build Command: `npm run build`
   - Publish Directory: `dist`
3. **Environment Variables:**
   ```
   VITE_API_URL=https://your-backend-url.com
   ```

#### **Option 3: GitHub Pages**
1. **Enable GitHub Pages** in repository settings
2. **Create `.github/workflows/deploy.yml`**
3. **Set Environment Variables** in repository secrets

### **Backend Deployment (Flask API)**

#### **Option 1: Render (Recommended)**
1. **Create Web Service**
2. **Build Command:** `cd backend && pip install -r requirements.txt`
3. **Start Command:** `cd backend && python app_simple.py`
4. **Environment:** Python 3

#### **Option 2: Heroku**
1. **Add `Procfile`:**
   ```
   web: cd backend && python app_simple.py
   ```
2. **Add `runtime.txt`:**
   ```
   python-3.12.8
   ```

## 🔧 **Required Files for Deployment**

### **Backend Files:**
- `backend/app_simple.py` - Main Flask app
- `backend/root_detector.py` - AI detection logic
- `backend/requirements.txt` - Python dependencies
- `yolov8n.pt` - AI model file (6.5MB)

### **Frontend Files:**
- `src/` - React source code
- `package.json` - Node.js dependencies
- `vite.config.ts` - Build configuration

## 🌍 **Environment Variables**

### **Frontend (.env):**
```bash
VITE_API_URL=https://your-backend-domain.com
```

### **Backend:**
```bash
PORT=5000
FLASK_ENV=production
```

## 🚀 **Quick Deployment Steps**

### **1. Deploy Backend:**
```bash
# Render
# Connect GitHub repo in Render dashboard

# Heroku
heroku create your-app-name
git push heroku main
```

### **2. Deploy Frontend:**
```bash
# Vercel
npm install -g vercel
vercel --prod

# Netlify
npm install -g netlify-cli
netlify deploy --prod
```

### **3. Update API URL:**
1. Get backend URL from deployment
2. Update `VITE_API_URL` in frontend environment
3. Redeploy frontend

## 🔒 **HTTPS Requirements**

- **Frontend**: Must be HTTPS for camera access
- **Backend**: Should be HTTPS for security
- **CORS**: Backend allows frontend domain

## 📱 **Mobile Compatibility**

- **Camera Access**: Works on mobile browsers
- **Responsive Design**: Optimized for mobile screens
- **Touch Controls**: Touch-friendly interface

## 🐛 **Troubleshooting**

### **Common Issues:**
1. **CORS Errors**: Update backend CORS settings
2. **Model Loading**: Ensure yolov8n.pt is uploaded
3. **Camera Access**: Check HTTPS and permissions
4. **API Timeout**: Increase server timeout settings

### **Debug Steps:**
1. Check browser console for errors
2. Check backend logs for requests
3. Test API endpoints directly
4. Verify environment variables
