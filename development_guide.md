# 🚀 Development Guide

## Quick Start

### Option 1: Use the Python starter script
```bash
python start_dev.py
```

### Option 2: Use npm script
```bash
npm run start:dev
```

### Option 3: Manual startup
```bash
# Terminal 1: Start Python backend
python web_server.py

# Terminal 2: Start React frontend  
npm run dev
```

## 🔧 How It Works

### Frontend (React + Vite)
- Runs on `http://localhost:5173`
- Proxies API requests to the Python backend
- All `/api/*` requests are forwarded to `http://localhost:8000`

### Backend (Flask)
- Runs on `http://localhost:8000`
- Handles file uploads and processing
- Connects to user's OpenAI and Supabase services

### API Endpoints
- `GET /health` - Health check
- `POST /process-files` - File upload and processing
- `GET /` - Fallback HTML interface

## 🛠️ Development Workflow

1. **Start Development Servers**
   ```bash
   npm run start:dev
   ```

2. **Frontend Development**
   - Edit files in `src/`
   - Hot reload is enabled
   - Access at `http://localhost:5173`

3. **Backend Development**
   - Edit `web_server.py` or `universal_file_processor.py`
   - Restart backend manually or use auto-reload tools

4. **Testing**
   - Upload test files through the web interface
   - Check browser console for frontend errors
   - Check terminal for backend logs

## 🔍 Debugging

### Frontend Issues
- Check browser console for JavaScript errors
- Verify API requests in Network tab
- Ensure Vite proxy is working

### Backend Issues
- Check terminal output for Python errors
- Verify Flask server is running on port 8000
- Test endpoints directly with curl:
  ```bash
  curl http://localhost:8000/health
  ```

### Connection Issues
- Ensure both servers are running
- Check that ports 5173 and 8000 are available
- Verify proxy configuration in `vite.config.ts`

## 📁 Project Structure

```
├── src/                    # React frontend
│   ├── components/         # React components
│   ├── App.tsx            # Main app component
│   └── main.tsx           # Entry point
├── web_server.py          # Flask backend
├── universal_file_processor.py  # File processing logic
├── vite.config.ts         # Vite configuration with proxy
├── package.json           # Node.js dependencies
├── requirements.txt       # Python dependencies
└── start_dev.py          # Development server starter
```

## 🚀 Production Deployment

The app is configured for Railway deployment:
- `Procfile` defines the production start command
- `railway.json` contains deployment configuration
- Frontend is built and served by the Flask backend in production

For local production testing:
```bash
npm run build
python web_server.py
```