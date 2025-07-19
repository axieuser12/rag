# 🚀 Deploy RAG File Processor to Railway

## Quick Deploy Steps:

### 1. **Prepare Your Repository**
```bash
# Make sure all files are committed
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

### 2. **Deploy to Railway**
1. Go to [Railway.app](https://railway.app)
2. Click "Deploy from GitHub repo"
3. Select your repository
4. Railway will automatically detect it's a Python app

### 3. **Set Environment Variables**
In Railway dashboard, go to your project → Variables tab and add:

```
OPENAI_API_KEY=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

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key_here
```

### 4. **Deploy!**
Railway will automatically:
- Install Python dependencies
- Start the Flask server
- Provide you with a public URL

## 🌐 **What You Get:**

- **Public URL** - Access your RAG processor from anywhere
- **File Upload Interface** - Beautiful web interface for uploading files
- **API Endpoints**:
  - `GET /health` - Health check
  - `POST /process-files` - Upload and process files
  - `GET /` - Web interface

## 📁 **Supported File Types:**
- TXT files
- PDF documents
- Word documents (.doc/.docx)
- CSV spreadsheets

## 🔧 **Features:**
- **Drag & Drop Upload**
- **Real-time Processing Status**
- **Comprehensive Chunking** - Captures ALL information
- **Automatic Supabase Integration**
- **Vector Embeddings** with OpenAI

## 🛡️ **Security:**
- Environment variables for API keys
- Secure file handling
- CORS enabled for web access
- File type validation

Your RAG system will be live on the web and ready to process files from anywhere! 🎉