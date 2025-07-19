# 🧠 Public RAG File Processor

A secure, production-ready RAG (Retrieval-Augmented Generation) system where users provide their own OpenAI and Supabase credentials to process documents and create searchable knowledge bases.

## 🔒 Security Features

- **Zero Credential Storage**: User credentials are never stored on our servers
- **Session-Only Processing**: Credentials are used only for the current session
- **Direct API Communication**: Your data goes directly to your own OpenAI and Supabase services
- **No Data Retention**: No user data is retained after processing
- **Open Source**: Full transparency of how your credentials are handled

## 🚀 Live Demo

Deploy to Railway with one click:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template-id)

## 🚀 For Users

### Quick Start

1. **Get Your Credentials**:
   - OpenAI API Key: [Get from OpenAI Platform](https://platform.openai.com/api-keys)
   - Supabase Project: [Create at Supabase](https://supabase.com/dashboard)

2. **Setup Your Supabase Database**:
   ```sql
   -- Enable pgvector extension
   CREATE EXTENSION IF NOT EXISTS vector;
   
   -- Create documents table
   CREATE TABLE IF NOT EXISTS documents (
       id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
       content text NOT NULL,
       embedding vector(1536) NOT NULL,
       source text,
       metadata jsonb,
       created_at timestamptz DEFAULT now()
   );
   
   -- Create index for fast similarity search
   CREATE INDEX IF NOT EXISTS idx_documents_embedding
   ON documents
   USING ivfflat (embedding vector_cosine_ops)
   WITH (lists = 100);
   
   -- Additional indexes for performance
   CREATE INDEX IF NOT EXISTS idx_documents_source ON documents (source);
   CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents (created_at);
   ```

3. **Use the System**:
   - Visit the web interface
   - Enter your credentials (they're never stored)
   - Upload your documents
   - Your RAG system is ready!

### Supported File Types
- **Text Files**: .txt
- **PDFs**: .pdf
- **Word Documents**: .doc, .docx
- **CSV Files**: .csv

### Features
- **Smart Text Extraction**: Handles various file formats with error recovery
- **Intelligent Chunking**: Optimized chunk sizes for better embedding quality
- **Comprehensive Processing**: Multiple chunking strategies for maximum coverage
- **Real-time Progress**: Live updates during processing
- **Error Handling**: Detailed error reporting and troubleshooting tips
- **Production Ready**: Optimized for Railway deployment with proper logging

## 🛠️ For Developers

### Local Development

```bash
# Clone the repository
git clone <your-repo>
cd rag-file-processor

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for frontend)
npm install

# Start both backend and frontend in development
npm run start:dev

# Or start them separately:
# Terminal 1: Start Python backend
python web_server.py

# Terminal 2: Start React frontend
npm run dev
```

### Deployment to Railway

1. **Connect Your Repository**:
   - Link your GitHub repository to Railway
   - Railway will auto-detect the Python app

2. **Environment Variables** (Optional):
   - `FLASK_DEBUG=false` (for production)
   - `PORT` (automatically set by Railway)
   - The system uses user-provided credentials

3. **Deploy**:
   - Railway will automatically deploy your app
   - Users can access it via the provided URL

### Production Configuration

The app is configured for production with:
- **Gunicorn WSGI server** with optimized worker settings
- **Comprehensive error handling** and logging
- **File upload limits** and security validation
- **Health check endpoints** for monitoring
- **Automatic cleanup** of temporary files
- **Rate limiting** and timeout protection

### Project Structure

```
├── src/                          # React frontend
│   ├── components/
│   │   ├── CredentialsForm.tsx   # Secure credential input
│   │   ├── FileUploader.tsx      # File upload interface
│   │   ├── ProcessingStatus.tsx  # Real-time status updates
│   │   ├── ResultsDisplay.tsx    # Results and statistics
│   │   └── ...
│   └── App.tsx                   # Main application
├── web_server.py                 # Flask backend
├── universal_file_processor.py   # File processing logic
├── requirements.txt              # Python dependencies
├── Procfile                      # Railway deployment config
├── railway.json                  # Railway settings
└── runtime.txt                   # Python version
```

## 🔧 Architecture

### Security Model
1. **Frontend**: Collects user credentials securely
2. **Backend**: Receives credentials in request, never stores them
3. **Processing**: Uses credentials to connect to user's services
4. **Cleanup**: Credentials are discarded after processing

### Data Flow
```
User Credentials → Temporary Session → User's OpenAI API → User's Supabase DB
                                   ↓
                              No Storage on Our Servers
```

### Performance Optimizations
- **Chunking Strategy**: Optimized chunk sizes (800 chars) for better embeddings
- **Batch Processing**: Efficient file processing with progress tracking
- **Memory Management**: Automatic cleanup of temporary files
- **Error Recovery**: Graceful handling of file processing errors
- **Connection Pooling**: Optimized database connections

## 🛡️ Security Best Practices

### For Users
- **Use API Keys with Minimal Permissions**: Create dedicated API keys for this service
- **Monitor Usage**: Check your OpenAI and Supabase usage regularly
- **Rotate Keys**: Regularly rotate your API keys
- **Dedicated Project**: Consider using a dedicated Supabase project

### For Developers
- **Never Log Credentials**: Ensure credentials are never logged
- **Memory Cleanup**: Clear credentials from memory after use
- **HTTPS Only**: Always use HTTPS in production
- **Input Validation**: Validate all user inputs
- **File Size Limits**: Enforce reasonable file upload limits
- **Timeout Protection**: Prevent long-running requests from hanging

## 📊 Features

### Multi-Format Support
- Intelligent text extraction from various file formats
- Optimized content chunking for better embedding quality
- Metadata preservation for enhanced search
- Error recovery for corrupted files

### Smart Processing
- **Intelligent Chunking**: Optimized chunk sizes and overlap
- **Text Cleaning**: Removes problematic characters and formatting
- **Progress Tracking**: Real-time processing updates
- **Batch Upload**: Efficient database operations

### User Experience
- **Modern Web Interface**: Drag & drop file uploads
- **Real-time Processing**: Live status updates
- **Responsive Design**: Works on all devices
- **Secure Credential Handling**: User-friendly credential input with validation
- **Detailed Results**: Comprehensive processing statistics
- **Error Guidance**: Helpful troubleshooting tips

## 🚀 Deployment Options

### Railway (Recommended)
- Automatic deployment from GitHub
- Built-in HTTPS
- Global CDN
- Easy scaling
- Health check monitoring

### Other Platforms
- **Heroku**: Use the included `Procfile`
- **DigitalOcean App Platform**: Works out of the box
- **Google Cloud Run**: Container-ready
- **AWS Elastic Beanstalk**: Python application support

## 📈 Monitoring and Maintenance

### Health Checks
- `/health` endpoint for monitoring
- Automatic restart on failures
- Comprehensive logging

### Performance Metrics
- Processing time tracking
- Upload success rates
- Error rate monitoring
- Resource usage optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes locally
4. Ensure no credentials are hardcoded
5. Submit a pull request

## 🔧 API Reference

### Endpoints

- `GET /health` - Health check
- `GET /` - Web interface
- `POST /process-files` - Upload and process files

### File Processing

```python
# Example usage
from universal_file_processor import UniversalFileProcessor

processor = UniversalFileProcessor()
processor.openai_api_key = "your-key"
processor.supabase_url = "your-url"
processor.supabase_service_key = "your-service-key"

chunks = processor.process_file("document.pdf")
result = processor.upload_to_supabase(chunks)
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes

### For Users
- Your credentials are your responsibility
- Monitor your API usage and costs
- This service processes your data using your own infrastructure
- No data is stored on our servers
- File size limit: 16MB per file
- Supported formats: TXT, PDF, DOC, DOCX, CSV

### For Developers
- Never store user credentials
- Always validate inputs
- Implement proper error handling
- Follow security best practices
- Monitor resource usage in production
- Implement rate limiting for high-traffic scenarios

---

**Production-ready RAG processing, secured by design** 🔒

### Support

- 📧 Issues: Use GitHub Issues
- 💬 Discussions: Use GitHub Discussions
- 🔒 Security: Report security issues privately
- 📖 Documentation: Check the README and code comments

---
