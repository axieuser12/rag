# 🧠 Public RAG File Processor

A secure, production-ready RAG (Retrieval-Augmented Generation) system with **cutting-edge neural processing** where users provide their own OpenAI and Supabase credentials to process documents and create searchable knowledge bases.

## 🚀 **NEW: Advanced Neural Processing**

This RAG system now features **centralized embedding processing** with multiple levels:

1. **🔍 Intelligent Processing** - Semantic categorization and smart retrieval
2. **⚡ Enhanced Processing** - Batch processing with retry logic  
3. **📝 Basic Processing** - Simple chunking (fallback)

### Centralized Engine Features
- **Pluggable Strategies**: Easy to switch between processing levels
- **Unified Configuration**: Single config object for all settings
- **Comprehensive Stats**: Detailed performance and quality metrics
- **Backward Compatibility**: Works with existing code
- **Content Categorization**: Automatic classification of document types
- **Quality Scoring**: Advanced chunk quality assessment

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
- **Neural Chunking**: AI-powered text segmentation with quality prediction
- **Adaptive Sizing**: Dynamic chunk optimization based on content complexity
- **Comprehensive Processing**: Multiple chunking strategies for maximum coverage
- **Real-time Progress**: Live updates during processing
- **Error Handling**: Detailed error reporting and troubleshooting tips
- **Multi-Model Embeddings**: Hybrid approach using multiple transformer models
- **Semantic Clustering**: Advanced grouping with HDBSCAN and UMAP
- **Concept Graphs**: Knowledge relationship mapping
- **Production Ready**: Optimized deployment with comprehensive logging

## 🧠 Neural Processing Architecture

### Processing Pipeline
```
Documents → Neural Chunking → Hybrid Embeddings → Semantic Clustering → FAISS Index → Supabase
     ↓              ↓                ↓                    ↓              ↓           ↓
  AI Analysis   Multi-Model    Advanced ML        Vector Index    Knowledge Base
```

### Model Stack
- **OpenAI**: `text-embedding-3-small` (primary embeddings)
- **Sentence Transformers**: `all-MiniLM-L6-v2` (local embeddings)
- **Transformers**: `microsoft/DialoGPT-medium` (contextual understanding)
- **spaCy**: `en_core_web_sm` (NLP processing)
- **Neural Chunker**: Custom PyTorch model for intelligent segmentation

## 🎯 Adaptive Features

### Self-Improving System
- **Dynamic Parameters**: Automatically adjusts based on content type
- **Quality Learning**: Improves chunk quality over time
- **Content Analysis**: Advanced readability and complexity scoring
- **Concept Extraction**: Intelligent key concept identification
- **Cross-References**: Automatic document relationship detection

## 🛠️ For Developers

### Local Development

```bash
# Clone the repository
git clone <your-repo>
cd rag-file-processor

# Install Python dependencies
pip install -r requirements.txt

# Setup advanced neural models (optional but recommended)
python setup_advanced_models.py

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

### Advanced Setup

For enhanced processing capabilities:

```bash
# Install advanced dependencies (optional)
# pip install torch transformers sentence-transformers faiss-cpu spacy textstat networkx umap-learn hdbscan

# Download spaCy English model
# python -m spacy download en_core_web_sm

# Test the embedding engine
python embedding_test.py
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
├── embedding_engine.py           # 🚀 Centralized embedding engine (NEW)
├── intelligent_embeddings.py     # 🔍 Intelligent processing
├── enhanced_embeddings.py        # ⚡ Enhanced processing
├── smart_rag_query.py            # Advanced query system
├── embedding_test.py             # Test script for embedding engine
├── requirements.txt              # Python dependencies
├── Procfile                      # Railway deployment config
├── railway.json                  # Railway settings
└── runtime.txt                   # Python version
```

## 🔧 Architecture

### Centralized Processing Flow
1. **Strategy Selection**: Choose processing level based on requirements
2. **Document Analysis**: Content categorization and analysis
3. **Smart Chunking**: Adaptive text segmentation
4. **Embedding Generation**: Batch processing with retry logic
5. **Quality Assessment**: Comprehensive chunk scoring
6. **Supabase Upload**: Efficient batch uploads
7. **Performance Tracking**: Detailed statistics and metrics

### Security Model
1. **Frontend**: Collects user credentials securely
2. **Backend**: Receives credentials in request, never stores them
3. **Processing**: Uses credentials to connect to user's services
4. **Cleanup**: Credentials are discarded after processing

### Data Flow
```
User Credentials → Centralized Engine → Strategy Processing → Quality Assessment → User's Supabase DB
                        ↓                      ↓                      ↓                    ↓
                 Strategy Selection      Smart Chunking        Embedding Generation    Secure Storage
```

### Performance Optimizations
- **Pluggable Strategies**: Choose optimal processing level
- **Smart Chunking**: Content-aware text segmentation
- **Quality Scoring**: Advanced chunk assessment
- **Batch Processing**: Efficient processing with intelligent grouping
- **Retry Logic**: Robust error handling
- **Content Categorization**: Automatic document classification
- **Memory Management**: Automatic cleanup of temporary files
- **Error Recovery**: Graceful handling of file processing errors
- **Performance Tracking**: Comprehensive statistics

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

### Centralized Processing
- **Strategy Pattern**: Pluggable processing strategies
- **Unified Configuration**: Single config object for all settings
- **Content Categorization**: Automatic document type classification
- **Quality Assessment**: Advanced chunk quality scoring
- **Performance Tracking**: Comprehensive statistics and metrics
- **Backward Compatibility**: Works with existing integrations

### Multi-Format Support
- Intelligent text extraction from various file formats
- Smart content chunking with quality prediction
- Category-aware sizing based on content type
- Metadata preservation for enhanced search
- Error recovery for corrupted files

### Smart Processing
- **Smart Chunking**: Content-aware text segmentation
- **Category-based Sizing**: Optimization based on content type
- **Quality Scoring**: Advanced chunk assessment
- **Text Cleaning**: Removes problematic characters and formatting
- **Progress Tracking**: Real-time processing updates
- **Content Analysis**: Automatic categorization and concept extraction
- **Retry Logic**: Robust error handling

### User Experience
- **Modern Web Interface**: Drag & drop file uploads
- **Real-time Processing**: Live status updates
- **Responsive Design**: Works on all devices
- **Secure Credential Handling**: User-friendly credential input with validation
- **Advanced Results**: Processing statistics and quality metrics
- **Error Guidance**: Helpful troubleshooting tips
- **Performance Insights**: Detailed processing analytics

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
- Strategy performance statistics
- Quality distribution analysis
- Embedding generation metrics
- Upload success rates
- Error rate monitoring
- Strategy usage tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes locally
4. Ensure no credentials are hardcoded
5. Test with neural processing enabled
5. Submit a pull request

## 🔧 API Reference

### Endpoints

- `GET /health` - Health check
- `GET /` - Web interface
- `POST /process-files` - Upload and process files

### File Processing

```python
# Basic usage
from embedding_engine import EmbeddingEngine, EmbeddingConfig, ProcessingLevel

# Create configuration
config = EmbeddingConfig(
    processing_level=ProcessingLevel.INTELLIGENT,
    chunk_size=800,
    batch_size=50
)

# Create engine
engine = EmbeddingEngine(config)
engine.set_credentials(api_key, supabase_url, service_key)

# Process documents
documents = [{'content': text, 'source': 'file.pdf'}]
result = await engine.process_documents(documents)

# Get statistics
stats = engine.get_stats()
```

## 🧪 Processing Levels

The centralized engine supports multiple processing levels:

1. **🔍 Intelligent** - Content categorization and smart chunking
2. **⚡ Enhanced** - Batch processing with retry logic
3. **📝 Basic** - Simple chunking (fallback)

Choose the level based on your performance and quality requirements.

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
- **NEW**: Centralized engine provides consistent, high-quality processing
- **NEW**: Pluggable strategies for different use cases

### For Developers
- Never store user credentials
- Always validate inputs
- Implement proper error handling
- Follow security best practices
- Monitor resource usage in production
- Implement rate limiting for high-traffic scenarios
- **NEW**: Use centralized engine for consistent processing
- **NEW**: Monitor strategy performance and quality metrics

---

**Next-generation RAG processing with centralized intelligence, secured by design** 🚀🔒

### Support

- 📧 Issues: Use GitHub Issues
- 💬 Discussions: Use GitHub Discussions
- 🔒 Security: Report security issues privately
- 📖 Documentation: Check the README and code comments
- 🚀 Centralized Engine: See embedding_test.py for examples

---
