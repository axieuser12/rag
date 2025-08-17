# 🧠 Public RAG File Processor

A secure, production-ready RAG (Retrieval-Augmented Generation) system with **cutting-edge neural processing** where users provide their own OpenAI and Supabase credentials to process documents and create searchable knowledge bases.

## 🚀 **NEW: Advanced Neural Processing**

This RAG system now features **4 levels of intelligent processing**:

1. **🧠 Neural Processing** (Latest) - Deep learning with transformer models
2. **🎯 Adaptive Processing** - Self-improving with dynamic optimization  
3. **🔍 Intelligent Processing** - Semantic categorization and smart retrieval
4. **⚡ Enhanced Processing** - Batch processing with retry logic
5. **📝 Basic Processing** - Simple chunking (fallback)

### Neural Processing Features
- **Neural Network Chunking**: AI-powered text segmentation
- **Hybrid Embeddings**: Multiple transformer models combined
- **Advanced Clustering**: HDBSCAN + UMAP for semantic grouping
- **FAISS Indexing**: Ultra-fast similarity search
- **Query Expansion**: Neural query enhancement
- **Real-time Quality Prediction**: ML-based chunk scoring

### Adaptive Processing Features  
- **Dynamic Chunk Sizing**: Content-aware optimization
- **Concept Knowledge Graphs**: Relationship mapping
- **Advanced NLP**: spaCy integration for concept extraction
- **Self-Learning**: Parameters improve over time
- **Cross-Reference Detection**: Document linking
- **Quality Metrics**: Comprehensive content analysis

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

For the full neural processing experience:

```bash
# Install advanced dependencies
pip install torch transformers sentence-transformers faiss-cpu spacy textstat networkx umap-learn hdbscan

# Download spaCy English model
python -m spacy download en_core_web_sm

# Run setup script for model caching
python setup_advanced_models.py
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
├── neural_embeddings.py          # 🧠 Neural processing (NEW)
├── adaptive_embeddings.py        # 🎯 Adaptive processing (NEW)
├── intelligent_embeddings.py     # 🔍 Intelligent processing
├── enhanced_embeddings.py        # ⚡ Enhanced processing
├── smart_rag_query.py            # Advanced query system
├── setup_advanced_models.py      # Model setup script
├── requirements.txt              # Python dependencies
├── Procfile                      # Railway deployment config
├── railway.json                  # Railway settings
└── runtime.txt                   # Python version
```

## 🔧 Architecture

### Neural Processing Flow
1. **Document Analysis**: AI-powered content understanding
2. **Neural Chunking**: Transformer-based text segmentation
3. **Hybrid Embeddings**: Multi-model embedding generation
4. **Semantic Clustering**: Advanced ML grouping
5. **Quality Prediction**: Real-time chunk scoring
6. **FAISS Indexing**: Ultra-fast vector search
7. **Knowledge Graphs**: Concept relationship mapping

### Adaptive Learning
1. **Content Analysis**: Complexity and readability scoring
2. **Dynamic Sizing**: Adaptive chunk size optimization
3. **Concept Extraction**: Advanced NLP processing
4. **Quality Metrics**: Comprehensive scoring system
5. **Parameter Learning**: Self-improving algorithms
6. **Cross-Reference Detection**: Document linking

### Security Model
1. **Frontend**: Collects user credentials securely
2. **Backend**: Receives credentials in request, never stores them
3. **Processing**: Uses credentials to connect to user's services
4. **Cleanup**: Credentials are discarded after processing

### Data Flow
```
User Credentials → Neural Processing → Multi-Model Embeddings → Semantic Analysis → User's Supabase DB
                        ↓                      ↓                      ↓                    ↓
                   AI Chunking          Hybrid Models         Advanced ML         Secure Storage
```

### Performance Optimizations
- **Neural Chunking**: AI-powered segmentation with quality prediction
- **Adaptive Sizing**: Dynamic optimization based on content complexity
- **Hybrid Embeddings**: Multiple models for comprehensive understanding
- **Batch Processing**: Efficient processing with intelligent grouping
- **FAISS Indexing**: Ultra-fast similarity search
- **Semantic Clustering**: Advanced ML grouping for better retrieval
- **Memory Management**: Automatic cleanup of temporary files
- **Error Recovery**: Graceful handling of file processing errors
- **Model Caching**: Optimized model loading and reuse

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

### Neural Processing
- **AI-Powered Chunking**: Neural networks determine optimal text segments
- **Multi-Model Embeddings**: Combines OpenAI, Sentence Transformers, and contextual models
- **Advanced Clustering**: HDBSCAN with UMAP for semantic grouping
- **Quality Prediction**: Real-time ML-based chunk quality scoring
- **FAISS Integration**: Ultra-fast vector similarity search
- **Query Expansion**: Neural enhancement of search queries

### Adaptive Intelligence
- **Dynamic Optimization**: Self-adjusting parameters based on content
- **Concept Graphs**: Knowledge relationship mapping with NetworkX
- **Advanced NLP**: spaCy integration for entity and concept extraction
- **Cross-Reference Detection**: Automatic document linking
- **Quality Metrics**: Comprehensive readability and complexity analysis
- **Learning Algorithms**: Parameters improve with each processing session

### Multi-Format Support
- Intelligent text extraction from various file formats
- Neural-powered content chunking with quality prediction
- Adaptive sizing based on content complexity
- Metadata preservation for enhanced search
- Error recovery for corrupted files

### Smart Processing
- **Neural Chunking**: AI-powered text segmentation
- **Adaptive Sizing**: Dynamic optimization based on content
- **Hybrid Embeddings**: Multi-model approach for comprehensive understanding
- **Text Cleaning**: Removes problematic characters and formatting
- **Progress Tracking**: Real-time processing updates
- **Semantic Clustering**: Advanced ML grouping
- **Quality Prediction**: Real-time chunk scoring

### User Experience
- **Modern Web Interface**: Drag & drop file uploads
- **Real-time Processing**: Live status updates
- **Responsive Design**: Works on all devices
- **Secure Credential Handling**: User-friendly credential input with validation
- **Advanced Results**: Neural processing statistics and quality metrics
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
- Neural processing statistics
- Quality distribution analysis
- Embedding generation metrics
- Clustering performance
- Upload success rates
- Error rate monitoring
- Model performance optimization

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
from universal_file_processor import UniversalFileProcessor

processor = UniversalFileProcessor()
processor.openai_api_key = "your-key"
processor.supabase_url = "your-url"
processor.supabase_service_key = "your-service-key"

chunks = processor.process_file("document.pdf")
result = processor.upload_to_supabase(chunks)

# Advanced neural processing
from neural_embeddings import run_neural_processing

result = await run_neural_processing(
    chunks, api_key, supabase_url, service_key
)

# Adaptive processing
from adaptive_embeddings import run_adaptive_processing

result = await run_adaptive_processing(
    chunks, api_key, supabase_url, service_key
)
```

## 🧪 Processing Levels

The system automatically selects the best available processing level:

1. **🧠 Neural** → **🎯 Adaptive** → **🔍 Intelligent** → **⚡ Enhanced** → **📝 Basic**

Each level provides increasingly sophisticated processing with better quality and performance.

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
- **NEW**: Neural processing provides significantly better results
- **NEW**: Adaptive learning improves quality over time

### For Developers
- Never store user credentials
- Always validate inputs
- Implement proper error handling
- Follow security best practices
- Monitor resource usage in production
- Implement rate limiting for high-traffic scenarios
- **NEW**: Consider GPU acceleration for neural processing
- **NEW**: Monitor model performance and quality metrics

---

**Next-generation RAG processing with neural intelligence, secured by design** 🧠🔒

### Support

- 📧 Issues: Use GitHub Issues
- 💬 Discussions: Use GitHub Discussions
- 🔒 Security: Report security issues privately
- 📖 Documentation: Check the README and code comments
- 🧠 Neural Processing: See setup_advanced_models.py for configuration

---
