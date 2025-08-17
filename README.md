# 🧠 Public RAG File Processor

A secure, production-ready RAG (Retrieval-Augmented Generation) system with **REVOLUTIONARY NEURAL SUPREME PROCESSING** where users provide their own OpenAI and Supabase credentials to process documents and create searchable knowledge bases.

## 🚀 **NEW: NEURAL SUPREME CORE SYSTEM**

This RAG system now features **NEURAL SUPREME PROCESSING** with revolutionary AI capabilities:

1. **🧠 NEURAL SUPREME** - Revolutionary multi-head attention neural networks
2. **🔍 Intelligent Processing** - Semantic categorization and smart retrieval
3. **⚡ Enhanced Processing** - Batch processing with retry logic  
4. **📝 Basic Processing** - Simple chunking (fallback)

### 🧠 Neural Supreme Features
- **🚀 Multi-Head Attention Networks**: Advanced transformer architecture for intelligent chunking
- **🔗 Concept Graph Engine**: Neural-powered knowledge graphs with PageRank algorithms
- **🎯 Multi-Modal Embedding Fusion**: Combines OpenAI, Sentence Transformers, and contextual models
- **💾 Intelligent Caching System**: SQLite-backed caching with LRU and semantic similarity
- **🔄 Adaptive Learning**: Real-time optimization and parameter adjustment
- **⚡ Parallel Processing**: Thread pool execution with GPU acceleration
- **🎨 Advanced Clustering**: HDBSCAN + UMAP for superior semantic grouping
- **🧮 Real-Time Monitoring**: Comprehensive performance and quality metrics
- **🎪 Neural Quality Prediction**: AI-powered chunk quality assessment
- **🌟 Automatic Fallbacks**: Graceful degradation when advanced features unavailable

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

# Install Python dependencies (including neural processing)
pip install -r requirements.txt

# Setup advanced neural models (REQUIRED for Neural Supreme)
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

### 🧠 Neural Supreme Setup

For the ultimate AI processing experience:

```bash
# Install neural dependencies (now included in requirements.txt)
pip install torch transformers sentence-transformers faiss-cpu spacy textstat networkx umap-learn hdbscan

# Download spaCy English model
python -m spacy download en_core_web_sm

# Test the neural core system
python embedding_test.py

# Test neural core directly
python neural_core.py
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
├── neural_core.py                # 🧠 NEURAL SUPREME CORE SYSTEM (NEW!)
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

### 🧠 Neural Supreme Processing Flow
1. **Neural Analysis**: Multi-head attention networks analyze document structure
2. **Concept Extraction**: Advanced NLP with spaCy and pattern recognition
3. **Intelligent Chunking**: Neural networks predict optimal chunk boundaries
4. **Multi-Modal Embeddings**: Fusion of OpenAI, Sentence Transformers, and contextual models
5. **Concept Graph Building**: Neural-powered knowledge graphs with PageRank
6. **Advanced Clustering**: HDBSCAN + UMAP for semantic grouping
7. **Quality Prediction**: AI-powered quality assessment and optimization
8. **Intelligent Caching**: SQLite-backed caching with semantic similarity
9. **Parallel Upload**: Thread pool execution with comprehensive error handling
10. **Real-Time Monitoring**: Advanced performance and quality metrics

### Security Model
1. **Frontend**: Collects user credentials securely
2. **Backend**: Receives credentials in request, never stores them
3. **Processing**: Uses credentials to connect to user's services
4. **Cleanup**: Credentials are discarded after processing

### Data Flow
```
User Credentials → Neural Supreme Core → Multi-Head Attention → Concept Graphs → Advanced Clustering → User's Supabase DB
                        ↓                      ↓                      ↓                    ↓                    ↓
                 Neural Analysis      Intelligent Chunking    Multi-Modal Fusion    Semantic Grouping    Secure Storage
```

### Performance Optimizations
- **🧠 Neural Networks**: Multi-head attention for intelligent processing
- **⚡ GPU Acceleration**: CUDA support for neural computations
- **🔄 Parallel Processing**: Thread pool execution with multiprocessing
- **💾 Intelligent Caching**: SQLite-backed caching with LRU eviction
- **🎯 Adaptive Learning**: Real-time parameter optimization
- **🔗 Concept Graphs**: Neural-powered knowledge relationship mapping
- **🎨 Advanced Clustering**: HDBSCAN + UMAP for superior grouping
- **🚀 Multi-Modal Fusion**: Combines multiple embedding models intelligently
- **📊 Real-Time Monitoring**: Comprehensive performance tracking
- **🛡️ Graceful Fallbacks**: Automatic degradation when features unavailable

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

### 🧠 Neural Supreme Processing
- **🚀 Multi-Head Attention**: Advanced transformer architecture
- **🔗 Neural Concept Graphs**: AI-powered knowledge relationship mapping
- **🎯 Multi-Modal Fusion**: Intelligent embedding combination
- **💾 Smart Caching**: SQLite-backed intelligent caching system
- **🔄 Adaptive Learning**: Real-time optimization and improvement
- **⚡ GPU Acceleration**: CUDA support for neural computations
- **🎨 Advanced Clustering**: HDBSCAN + UMAP semantic grouping
- **🧮 Quality Prediction**: AI-powered chunk quality assessment
- **📊 Real-Time Monitoring**: Comprehensive performance analytics
- **🛡️ Automatic Fallbacks**: Graceful degradation system

### Multi-Format Support
- Intelligent text extraction from various file formats
- Smart content chunking with quality prediction
- Category-aware sizing based on content type
- Metadata preservation for enhanced search
- Error recovery for corrupted files

### 🧠 Neural Processing
- **🚀 Neural Chunking**: Multi-head attention networks for intelligent segmentation
- **🔗 Concept Extraction**: Advanced NLP with spaCy and neural networks
- **🎯 Quality Prediction**: AI-powered chunk quality assessment
- **💾 Intelligent Caching**: Semantic similarity-based caching
- **🔄 Adaptive Optimization**: Real-time parameter adjustment
- **⚡ Parallel Execution**: Thread pool processing with GPU support
- **🎨 Semantic Clustering**: Advanced HDBSCAN + UMAP grouping
- **📊 Performance Analytics**: Real-time monitoring and optimization

### User Experience
- **Modern Web Interface**: Drag & drop file uploads
- **Real-time Processing**: Live status updates
- **Responsive Design**: Works on all devices
- **Secure Credential Handling**: User-friendly credential input with validation
- **🧠 Neural Results**: Advanced processing statistics and AI insights
- **Error Guidance**: Helpful troubleshooting tips
- **🚀 Performance Insights**: Neural processing analytics and optimization tips

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
- 🧠 Neural system health monitoring

### Performance Metrics
- 🧠 Neural processing performance
- 🚀 Multi-head attention efficiency
- 🔗 Concept graph generation stats
- 🎯 Embedding fusion quality metrics
- 💾 Cache hit rates and efficiency
- 🔄 Adaptive learning improvements
- ⚡ GPU utilization and performance
- 🎨 Clustering quality and coherence
- 📊 Real-time optimization metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes locally
4. Ensure no credentials are hardcoded
5. Test with Neural Supreme processing enabled
5. Submit a pull request

## 🔧 API Reference

### Endpoints

- `GET /health` - Health check
- `GET /` - Web interface
- `POST /process-files` - Upload and process files

### File Processing

```python
# 🧠 Neural Supreme usage
from embedding_engine import EmbeddingEngine, EmbeddingConfig, ProcessingLevel
from neural_core import get_neural_core, NeuralConfig, ProcessingMode

# Create Neural Supreme configuration
config = EmbeddingConfig(
    processing_level=ProcessingLevel.NEURAL_SUPREME,
    chunk_size=800,
    batch_size=50
)

# Create engine
engine = EmbeddingEngine(config)
engine.set_credentials(api_key, supabase_url, service_key)

# Process documents
documents = [{'content': text, 'source': 'file.pdf'}]
result = await engine.process_documents(documents)

# Get neural statistics
stats = engine.get_stats()

# Direct neural core usage
neural_core = get_neural_core()
neural_core.set_credentials(api_key, supabase_url, service_key)
result = await neural_core.process_documents_neural(documents)
```

## 🧠 Processing Levels

The Neural Supreme system supports multiple processing levels:

1. **🧠 NEURAL SUPREME** - Revolutionary multi-head attention neural networks
2. **🔍 Intelligent** - Content categorization and smart chunking
3. **⚡ Enhanced** - Batch processing with retry logic
4. **📝 Basic** - Simple chunking (fallback)

Choose Neural Supreme for the ultimate AI-powered processing experience!

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
- **🧠 NEW**: Neural Supreme provides revolutionary AI processing
- **🚀 NEW**: Multi-head attention networks for superior quality
- **🔗 NEW**: Neural concept graphs for advanced knowledge mapping

### For Developers
- Never store user credentials
- Always validate inputs
- Implement proper error handling
- Follow security best practices
- Monitor resource usage in production
- Implement rate limiting for high-traffic scenarios
- **🧠 NEW**: Use Neural Supreme for revolutionary processing
- **🚀 NEW**: Monitor neural performance and optimization metrics
- **💾 NEW**: Leverage intelligent caching for better performance
- **🔄 NEW**: Enable adaptive learning for continuous improvement

---

**🧠 REVOLUTIONARY RAG PROCESSING WITH NEURAL SUPREME INTELLIGENCE** 🚀🔒

### Support

- 📧 Issues: Use GitHub Issues
- 💬 Discussions: Use GitHub Discussions
- 🔒 Security: Report security issues privately
- 📖 Documentation: Check the README and code comments
- 🧠 Neural Supreme: See neural_core.py and embedding_test.py for examples

---
