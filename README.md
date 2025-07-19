# 🧠 RAG File Processor

A secure, production-ready RAG (Retrieval-Augmented Generation) system that processes documents and creates a searchable knowledge base using OpenAI embeddings and Supabase vector storage.

## 🔒 Security Features

- **No API Key Leakage**: All sensitive credentials are stored in environment variables
- **Environment-based Configuration**: Uses `.env` file for all configuration
- **Security Validation**: Built-in security scanner to detect potential API key exposure
- **Production Ready**: Secure deployment configuration

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual credentials
nano .env
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Validate Security

```bash
# Run security scan to ensure no API keys are exposed
python security_validator.py
```

### 4. Process Files

```bash
# Process all txt files in the Txt File directory
python run_comprehensive_processing.py

# Or start the web server for file uploads
python web_server.py
```

## 📁 Project Structure

```
├── .env                          # Environment variables (DO NOT COMMIT)
├── .env.example                  # Example environment file
├── .gitignore                    # Git ignore file (includes .env)
├── security_validator.py         # Security scanner
├── web_server.py                 # Web interface for file uploads
├── universal_file_processor.py   # Universal file processing
├── improved_chunk_processor.py   # Advanced chunking strategies
├── enhanced_query_system.py      # Query and search system
└── Embedded_Rag_Vectorstore_Supabase/
    ├── prepare_rag_chunks.py     # Basic chunking
    ├── ingest_to_supabase.py     # Supabase ingestion
    └── rag_query_example.py      # Query examples
```

## 🔧 Configuration

All configuration is done through environment variables in the `.env` file:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Supabase Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_SERVICE_KEY=your_supabase_service_key_here

# Optional: Direct database connection
SUPABASE_DB_HOST=your_db_host_here
SUPABASE_DB_PORT=5432
SUPABASE_DB_NAME=postgres
SUPABASE_DB_USER=postgres
SUPABASE_DB_PASSWORD=your_db_password_here
```

## 🛡️ Security Best Practices

1. **Never commit `.env` files** - They're automatically ignored by git
2. **Use environment variables** - All scripts read from environment variables only
3. **Run security scans** - Use `python security_validator.py` regularly
4. **Rotate API keys** - Regularly update your API keys
5. **Limit permissions** - Use service role keys with minimal required permissions

## 📊 Features

### File Processing
- **Multiple Formats**: TXT, PDF, Word documents, CSV files
- **Smart Chunking**: Multiple chunking strategies for comprehensive coverage
- **Content Extraction**: Automatic extraction of key information (contacts, pricing, etc.)

### Vector Storage
- **Supabase Integration**: Secure vector storage with PostgreSQL + pgvector
- **OpenAI Embeddings**: High-quality text embeddings using `text-embedding-3-small`
- **Metadata Support**: Rich metadata for enhanced search and filtering

### Web Interface
- **Drag & Drop Upload**: Modern web interface for file uploads
- **Real-time Processing**: Live status updates during processing
- **Responsive Design**: Works on desktop and mobile devices

### Query System
- **Semantic Search**: Vector similarity search for relevant content
- **Category Filtering**: Search by content type (pricing, contact, etc.)
- **Comprehensive Results**: Multiple search strategies for best coverage

## 🚀 Deployment

### Local Development
```bash
python web_server.py
```

### Production Deployment
The project includes configuration for Railway, Heroku, and other platforms:

1. Set environment variables in your deployment platform
2. Deploy using the included `Procfile` and `railway.json`
3. The web server will automatically start on the assigned port

## 🔍 Security Validation

Run the security validator to ensure no API keys are exposed:

```bash
python security_validator.py
```

This will scan all project files and report any potential security issues.

## 📝 Usage Examples

### Process Files via Web Interface
1. Start the web server: `python web_server.py`
2. Open `http://localhost:8000` in your browser
3. Upload files using the drag & drop interface
4. Files are automatically processed and stored in Supabase

### Process Files via Command Line
```bash
# Process all txt files
python run_comprehensive_processing.py

# Process specific file types
python universal_file_processor.py
```

### Query the Knowledge Base
```bash
# Interactive query system
python enhanced_query_system.py

# Basic query example
python Embedded_Rag_Vectorstore_Supabase/rag_query_example.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run security validation: `python security_validator.py`
4. Ensure all API keys are in environment variables
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Security Notes

- **Never commit API keys** to version control
- **Always use environment variables** for sensitive configuration
- **Run security scans** before deploying
- **Rotate API keys** regularly
- **Use minimal permissions** for service accounts

---

**Built with security in mind** 🔒