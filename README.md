# 🧠 Public RAG File Processor

A secure, multi-user RAG (Retrieval-Augmented Generation) system where users provide their own OpenAI and Supabase credentials to process documents and create searchable knowledge bases.

## 🔒 Security Features

- **Zero Credential Storage**: User credentials are never stored on our servers
- **Session-Only Processing**: Credentials are used only for the current session
- **Direct API Communication**: Your data goes directly to your own OpenAI and Supabase services
- **No Data Retention**: No user data is retained after processing
- **Open Source**: Full transparency of how your credentials are handled

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
       metadata jsonb
   );
   
   -- Create index for fast similarity search
   CREATE INDEX IF NOT EXISTS idx_documents_embedding
   ON documents
   USING ivfflat (embedding vector_cosine_ops)
   WITH (lists = 100);
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
- **Spreadsheets**: .csv

## 🛠️ For Developers

### Local Development

```bash
# Clone the repository
git clone <your-repo>
cd rag-file-processor

# Install dependencies
pip install -r requirements.txt

# Start the development server
python web_server.py
```

### Deployment to Railway

1. **Connect Your Repository**:
   - Link your GitHub repository to Railway
   - Railway will auto-detect the Python app

2. **Environment Variables** (Optional):
   - No environment variables needed for basic deployment
   - The system uses user-provided credentials

3. **Deploy**:
   - Railway will automatically deploy your app
   - Users can access it via the provided URL

### Project Structure

```
├── src/                          # React frontend
│   ├── components/
│   │   ├── CredentialsForm.tsx   # Secure credential input
│   │   ├── FileUploader.tsx      # File upload interface
│   │   └── ...
│   └── App.tsx                   # Main application
├── web_server.py                 # Flask backend
├── universal_file_processor.py   # File processing logic
├── security_validator.py         # Security scanner
└── requirements.txt              # Python dependencies
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

## 📊 Features

### Multi-Format Support
- Intelligent text extraction from various file formats
- Automatic content chunking for optimal embedding
- Metadata preservation for enhanced search

### Smart Processing
- **Multiple Chunking Strategies**: Ensures comprehensive coverage
- **Key Information Extraction**: Automatically identifies contacts, pricing, etc.
- **Content Categorization**: Organizes chunks by type for better retrieval

### User Experience
- **Modern Web Interface**: Drag & drop file uploads
- **Real-time Processing**: Live status updates
- **Responsive Design**: Works on all devices
- **Secure Credential Handling**: User-friendly credential input with validation

## 🚀 Deployment Options

### Railway (Recommended)
- Automatic deployment from GitHub
- Built-in HTTPS
- Global CDN
- Easy scaling

### Other Platforms
- **Heroku**: Use the included `Procfile`
- **DigitalOcean App Platform**: Works out of the box
- **Google Cloud Run**: Container-ready
- **AWS Elastic Beanstalk**: Python application support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run security validation: `python security_validator.py`
4. Ensure no credentials are hardcoded
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes

### For Users
- Your credentials are your responsibility
- Monitor your API usage and costs
- This service processes your data using your own infrastructure
- No data is stored on our servers

### For Developers
- Never store user credentials
- Always validate inputs
- Implement proper error handling
- Follow security best practices

---

**Built for the community, secured by design** 🔒

### Support

- 📧 Issues: Use GitHub Issues
- 💬 Discussions: Use GitHub Discussions
- 🔒 Security: Report security issues privately

---

**Your data, your infrastructure, your control** ✨