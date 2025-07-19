#!/usr/bin/env python3
"""
Web Server for RAG File Processing
Handles file uploads and processing via HTTP API
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from universal_file_processor import process_files_from_directory, UniversalFileProcessor

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "message": "RAG File Processor is running",
        "version": "1.0.0"
    })

@app.route('/', methods=['GET'])
def index():
    """Serve the main page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG File Processor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .container { max-width: 800px; margin: 0 auto; text-align: center; }
            .upload-area { border: 2px dashed white; padding: 40px; margin: 20px 0; border-radius: 10px; }
            input[type="file"] { margin: 20px 0; }
            button { background: white; color: #667eea; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; }
            button:hover { background: #f0f0f0; }
            .result { margin: 20px 0; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 RAG File Processor</h1>
            <p>Upload your documents to create a searchable knowledge base</p>
            
            <div class="upload-area">
                <h3>Upload Files</h3>
                <input type="file" id="fileInput" multiple accept=".txt,.pdf,.doc,.docx,.csv">
                <br>
                <button onclick="uploadFiles()">Process Files</button>
            </div>
            
            <div id="result" class="result" style="display:none;"></div>
        </div>
        
        <script>
            async function uploadFiles() {
                const fileInput = document.getElementById('fileInput');
                const resultDiv = document.getElementById('result');
                
                if (fileInput.files.length === 0) {
                    alert('Please select files to upload');
                    return;
                }
                
                const formData = new FormData();
                for (let i = 0; i < fileInput.files.length; i++) {
                    formData.append('file_' + i, fileInput.files[i]);
                }
                
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<h3>Processing files...</h3><p>Please wait while we process your files and create embeddings.</p>';
                
                try {
                    const response = await fetch('/process-files', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        resultDiv.innerHTML = `
                            <h3>✅ Success!</h3>
                            <p>${result.message}</p>
                            <p>Files processed: ${result.files_processed}</p>
                            <p>Chunks created: ${result.chunks_created}</p>
                        `;
                    } else {
                        resultDiv.innerHTML = `
                            <h3>❌ Error</h3>
                            <p>${result.message}</p>
                            ${result.error ? '<p>Error: ' + result.error + '</p>' : ''}
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `
                        <h3>❌ Error</h3>
                        <p>Failed to process files: ${error.message}</p>
                    `;
                }
            }
        </script>
    </body>
    </html>
    """

@app.route('/process-files', methods=['POST'])
def process_files():
    """Handle file upload and processing"""
    try:
        # Check if files were uploaded
        if not request.files:
            return jsonify({
                "success": False,
                "message": "No files were uploaded"
            }), 400
        
        # Create temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
        
        try:
            files_saved = 0
            
            # Save uploaded files
            for key in request.files:
                file = request.files[key]
                if file and file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(temp_dir, filename)
                    file.save(file_path)
                    files_saved += 1
            
            if files_saved == 0:
                return jsonify({
                    "success": False,
                    "message": "No valid files were uploaded"
                }), 400
            
            # Process the uploaded files
            result = process_files_from_directory(temp_dir)
            return jsonify(result)
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": "Error processing files",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)