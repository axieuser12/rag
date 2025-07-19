@@ .. @@
 #!/usr/bin/env python3
 """
 Web Server for RAG File Processing
 Handles file uploads and processing via HTTP API
 """
 
 import os
 import json
 import tempfile
 import shutil
+import logging
 from pathlib import Path
 from typing import Dict, Any
 from flask import Flask, request, jsonify
 from flask_cors import CORS
 from werkzeug.utils import secure_filename
-from universal_file_processor import process_files_from_directory, UniversalFileProcessor
+from universal_file_processor import UniversalFileProcessor
+
+# Configure logging
+logging.basicConfig(level=logging.INFO)
+logger = logging.getLogger(__name__)
 
 # Initialize Flask app
 app = Flask(__name__)
 CORS(app)  # Enable CORS for all routes
 
 # Configuration
 ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'csv'}
 MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
 
 app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
 
 def allowed_file(filename):
     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
+def validate_credentials(credentials):
+    """Validate user-provided credentials"""
+    errors = []
+    
+    if not credentials.get('openai_api_key'):
+        errors.append("OpenAI API key is required")
+    elif not credentials['openai_api_key'].startswith('sk-'):
+        errors.append("Invalid OpenAI API key format")
+    
+    if not credentials.get('supabase_url'):
+        errors.append("Supabase URL is required")
+    elif 'supabase.co' not in credentials['supabase_url']:
+        errors.append("Invalid Supabase URL format")
+    
+    if not credentials.get('supabase_service_key'):
+        errors.append("Supabase service key is required")
+    elif not credentials['supabase_service_key'].startswith('eyJ'):
+        errors.append("Invalid Supabase service key format")
+    
+    return errors
+
+def process_files_with_credentials(temp_dir: str, credentials: Dict[str, str]) -> Dict[str, Any]:
+    """Process files using user-provided credentials"""
+    try:
+        # Create processor instance
+        processor = UniversalFileProcessor()
+        
+        # Override credentials for this session
+        processor.openai_api_key = credentials['openai_api_key']
+        processor.supabase_url = credentials['supabase_url']
+        processor.supabase_service_key = credentials['supabase_service_key']
+        
+        all_chunks = []
+        files_processed = 0
+        
+        if not os.path.exists(temp_dir):
+            return {
+                "success": False,
+                "message": "Upload directory not found",
+                "error": f"Directory {temp_dir} does not exist"
+            }
+        
+        # Process all files in the directory
+        for filename in os.listdir(temp_dir):
+            file_path = os.path.join(temp_dir, filename)
+            
+            if os.path.isfile(file_path):
+                file_extension = Path(file_path).suffix.lower()
+                
+                if file_extension in processor.supported_extensions:
+                    try:
+                        file_chunks = processor.process_file(file_path)
+                        all_chunks.extend(file_chunks)
+                        files_processed += 1
+                        logger.info(f"Processed {filename}: {len(file_chunks)} chunks")
+                    except Exception as e:
+                        logger.error(f"Error processing {filename}: {e}")
+        
+        if not all_chunks:
+            return {
+                "success": False,
+                "message": "No content could be extracted from the uploaded files",
+                "files_processed": files_processed
+            }
+        
+        # Upload to user's Supabase
+        upload_result = processor.upload_to_supabase(all_chunks)
+        
+        return {
+            "success": True,
+            "message": f"Successfully processed {files_processed} files and created {len(all_chunks)} chunks",
+            "files_processed": files_processed,
+            "chunks_created": len(all_chunks),
+            "upload_stats": upload_result
+        }
+        
+    except Exception as e:
+        logger.error(f"Error in process_files_with_credentials: {e}")
+        return {
+            "success": False,
+            "message": "Error processing files",
+            "error": str(e)
+        }
+
 @app.route('/health', methods=['GET'])
 def health_check():
     """Health check endpoint"""
     return jsonify({
         "status": "healthy", 
-        "message": "RAG File Processor is running",
+        "message": "Public RAG File Processor is running",
         "version": "1.0.0"
     })
 
 @app.route('/', methods=['GET'])
 def index():
     """Serve the main page"""
     return """
     <!DOCTYPE html>
     <html>
     <head>
-        <title>RAG File Processor</title>
+        <title>Public RAG File Processor</title>
         <style>
             body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
             .container { max-width: 800px; margin: 0 auto; text-align: center; }
             .upload-area { border: 2px dashed white; padding: 40px; margin: 20px 0; border-radius: 10px; }
             input[type="file"] { margin: 20px 0; }
+            input[type="text"], input[type="password"] { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ccc; border-radius: 5px; }
             button { background: white; color: #667eea; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; }
             button:hover { background: #f0f0f0; }
             .result { margin: 20px 0; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px; }
+            .credentials { background: rgba(255,255,255,0.1); padding: 20px; margin: 20px 0; border-radius: 10px; text-align: left; }
         </style>
     </head>
     <body>
         <div class="container">
-            <h1>🧠 RAG File Processor</h1>
-            <p>Upload your documents to create a searchable knowledge base</p>
+            <h1>🧠 Public RAG File Processor</h1>
+            <p>Upload your documents using your own OpenAI and Supabase credentials</p>
+            
+            <div class="credentials">
+                <h3>🔐 Your Credentials (Secure)</h3>
+                <p>Your credentials are never stored on our servers</p>
+                <input type="text" id="openaiKey" placeholder="OpenAI API Key (sk-...)">
+                <input type="text" id="supabaseUrl" placeholder="Supabase URL (https://your-project.supabase.co)">
+                <input type="password" id="supabaseKey" placeholder="Supabase Service Key (eyJ...)">
+            </div>
             
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
+                const openaiKey = document.getElementById('openaiKey').value;
+                const supabaseUrl = document.getElementById('supabaseUrl').value;
+                const supabaseKey = document.getElementById('supabaseKey').value;
                 const resultDiv = document.getElementById('result');
                 
                 if (fileInput.files.length === 0) {
                     alert('Please select files to upload');
                     return;
                 }
                 
+                if (!openaiKey || !supabaseUrl || !supabaseKey) {
+                    alert('Please provide all credentials');
+                    return;
+                }
+                
                 const formData = new FormData();
                 for (let i = 0; i < fileInput.files.length; i++) {
                     formData.append('file_' + i, fileInput.files[i]);
                 }
                 
+                // Add credentials
+                formData.append('openai_api_key', openaiKey);
+                formData.append('supabase_url', supabaseUrl);
+                formData.append('supabase_service_key', supabaseKey);
+                
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
+        # Extract credentials from form data
+        credentials = {
+            'openai_api_key': request.form.get('openai_api_key', ''),
+            'supabase_url': request.form.get('supabase_url', ''),
+            'supabase_service_key': request.form.get('supabase_service_key', '')
+        }
+        
+        # Validate credentials
+        credential_errors = validate_credentials(credentials)
+        if credential_errors:
+            return jsonify({
+                "success": False,
+                "message": "Invalid credentials provided",
+                "error": "; ".join(credential_errors)
+            }), 400
+        
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
-                file = request.files[key]
-                if file and file.filename and allowed_file(file.filename):
-                    filename = secure_filename(file.filename)
-                    file_path = os.path.join(temp_dir, filename)
-                    file.save(file_path)
-                    files_saved += 1
+                if key.startswith('file_'):  # Only process file uploads, not credentials
+                    file = request.files[key]
+                    if file and file.filename and allowed_file(file.filename):
+                        filename = secure_filename(file.filename)
+                        file_path = os.path.join(temp_dir, filename)
+                        file.save(file_path)
+                        files_saved += 1
             
             if files_saved == 0:
                 return jsonify({
                     "success": False,
                     "message": "No valid files were uploaded"
                 }), 400
             
-            # Process the uploaded files
-            result = process_files_from_directory(temp_dir)
+            # Process the uploaded files with user credentials
+            result = process_files_with_credentials(temp_dir, credentials)
             return jsonify(result)
             
         finally:
             # Clean up temporary directory
             shutil.rmtree(temp_dir, ignore_errors=True)
             
     except Exception as e:
+        logger.error(f"Error in process_files endpoint: {e}")
         return jsonify({
             "success": False,
             "message": "Error processing files",
             "error": str(e)
         }), 500
 
 if __name__ == "__main__":
     port = int(os.environ.get("PORT", 8000))
-    app.run(host="0.0.0.0", port=port, debug=False)
+    app.run(host="0.0.0.0", port=port, debug=False)