@@ .. @@
 class UniversalFileProcessor:
     def __init__(self):
         self.supported_extensions = {'.txt', '.pdf', '.doc', '.docx', '.csv'}
         self.chunks = []
+        # Initialize with None - will be set by web server with user credentials
+        self.openai_api_key = None
+        self.supabase_url = None
+        self.supabase_service_key = None
+    
+    def get_openai_client(self):
+        """Get OpenAI client with user-provided API key"""
+        if not self.openai_api_key:
+            raise ValueError("OpenAI API key not provided")
+        return OpenAI(api_key=self.openai_api_key)
+    
+    def get_supabase_client(self):
+        """Get Supabase client with user-provided credentials"""
+        if not self.supabase_url or not self.supabase_service_key:
+            raise ValueError("Supabase credentials not provided")
+        return create_client(self.supabase_url, self.supabase_service_key)
         
     def extract_text_from_file(self, file_path: str, file_content: bytes = None) -> str:
@@ .. @@
     def get_embedding(self, text: str) -> List[float]:
         """Generate embedding for text"""
         try:
-            response = openai_client.embeddings.create(
+            client = self.get_openai_client()
+            response = client.embeddings.create(
                 input=text[:8000],  # Limit text length for embedding
                 model="text-embedding-3-small"
             )
             return response.data[0].embedding
         except Exception as e:
             print(f"Error generating embedding: {e}")
             return None
     
     def process_file(self, file_path: str, file_content: bytes = None) -> List[Dict[str, Any]]:
@@ .. @@
     def upload_to_supabase(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
         """Upload all chunks to Supabase with embeddings"""
         print("Starting upload to Supabase...")
         
+        # Get user's Supabase client
+        supabase = self.get_supabase_client()
+        
         successful_uploads = 0
         failed_uploads = 0
         
         for i, chunk in enumerate(chunks, 1):
             try:
                 # Generate embedding
                 embedding = self.get_embedding(chunk["content"])
                 if embedding is None:
                     print(f"Chunk {i}: Failed to generate embedding")
                     failed_uploads += 1
                     continue
                 
                 # Prepare data for Supabase
                 data = {
                     "content": chunk["content"],
                     "embedding": embedding,
                     "source": chunk["source"],
                     "metadata": {
                         "title": chunk["title"],
                         "chunk_type": chunk["chunk_type"],
                         **chunk.get("metadata", {})
                     }
                 }
                 
                 # Insert into Supabase
                 response = supabase.table("documents").insert(data).execute()
                 
                 if hasattr(response, 'error') and response.error:
                     print(f"Chunk {i}: Upload failed - {response.error}")
                     failed_uploads += 1
                 else:
                     successful_uploads += 1
                     if i % 10 == 0:  # Progress update every 10 chunks
                         print(f"Uploaded {i}/{len(chunks)} chunks...")
                 
             except Exception as e:
                 print(f"Chunk {i}: Exception during upload - {e}")
                 failed_uploads += 1
         
         result = {
             "successful_uploads": successful_uploads,
             "failed_uploads": failed_uploads,
             "total_chunks": len(chunks)
         }
         
         print(f"Upload complete! Success: {successful_uploads}, Failed: {failed_uploads}")
         return result
 
-def process_files_from_directory(directory_path: str = "uploaded_files") -> Dict[str, Any]:
-    """Process all files from a directory"""
-    processor = UniversalFileProcessor()
-    all_chunks = []
-    files_processed = 0
-    
-    if not os.path.exists(directory_path):
-        return {
-            "success": False,
-            "message": "Upload directory not found",
-            "error": f"Directory {directory_path} does not exist"
-        }
-    
-    # Process all files in the directory
-    for filename in os.listdir(directory_path):
-        file_path = os.path.join(directory_path, filename)
-        
-        if os.path.isfile(file_path):
-            file_extension = Path(file_path).suffix.lower()
-            
-            if file_extension in processor.supported_extensions:
-                try:
-                    file_chunks = processor.process_file(file_path)
-                    all_chunks.extend(file_chunks)
-                    files_processed += 1
-                except Exception as e:
-                    print(f"Error processing {filename}: {e}")
-    
-    if not all_chunks:
-        return {
-            "success": False,
-            "message": "No content could be extracted from the uploaded files",
-            "files_processed": files_processed
-        }
-    
-    # Upload to Supabase
-    upload_result = processor.upload_to_supabase(all_chunks)
-    
-    return {
-        "success": True,
-        "message": f"Successfully processed {files_processed} files and created {len(all_chunks)} chunks",
-        "files_processed": files_processed,
-        "chunks_created": len(all_chunks),
-        "upload_stats": upload_result
-    }
-
 def main():
     """Main function for testing"""
-    # Test with existing txt files
-    txt_folder = "Txt File"
-    if os.path.exists(txt_folder):
-        result = process_files_from_directory(txt_folder)
-        print(json.dumps(result, indent=2))
-    else:
-        print("Txt File folder not found. Please upload files through the web interface.")
+    print("This is a library module. Use the web interface to process files.")
+    print("Start the web server with: python web_server.py")
 
 if __name__ == "__main__":
     main()