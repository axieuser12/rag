import os
import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from supabase import create_client

class UniversalFileProcessor:
    def __init__(self):
        self.supported_extensions = {'.txt', '.pdf', '.doc', '.docx', '.csv'}
        self.chunks = []
        # Initialize with None - will be set by web server with user credentials
        self.openai_api_key = None
        self.supabase_url = None
        self.supabase_service_key = None
    
    def get_openai_client(self):
        """Get OpenAI client with user-provided API key"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided")
        return OpenAI(api_key=self.openai_api_key)
    
    def get_supabase_client(self):
        """Get Supabase client with user-provided credentials"""
        if not self.supabase_url or not self.supabase_service_key:
            raise ValueError("Supabase credentials not provided")
        return create_client(self.supabase_url, self.supabase_service_key)
        
    def extract_text_from_file(self, file_path: str, file_content: bytes = None) -> str:
        """Extract text from various file formats"""
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.txt':
                if file_content:
                    return file_content.decode('utf-8', errors='ignore')
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        return file.read()
            
            elif file_extension == '.pdf':
                try:
                    import PyPDF2
                    if file_content:
                        import io
                        pdf_file = io.BytesIO(file_content)
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                    else:
                        with open(file_path, 'rb') as file:
                            pdf_reader = PyPDF2.PdfReader(file)
                    
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
                except ImportError:
                    print("PyPDF2 not installed. Install with: pip install PyPDF2")
                    return ""
            
            elif file_extension in ['.doc', '.docx']:
                try:
                    import docx
                    if file_content:
                        import io
                        doc_file = io.BytesIO(file_content)
                        doc = docx.Document(doc_file)
                    else:
                        doc = docx.Document(file_path)
                    
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                    return text
                except ImportError:
                    print("python-docx not installed. Install with: pip install python-docx")
                    return ""
            
            elif file_extension == '.csv':
                try:
                    import pandas as pd
                    if file_content:
                        import io
                        csv_file = io.StringIO(file_content.decode('utf-8', errors='ignore'))
                        df = pd.read_csv(csv_file)
                    else:
                        df = pd.read_csv(file_path)
                    
                    # Convert DataFrame to text representation
                    text = df.to_string(index=False)
                    return text
                except ImportError:
                    print("pandas not installed. Install with: pip install pandas")
                    return ""
            
            else:
                print(f"Unsupported file format: {file_extension}")
                return ""
                
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in text using tiktoken"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception as e:
            print(f"Error counting tokens: {e}")
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    def create_chunks(self, text: str, source: str, title: str = None) -> List[Dict[str, Any]]:
        """Create chunks from text with metadata"""
        if not text.strip():
            return []
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only add non-empty chunks
                chunk_obj = {
                    "content": chunk.strip(),
                    "source": source,
                    "title": title or Path(source).stem,
                    "chunk_index": i,
                    "chunk_type": "text",
                    "token_count": self.count_tokens(chunk),
                    "metadata": {
                        "file_extension": Path(source).suffix.lower(),
                        "total_chunks": len(chunks)
                    }
                }
                chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            client = self.get_openai_client()
            response = client.embeddings.create(
                input=text[:8000],  # Limit text length for embedding
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def process_file(self, file_path: str, file_content: bytes = None) -> List[Dict[str, Any]]:
        """Process a single file and return chunks"""
        print(f"Processing file: {file_path}")
        
        # Extract text from file
        text = self.extract_text_from_file(file_path, file_content)
        
        if not text.strip():
            print(f"No text extracted from {file_path}")
            return []
        
        # Create chunks
        chunks = self.create_chunks(text, file_path)
        
        print(f"Created {len(chunks)} chunks from {file_path}")
        return chunks
    
    def upload_to_supabase(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upload all chunks to Supabase with embeddings"""
        print("Starting upload to Supabase...")
        
        # Get user's Supabase client
        supabase = self.get_supabase_client()
        
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

def main():
    """Main function for testing"""
    print("This is a library module. Use the web interface to process files.")
    print("Start the web server with: python web_server.py")

if __name__ == "__main__":
    main()