import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import tiktoken

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate required environment variable
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable. Please check your .env file.")

# Files to process
files_to_process = [
    "Txt File/01_Intro_Vad_Är_Axie_Studio.txt",
    "Txt File/02_Tjänster_Översikt.txt",
    "Txt File/03_Paket_och_Priser.txt",
    "Txt File/04_Teknik_och_Stabilitet.txt",
    "Txt File/05_Support_och_Samarbete.txt",
    "Txt File/06_Mobilappar_och_Ehandel.txt",
    "Txt File/07_FAQ_Vanliga_Frågor.txt",
    "Txt File/08_Kontakt_och_Företagsinfo.txt"
]

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def get_embedding(text, client):
    """Generate embedding for text"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def process_files():
    """Process all files and create chunks with embeddings"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks = []
    
    for file_path in files_to_process:
        if os.path.exists(file_path):
            print(f"Processing {file_path}...")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split into chunks
            chunks = text_splitter.split_text(content)
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = get_embedding(chunk, client)
                
                # Create chunk object
                chunk_data = {
                    "content": chunk,
                    "embedding": embedding,
                    "source": file_path,
                    "chunk_index": i,
                    "token_count": count_tokens(chunk),
                    "metadata": {
                        "file_name": os.path.basename(file_path),
                        "total_chunks": len(chunks)
                    }
                }
                
                all_chunks.append(chunk_data)
            
            print(f"Created {len(chunks)} chunks from {file_path}")
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    chunks = process_files()
    print("Processing complete!")