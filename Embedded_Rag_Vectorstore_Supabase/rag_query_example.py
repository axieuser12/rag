import os
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
import numpy as np

# Load environment variables
load_dotenv()

# Get credentials from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_HOST = os.getenv("SUPABASE_DB_HOST")
SUPABASE_DB_PORT = int(os.getenv("SUPABASE_DB_PORT", 5432))
SUPABASE_DB_NAME = os.getenv("SUPABASE_DB_NAME")
SUPABASE_DB_USER = os.getenv("SUPABASE_DB_USER")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")

# Validate required environment variables
required_vars = [OPENAI_API_KEY, SUPABASE_DB_HOST, SUPABASE_DB_NAME, SUPABASE_DB_USER, SUPABASE_DB_PASSWORD]
if not all(required_vars):
    raise ValueError("Missing required environment variables. Please check your .env file.")

def get_embedding(text, client):
    """Generate embedding for query text"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def search_similar_chunks(query, client, limit=5):
    """Search for similar chunks using vector similarity"""
    
    # Generate embedding for the query
    query_embedding = get_embedding(query, client)
    
    # Connect to Supabase database
    conn = psycopg2.connect(
        host=SUPABASE_DB_HOST,
        port=SUPABASE_DB_PORT,
        database=SUPABASE_DB_NAME,
        user=SUPABASE_DB_USER,
        password=SUPABASE_DB_PASSWORD
    )
    
    try:
        cursor = conn.cursor()
        
        # Search for similar vectors using cosine similarity
        search_query = """
        SELECT content, source, metadata, 
               1 - (embedding <=> %s::vector) as similarity
        FROM documents 
        ORDER BY embedding <=> %s::vector 
        LIMIT %s;
        """
        
        cursor.execute(search_query, (query_embedding, query_embedding, limit))
        results = cursor.fetchall()
        
        return results
        
    finally:
        conn.close()

def generate_answer(query, context_chunks, client):
    """Generate answer using retrieved context"""
    
    # Combine context chunks
    context = "\n\n".join([chunk[0] for chunk in context_chunks])
    
    # Create prompt
    prompt = f"""
    Based on the following context, please answer the question. If the answer is not in the context, say so.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    # Generate response
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    
    return response.choices[0].message.content

def main():
    """Main RAG query function"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Example query
    query = "Vad är Axie Studio?"
    
    print(f"Query: {query}")
    print("-" * 50)
    
    # Search for similar chunks
    similar_chunks = search_similar_chunks(query, client)
    
    print("Similar chunks found:")
    for i, (content, source, metadata, similarity) in enumerate(similar_chunks, 1):
        print(f"\n{i}. Similarity: {similarity:.3f}")
        print(f"Source: {source}")
        print(f"Content: {content[:200]}...")
    
    print("\n" + "=" * 50)
    
    # Generate answer
    answer = generate_answer(query, similar_chunks, client)
    print(f"Generated Answer:\n{answer}")

if __name__ == "__main__":
    main()