-- Create a function for vector similarity search
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id uuid,
  content text,
  source text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    documents.id,
    documents.content,
    documents.source,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  FROM documents
  WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
  ORDER BY documents.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_embedding_cosine 
ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_documents_source 
ON documents (source);

CREATE INDEX IF NOT EXISTS idx_documents_metadata_chunk_type 
ON documents USING gin ((metadata->>'chunk_type'));

-- Create a view for easy querying
CREATE OR REPLACE VIEW document_summary AS
SELECT 
  id,
  content,
  source,
  metadata->>'title' as title,
  metadata->>'chunk_type' as chunk_type,
  metadata->>'filename' as filename,
  length(content) as content_length
FROM documents;