import React from 'react';
import { CheckCircle, AlertCircle, FileText, Database, Brain, ExternalLink, Copy } from 'lucide-react';
import { useLanguage } from '../hooks/useLanguage';

interface ProcessingResult {
  success: boolean;
  message: string;
  chunks_created?: number;
  files_processed?: number;
  error?: string;
  upload_stats?: {
    successful_uploads: number;
    failed_uploads: number;
    embedding_errors?: number;
    total_chunks: number;
  };
  processing_errors?: string[];
}

interface ResultsDisplayProps {
  result: ProcessingResult;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ result }) => {
  const { t } = useLanguage();

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const sqlSchema = `-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table for RAG
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

-- Create additional indexes
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents (source);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents (created_at);`;

  return (
    <div className={`glass-effect rounded-2xl p-8 ${
      result.success ? 'border-white/30' : 'border-white/20'
    }`}>
      <div className="flex items-center mb-6">
        {result.success ? (
          <CheckCircle className="w-6 h-6 text-white mr-3" />
        ) : (
          <AlertCircle className="w-6 h-6 text-white mr-3" />
        )}
        <h2 className="text-2xl font-semibold text-white">
          {result.success ? t('processingComplete') : t('processingFailed')}
        </h2>
      </div>

      <div className="space-y-4">
        <p className="text-white/90 text-lg">
          {result.message}
        </p>

        {result.success && (
          <>
            <div className="grid md:grid-cols-3 gap-4 mt-6">
              <div className="bg-black/20 border border-white/10 rounded-lg p-4 text-center">
                <FileText className="w-8 h-8 text-white mx-auto mb-2" />
                <div className="text-2xl font-bold text-white">
                  {result.files_processed || 0}
                </div>
                <div className="text-white/70">{t('filesProcessed')}</div>
              </div>
              
              <div className="bg-black/20 border border-white/10 rounded-lg p-4 text-center">
                <Brain className="w-8 h-8 text-white mx-auto mb-2" />
                <div className="text-2xl font-bold text-white">
                  {result.chunks_created || 0}
                </div>
                <div className="text-white/70">{t('chunksCreated')}</div>
              </div>
              
              <div className="bg-black/20 border border-white/10 rounded-lg p-4 text-center">
                <Database className="w-8 h-8 text-white mx-auto mb-2" />
                <div className="text-2xl font-bold text-white">
                  {result.upload_stats?.successful_uploads || 0}
                </div>
                <div className="text-white/70">{t('uploadedToSupabase')}</div>
              </div>
            </div>

            {result.upload_stats && (
              <div className="mt-6 p-4 bg-gray-800/50 border border-white/20 rounded-lg">
                <h3 className="text-white/90 font-medium mb-2">Upload Statistics:</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-white/80">Successful:</span>
                    <span className="text-white font-bold ml-2">{result.upload_stats.successful_uploads}</span>
                  </div>
                  <div>
                    <span className="text-white/80">Failed:</span>
                    <span className="text-white font-bold ml-2">{result.upload_stats.failed_uploads}</span>
                  </div>
                  <div>
                    <span className="text-white/80">Total:</span>
                    <span className="text-white font-bold ml-2">{result.upload_stats.total_chunks}</span>
                  </div>
                  <div>
                    <span className="text-white/80">Success Rate:</span>
                    <span className="text-white font-bold ml-2">
                      {Math.round((result.upload_stats.successful_uploads / result.upload_stats.total_chunks) * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div className="mt-6 p-4 bg-gray-800/50 border border-white/20 rounded-lg">
              <p className="text-white/90 font-medium">✅ Success! Your RAG system is ready:</p>
              <ul className="text-white/80 mt-2 space-y-1 text-sm">
                <li>• Files processed and stored in your Supabase vector database</li>
                <li>• Embeddings generated using your OpenAI API</li>
                <li>• Ready for semantic search and AI-powered retrieval</li>
                <li>• Use the query functions to search your knowledge base</li>
              </ul>
            </div>

            <div className="mt-6 p-4 bg-gray-800/50 border border-white/20 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-white/90 font-medium">📋 Database Schema (if needed):</h3>
                <button
                  onClick={() => copyToClipboard(sqlSchema)}
                  className="flex items-center px-3 py-1 text-sm bg-black text-white border border-white/20 rounded hover:bg-white hover:text-black transition-colors"
                >
                  <Copy className="w-4 h-4 mr-1" />
                  Copy SQL
                </button>
              </div>
              <p className="text-white/80 text-sm">
                If you haven't set up your Supabase database yet, copy and run this SQL in your Supabase SQL editor.
              </p>
            </div>
          </>
        )}

        {result.processing_errors && result.processing_errors.length > 0 && (
          <div className="mt-4 p-4 bg-gray-800/50 border border-white/20 rounded-lg">
            <p className="text-white/90 font-medium">⚠️ Processing Warnings:</p>
            <ul className="text-white/80 mt-2 space-y-1 text-sm">
              {result.processing_errors.map((error, index) => (
                <li key={index}>• {error}</li>
              ))}
            </ul>
          </div>
        )}

        {result.error && (
          <div className="mt-4 p-4 bg-gray-800/50 border border-white/20 rounded-lg">
            <p className="text-white/90 font-medium">Error Details:</p>
            <p className="text-white/80 mt-1 text-sm">{result.error}</p>
          </div>
        )}

        {!result.success && (
          <div className="mt-6 p-4 bg-gray-800/50 border border-white/20 rounded-lg">
            <p className="text-white/90 font-medium">💡 Troubleshooting Tips:</p>
            <ul className="text-white/80 mt-2 space-y-1 text-sm">
              <li>• Verify your OpenAI API key is valid and has sufficient credits</li>
              <li>• Check that your Supabase URL and service key are correct</li>
              <li>• Ensure the pgvector extension is enabled in your Supabase database</li>
              <li>• Make sure the documents table exists (use the SQL schema above)</li>
              <li>• Try with smaller files if you're hitting rate limits</li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsDisplay;