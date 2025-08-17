import React, { useState } from 'react';
import { X, Copy, Database, CheckCircle, ExternalLink } from 'lucide-react';
import { useLanguage } from '../hooks/useLanguage';

interface SqlSetupModalProps {
  onClose: () => void;
}

const SqlSetupModal: React.FC<SqlSetupModalProps> = ({ onClose }) => {
  const { t } = useLanguage();
  const [copied, setCopied] = useState(false);

  const sqlSchema = `-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table for RAG system
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

-- Create additional indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents (source);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents (created_at);

-- Create a function to search similar documents
CREATE OR REPLACE FUNCTION search_documents(
  query_embedding vector(1536),
  match_threshold float DEFAULT 0.78,
  match_count int DEFAULT 10
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
    1 - (documents.embedding <=> query_embedding) AS similarity
  FROM documents
  WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
  ORDER BY documents.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;`;

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(sqlSchema);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="glass-effect-dark rounded-2xl p-8 max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-white/20">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            <Database className="w-6 h-6 text-white mr-3" />
            <h2 className="text-2xl font-semibold text-white">{t('sqlSetupTitle')}</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/20 rounded-full transition-colors"
          >
            <X className="w-5 h-5 text-white" />
          </button>
        </div>

        {/* Instructions */}
        <div className="mb-6 p-4 bg-gray-800/50 border border-white/20 rounded-lg">
          <h3 className="text-white/90 font-medium mb-2">{t('setupInstructions')}</h3>
          <ol className="text-white/80 text-sm space-y-2 list-decimal list-inside">
            <li>{t('sqlStep1')}</li>
            <li>{t('sqlStep2')}</li>
            <li>{t('sqlStep3')}</li>
            <li>{t('sqlStep4')}</li>
          </ol>
          <div className="mt-3 flex items-center">
            <ExternalLink className="w-4 h-4 mr-2 text-white/70" />
            <a
              href="https://supabase.com/dashboard"
              target="_blank"
              rel="noopener noreferrer"
              className="text-white/70 hover:text-white text-sm underline"
            >
              {t('openSupabaseDashboard')}
            </a>
          </div>
        </div>

        {/* SQL Code Block */}
        <div className="relative">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-white font-medium">{t('sqlCommands')}</h3>
            <button
              onClick={copyToClipboard}
              className="flex items-center px-4 py-2 bg-black text-white border border-white/20 rounded-lg hover:bg-white hover:text-black transition-colors"
            >
              {copied ? (
                <>
                  <CheckCircle className="w-4 h-4 mr-2" />
                  {t('copied')}
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4 mr-2" />
                  {t('copySQL')}
                </>
              )}
            </button>
          </div>
          
          <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <pre className="text-white text-sm font-mono whitespace-pre-wrap">
              {sqlSchema}
            </pre>
          </div>
        </div>

        {/* Features Explanation */}
        <div className="mt-6 p-4 bg-gray-800/50 border border-white/20 rounded-lg">
          <h3 className="text-white/90 font-medium mb-2">{t('whatThisCreates')}</h3>
          <ul className="text-white/80 text-sm space-y-1">
            <li>• <strong>{t('documentsTable')}</strong> - {t('documentsTableDesc')}</li>
            <li>• <strong>{t('vectorIndex')}</strong> - {t('vectorIndexDesc')}</li>
            <li>• <strong>{t('searchFunction')}</strong> - {t('searchFunctionDesc')}</li>
            <li>• <strong>{t('performanceIndexes')}</strong> - {t('performanceIndexesDesc')}</li>
          </ul>
        </div>

        {/* Warning */}
        <div className="mt-4 p-4 bg-gray-800/50 border border-white/20 rounded-lg">
          <p className="text-white/80 text-sm">
            <strong>{t('important')}:</strong> {t('pgvectorWarning')}
          </p>
        </div>

        {/* Close Button */}
        <div className="flex justify-end mt-6">
          <button
            onClick={onClose}
            className="px-6 py-3 btn-primary rounded-lg font-semibold transition-colors"
          >
            {t('close')}
          </button>
        </div>
      </div>
    </div>
  );
};

export default SqlSetupModal;