import React, { useState } from 'react';
import { X, Copy, Database, CheckCircle, ExternalLink, Play, Loader2 } from 'lucide-react';
import { useLanguage } from '../hooks/useLanguage';

interface SqlSetupModalProps {
  onClose: () => void;
  credentials?: {
    openai_api_key: string;
    supabase_url: string;
    supabase_service_key: string;
  } | null;
}

const SqlSetupModal: React.FC<SqlSetupModalProps> = ({ onClose, credentials }) => {
  const { t } = useLanguage();
  const [copied, setCopied] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState<{
    success: boolean;
    message: string;
    error?: string;
  } | null>(null);

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

  const executeSQL = async () => {
    if (!credentials) {
      setExecutionResult({
        success: false,
        message: 'Credentials not available. Please configure your credentials first.',
        error: 'No credentials provided'
      });
      return;
    }

    setIsExecuting(true);
    setExecutionResult(null);

    try {
      const response = await fetch('/api/setup-database', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          supabase_url: credentials.supabase_url,
          supabase_service_key: credentials.supabase_service_key,
          sql_commands: sqlSchema
        }),
      });

      const result = await response.json();
      
      if (result.success) {
        setExecutionResult({
          success: true,
          message: 'Database setup completed successfully! Your RAG system is ready to use.'
        });
      } else {
        setExecutionResult({
          success: false,
          message: result.message || 'Failed to setup database',
          error: result.error
        });
      }
    } catch (error) {
      setExecutionResult({
        success: false,
        message: 'Failed to connect to setup service',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setIsExecuting(false);
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
            <div className="flex gap-2">
              {credentials && (
                <button
                  onClick={executeSQL}
                  disabled={isExecuting}
                  className="flex items-center px-4 py-2 bg-green-600 text-white border border-green-500 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isExecuting ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Executing...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Auto Setup
                    </>
                  )}
                </button>
              )}
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
          </div>
          
          <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <pre className="text-white text-sm font-mono whitespace-pre-wrap">
              {sqlSchema}
            </pre>
          </div>
        </div>

        {/* Execution Result */}
        {executionResult && (
          <div className={`mt-4 p-4 border rounded-lg ${
            executionResult.success 
              ? 'bg-green-900/20 border-green-500/30' 
              : 'bg-red-900/20 border-red-500/30'
          }`}>
            <div className="flex items-center mb-2">
              {executionResult.success ? (
                <CheckCircle className="w-5 h-5 text-green-400 mr-2" />
              ) : (
                <X className="w-5 h-5 text-red-400 mr-2" />
              )}
              <h4 className={`font-medium ${
                executionResult.success ? 'text-green-400' : 'text-red-400'
              }`}>
                {executionResult.success ? 'Success!' : 'Error'}
              </h4>
            </div>
            <p className="text-white/90 text-sm">{executionResult.message}</p>
            {executionResult.error && (
              <p className="text-white/70 text-xs mt-2">Details: {executionResult.error}</p>
            )}
          </div>
        )}

        {/* Auto Setup Notice */}
        {credentials && !executionResult && (
          <div className="mt-4 p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg">
            <div className="flex items-center mb-2">
              <Play className="w-5 h-5 text-blue-400 mr-2" />
              <h4 className="font-medium text-blue-400">Automatic Setup Available</h4>
            </div>
            <p className="text-white/90 text-sm">
              Since you have configured your Supabase credentials, you can use the "Auto Setup" button 
              to automatically execute these SQL commands in your database.
            </p>
          </div>
        )}

        {!credentials && (
          <div className="mt-4 p-4 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
            <div className="flex items-center mb-2">
              <ExternalLink className="w-5 h-5 text-yellow-400 mr-2" />
              <h4 className="font-medium text-yellow-400">Manual Setup Required</h4>
            </div>
            <p className="text-white/90 text-sm">
              Configure your Supabase credentials first to enable automatic database setup.
            </p>
          </div>
        )}

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