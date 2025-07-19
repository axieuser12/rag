import React, { useState, useCallback, useEffect } from 'react';
import { Upload, FileText, Database, CheckCircle, AlertCircle, Loader2, Brain, Zap, Settings, Key, Shield } from 'lucide-react';
import { useLanguage } from './hooks/useLanguage';
import LanguageToggle from './components/LanguageToggle';
import FileUploader from './components/FileUploader.tsx';
import ProcessingStatus from './components/ProcessingStatus.tsx';
import ResultsDisplay from './components/ResultsDisplay.tsx';
import CredentialsForm from './components/CredentialsForm.tsx';

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

interface Credentials {
  openai_api_key: string;
  supabase_url: string;
  supabase_service_key: string;
}

function App() {
  const { t } = useLanguage();
  const [files, setFiles] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<ProcessingResult | null>(null);
  const [credentials, setCredentials] = useState<Credentials | null>(null);
  const [showCredentials, setShowCredentials] = useState(false);

  // Load credentials from localStorage on component mount
  useEffect(() => {
    const savedCredentials = localStorage.getItem('rag-credentials');
    if (savedCredentials) {
      try {
        const parsed = JSON.parse(savedCredentials);
        setCredentials(parsed);
      } catch (error) {
        console.error('Error parsing saved credentials:', error);
        localStorage.removeItem('rag-credentials');
      }
    }
  }, []);

  const handleFilesSelected = useCallback((selectedFiles: File[]) => {
    // Limit to 5 files maximum
    const limitedFiles = selectedFiles.slice(0, 5);
    setFiles(limitedFiles);
    setResult(null);
  }, []);

  const handleCredentialsSubmit = useCallback((creds: Credentials) => {
    // Save credentials to localStorage
    localStorage.setItem('rag-credentials', JSON.stringify(creds));
    setCredentials(creds);
    setShowCredentials(false);
  }, []);

  const handleProcess = async () => {
    if (files.length === 0) return;
    if (files.length > 5) return; // Additional safety check
    if (!credentials) {
      setShowCredentials(true);
      return;
    }

    setIsProcessing(true);
    setResult(null);

    try {
      const formData = new FormData();
      files.forEach((file, index) => {
        formData.append(`file_${index}`, file);
      });

      // Add credentials to the request
      formData.append('openai_api_key', credentials.openai_api_key);
      formData.append('supabase_url', credentials.supabase_url);
      formData.append('supabase_service_key', credentials.supabase_service_key);

      const response = await fetch('/process-files', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({
        success: false,
        message: 'Failed to process files',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClear = () => {
    setFiles([]);
    setResult(null);
  };

  const handleClearCredentials = () => {
    localStorage.removeItem('rag-credentials');
    setCredentials(null);
    setShowCredentials(true);
  };

  return (
    <div className="min-h-screen gradient-bg">
      <LanguageToggle />
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-6">
            <div className="p-4 rounded-full glass-effect">
              <Brain className="w-12 h-12 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-white mb-4">{t('title')}</h1>
          <p className="text-xl text-white/80 max-w-2xl mx-auto">{t('subtitle')}</p>
        </div>

        {/* Credentials Status */}
        <div className="max-w-4xl mx-auto mb-8">
          <div className="glass-effect rounded-xl p-4 flex items-center justify-between">
            <div className="flex items-center">
              <Shield className="w-5 h-5 text-white mr-3" />
              <span className="text-white">
                {credentials ? t('credentialsConfigured') : t('credentialsRequired')}
              </span>
            </div>
            <div className="flex gap-2">
              {credentials && (
                <button
                  onClick={handleClearCredentials}
                  className="px-4 py-2 text-sm border border-white/30 text-white rounded-lg hover:bg-white/10 transition-colors"
                >
                  {t('changeCredentials')}
                </button>
              )}
              <button
                onClick={() => setShowCredentials(true)}
                className="flex items-center px-4 py-2 text-sm bg-white/20 text-white rounded-lg hover:bg-white/30 transition-colors"
              >
                <Key className="w-4 h-4 mr-2" />
                {credentials ? t('updateCredentials') : t('setupCredentials')} {t('credentialsRequired').split(' ')[0]}
              </button>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto">
          <div className="grid gap-8">
            {/* File Upload Section */}
            <div className="glass-effect rounded-2xl p-8">
              <div className="flex items-center mb-6">
                <Upload className="w-6 h-6 text-white mr-3" />
                <h2 className="text-2xl font-semibold text-white">{t('uploadFiles')}</h2>
              </div>
              
              <FileUploader 
                onFilesSelected={handleFilesSelected}
                selectedFiles={files}
                maxFiles={maxFiles}
              />

              {files.length > 0 && (
                <div className="mt-6 flex gap-4">
                  <button
                    onClick={handleProcess}
                    disabled={isProcessing || !credentials || files.length > 5}
                    className="flex items-center px-6 py-3 bg-white text-purple-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        {t('processing')}
                      </>
                    ) : files.length > 5 ? (
                      <>
                        <AlertCircle className="w-5 h-5 mr-2" />
                        {t('fileLimitReached')}
                      </>
                    ) : (
                      <>
                        <Zap className="w-5 h-5 mr-2" />
                        {t('processFiles')}
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={handleClear}
                    disabled={isProcessing}
                    className="px-6 py-3 border border-white/30 text-white rounded-lg font-semibold hover:bg-white/10 transition-colors disabled:opacity-50"
                  >
                    {t('clearAll')}
                  </button>
                </div>
              )}

              {files.length > 5 && (
                <div className="mt-4 p-4 bg-orange-500/20 border border-orange-400/30 rounded-lg">
                  <p className="text-orange-200 text-center flex items-center justify-center">
                    <AlertCircle className="w-5 h-5 mr-2" />
                    {t('fileLimitWarning')}
                  </p>
                </div>
              )}

              {!credentials && files.length > 0 && (
                <div className="mt-4 p-4 bg-yellow-500/20 border border-yellow-400/30 rounded-lg">
                  <p className="text-yellow-200 text-center">
                    {t('credentialsBeforeProcessing')}
                  </p>
                </div>
              )}
            </div>

            {/* Processing Status */}
            {isProcessing && (
              <ProcessingStatus />
            )}

            {/* Results */}
            {result && (
              <ResultsDisplay result={result} />
            )}

            {/* Features */}
            <div className="grid md:grid-cols-4 gap-6">
              <div className="glass-effect rounded-xl p-6 text-center">
                <Shield className="w-8 h-8 text-green-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-white mb-2">{t('secure')}</h3>
                <p className="text-white/70">{t('secureDesc')}</p>
              </div>
              
              <div className="glass-effect rounded-xl p-6 text-center">
                <FileText className="w-8 h-8 text-blue-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-white mb-2">{t('multipleFormats')}</h3>
                <p className="text-white/70">{t('multipleFormatsDesc')}</p>
              </div>
              
              <div className="glass-effect rounded-xl p-6 text-center">
                <Database className="w-8 h-8 text-purple-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-white mb-2">{t('yourDatabase')}</h3>
                <p className="text-white/70">{t('yourDatabaseDesc')}</p>
              </div>
              
              <div className="glass-effect rounded-xl p-6 text-center">
                <Brain className="w-8 h-8 text-orange-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-white mb-2">{t('aiPowered')}</h3>
                <p className="text-white/70">{t('aiPoweredDesc')}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Credentials Modal */}
      {showCredentials && (
        <CredentialsForm
          onSubmit={handleCredentialsSubmit}
          onClose={() => setShowCredentials(false)}
          initialCredentials={credentials}
        />
      )}
    </div>
  );
}

export default App;