import React, { useState, useCallback, useEffect } from 'react';
import { Upload, FileText, Database, CheckCircle, AlertCircle, Loader2, Brain, Zap, Settings, Key, Shield } from 'lucide-react';
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
}

interface Credentials {
  openai_api_key: string;
  supabase_url: string;
  supabase_service_key: string;
}

function App() {
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
    setFiles(selectedFiles);
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

      const response = await fetch('/api/process-files', {
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
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-6">
            <div className="p-4 rounded-full glass-effect">
              <Brain className="w-12 h-12 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-white mb-4">
            Public RAG File Processor
          </h1>
          <p className="text-xl text-white/80 max-w-2xl mx-auto">
            Upload your documents and create a searchable knowledge base using your own OpenAI and Supabase credentials
          </p>
        </div>

        {/* Credentials Status */}
        <div className="max-w-4xl mx-auto mb-8">
          <div className="glass-effect rounded-xl p-4 flex items-center justify-between">
            <div className="flex items-center">
              <Shield className="w-5 h-5 text-white mr-3" />
              <span className="text-white">
                {credentials ? 'Credentials configured ✓' : 'Credentials required'}
              </span>
            </div>
            <div className="flex gap-2">
              {credentials && (
                <button
                  onClick={handleClearCredentials}
                  className="px-4 py-2 text-sm border border-white/30 text-white rounded-lg hover:bg-white/10 transition-colors"
                >
                  Change Credentials
                </button>
              )}
              <button
                onClick={() => setShowCredentials(true)}
                className="flex items-center px-4 py-2 text-sm bg-white/20 text-white rounded-lg hover:bg-white/30 transition-colors"
              >
                <Key className="w-4 h-4 mr-2" />
                {credentials ? 'Update' : 'Setup'} Credentials
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
                <h2 className="text-2xl font-semibold text-white">Upload Files</h2>
              </div>
              
              <FileUploader 
                onFilesSelected={handleFilesSelected}
                selectedFiles={files}
              />

              {files.length > 0 && (
                <div className="mt-6 flex gap-4">
                  <button
                    onClick={handleProcess}
                    disabled={isProcessing || !credentials}
                    className="flex items-center px-6 py-3 bg-white text-purple-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Zap className="w-5 h-5 mr-2" />
                        Process Files
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={handleClear}
                    disabled={isProcessing}
                    className="px-6 py-3 border border-white/30 text-white rounded-lg font-semibold hover:bg-white/10 transition-colors disabled:opacity-50"
                  >
                    Clear All
                  </button>
                </div>
              )}

              {!credentials && files.length > 0 && (
                <div className="mt-4 p-4 bg-yellow-500/20 border border-yellow-400/30 rounded-lg">
                  <p className="text-yellow-200 text-center">
                    Please configure your credentials before processing files
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
                <h3 className="text-lg font-semibold text-white mb-2">Secure</h3>
                <p className="text-white/70">Your credentials are never stored on our servers</p>
              </div>
              
              <div className="glass-effect rounded-xl p-6 text-center">
                <FileText className="w-8 h-8 text-blue-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-white mb-2">Multiple Formats</h3>
                <p className="text-white/70">Support for TXT, PDF, Word, and CSV files</p>
              </div>
              
              <div className="glass-effect rounded-xl p-6 text-center">
                <Database className="w-8 h-8 text-purple-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-white mb-2">Your Database</h3>
                <p className="text-white/70">Data stored in your own Supabase instance</p>
              </div>
              
              <div className="glass-effect rounded-xl p-6 text-center">
                <Brain className="w-8 h-8 text-orange-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-white mb-2">AI Powered</h3>
                <p className="text-white/70">Smart processing with your OpenAI API</p>
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