import React from 'react';
import { Loader2, FileText, Brain, Database } from 'lucide-react';

const ProcessingStatus: React.FC = () => {
  return (
    <div className="glass-effect rounded-2xl p-8">
      <div className="flex items-center mb-6">
        <Loader2 className="w-6 h-6 text-white mr-3 animate-spin" />
        <h2 className="text-2xl font-semibold text-white">Processing Files</h2>
      </div>

      <div className="space-y-4">
        <p className="text-white/90 text-lg">
          Please wait while we process your files and create embeddings...
        </p>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white/10 rounded-lg p-4 text-center">
            <FileText className="w-8 h-8 text-blue-400 mx-auto mb-2" />
            <div className="text-white/70">Extracting Text</div>
            <div className="text-sm text-white/50 mt-1">Reading file contents</div>
          </div>
          
          <div className="bg-white/10 rounded-lg p-4 text-center">
            <Brain className="w-8 h-8 text-purple-400 mx-auto mb-2" />
            <div className="text-white/70">Creating Embeddings</div>
            <div className="text-sm text-white/50 mt-1">Using OpenAI API</div>
          </div>
          
          <div className="bg-white/10 rounded-lg p-4 text-center">
            <Database className="w-8 h-8 text-green-400 mx-auto mb-2" />
            <div className="text-white/70">Storing Data</div>
            <div className="text-sm text-white/50 mt-1">Uploading to Supabase</div>
          </div>
        </div>

        <div className="mt-6 p-4 bg-blue-500/20 border border-blue-400/30 rounded-lg">
          <p className="text-blue-200 text-center">
            ⏳ This process may take a few minutes depending on file size and number of files
          </p>
        </div>
      </div>
    </div>
  );
};

export default ProcessingStatus;