import React from 'react';
import { Loader2, FileText, Brain, Database } from 'lucide-react';
import { useLanguage } from '../hooks/useLanguage';

const ProcessingStatus: React.FC = () => {
  const { t } = useLanguage();

  return (
    <div className="glass-effect rounded-2xl p-8">
      <div className="flex items-center mb-6">
        <Loader2 className="w-6 h-6 text-white mr-3 animate-spin" />
        <h2 className="text-2xl font-semibold text-white">{t('processingFiles')}</h2>
      </div>

      <div className="space-y-4">
        <p className="text-white/90 text-lg">{t('pleaseWait')}</p>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white/10 rounded-lg p-4 text-center">
            <FileText className="w-8 h-8 text-blue-400 mx-auto mb-2" />
            <div className="text-white/70">{t('extractingText')}</div>
            <div className="text-sm text-white/50 mt-1">{t('readingContents')}</div>
          </div>
          
          <div className="bg-white/10 rounded-lg p-4 text-center">
            <Brain className="w-8 h-8 text-purple-400 mx-auto mb-2" />
            <div className="text-white/70">{t('creatingEmbeddings')}</div>
            <div className="text-sm text-white/50 mt-1">{t('usingOpenAI')}</div>
          </div>
          
          <div className="bg-white/10 rounded-lg p-4 text-center">
            <Database className="w-8 h-8 text-green-400 mx-auto mb-2" />
            <div className="text-white/70">{t('storingData')}</div>
            <div className="text-sm text-white/50 mt-1">{t('uploadingSupabase')}</div>
          </div>
        </div>

        <div className="mt-6 p-4 bg-blue-500/20 border border-blue-400/30 rounded-lg">
          <p className="text-blue-200 text-center">⏳ {t('processingTime')}</p>
        </div>
      </div>
    </div>
  );
};

export default ProcessingStatus;