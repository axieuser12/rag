import React, { useState, useEffect } from 'react';
import { X, Eye, EyeOff, Key, Database, Brain, Shield, ExternalLink, AlertCircle } from 'lucide-react';
import { useLanguage } from '../hooks/useLanguage';

interface Credentials {
  openai_api_key: string;
  supabase_url: string;
  supabase_service_key: string;
}

interface CredentialsFormProps {
  onSubmit: (credentials: Credentials) => void;
  onClose: () => void;
  initialCredentials?: Credentials | null;
}

const CredentialsForm: React.FC<CredentialsFormProps> = ({ 
  onSubmit, 
  onClose, 
  initialCredentials 
}) => {
  const { t } = useLanguage();
  const [credentials, setCredentials] = useState<Credentials>({
    openai_api_key: initialCredentials?.openai_api_key || '',
    supabase_url: initialCredentials?.supabase_url || '',
    supabase_service_key: initialCredentials?.supabase_service_key || '',
  });

  const [showKeys, setShowKeys] = useState({
    openai_api_key: false,
    supabase_service_key: false,
  });

  const [errors, setErrors] = useState<Partial<Credentials>>({});

  const validateCredentials = () => {
    const newErrors: Partial<Credentials> = {};

    if (!credentials.openai_api_key) {
      newErrors.openai_api_key = 'OpenAI API key is required';
    } else if (!credentials.openai_api_key.startsWith('sk-')) {
      newErrors.openai_api_key = 'OpenAI API key should start with "sk-"';
    }

    if (!credentials.supabase_url) {
      newErrors.supabase_url = 'Supabase URL is required';
    } else if (!credentials.supabase_url.includes('supabase.co')) {
      newErrors.supabase_url = 'Please enter a valid Supabase URL';
    }

    if (!credentials.supabase_service_key) {
      newErrors.supabase_service_key = 'Supabase service key is required';
    } else if (!credentials.supabase_service_key.startsWith('eyJ')) {
      newErrors.supabase_service_key = 'Service key should start with "eyJ"';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateCredentials()) {
      onSubmit(credentials);
    }
  };

  const handleInputChange = (field: keyof Credentials, value: string) => {
    setCredentials(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: undefined }));
    }
  };

  const toggleShowKey = (field: keyof typeof showKeys) => {
    setShowKeys(prev => ({ ...prev, [field]: !prev[field] }));
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="glass-effect rounded-2xl p-8 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            <Key className="w-6 h-6 text-white mr-3" />
            <h2 className="text-2xl font-semibold text-white">{t('configureCredentials')}</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/20 rounded-full transition-colors"
          >
            <X className="w-5 h-5 text-white" />
          </button>
        </div>

        {/* Security Notice */}
        <div className="mb-6 p-4 bg-green-500/20 border border-green-400/30 rounded-lg">
          <div className="flex items-start">
            <Shield className="w-5 h-5 text-green-400 mr-3 mt-0.5" />
            <div>
              <p className="text-green-200 font-medium">{t('credentialsSecure')}</p>
              <p className="text-green-100 text-sm mt-1">{t('credentialsNotStored')}</p>
            </div>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* OpenAI API Key */}
          <div>
            <label className="flex items-center text-white font-medium mb-2">
              <Brain className="w-4 h-4 mr-2" />
              {t('openaiApiKey')}
              <a
                href="https://platform.openai.com/api-keys"
                target="_blank"
                rel="noopener noreferrer"
                className="ml-2 text-blue-300 hover:text-blue-200"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            </label>
            <div className="relative">
              <input
                type={showKeys.openai_api_key ? 'text' : 'password'}
                value={credentials.openai_api_key}
                onChange={(e) => handleInputChange('openai_api_key', e.target.value)}
                placeholder="sk-proj-..."
                className={`w-full p-3 pr-12 bg-white/10 border rounded-lg text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-white/30 ${
                  errors.openai_api_key ? 'border-red-400' : 'border-white/30'
                }`}
              />
              <button
                type="button"
                onClick={() => toggleShowKey('openai_api_key')}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-white/70 hover:text-white"
              >
                {showKeys.openai_api_key ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            {errors.openai_api_key && (
              <p className="text-red-300 text-sm mt-1 flex items-center">
                <AlertCircle className="w-4 h-4 mr-1" />
                {errors.openai_api_key}
              </p>
            )}
            <p className="text-white/60 text-sm mt-1">
              {t('openaiApiKey')} - {t('usingOpenAI')}
            </p>
          </div>

          {/* Supabase URL */}
          <div>
            <label className="flex items-center text-white font-medium mb-2">
              <Database className="w-4 h-4 mr-2" />
              {t('supabaseProjectUrl')}
              <a
                href="https://supabase.com/dashboard"
                target="_blank"
                rel="noopener noreferrer"
                className="ml-2 text-blue-300 hover:text-blue-200"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            </label>
            <input
              type="url"
              value={credentials.supabase_url}
              onChange={(e) => handleInputChange('supabase_url', e.target.value)}
              placeholder="https://your-project.supabase.co"
              className={`w-full p-3 bg-white/10 border rounded-lg text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-white/30 ${
                errors.supabase_url ? 'border-red-400' : 'border-white/30'
              }`}
            />
            {errors.supabase_url && (
              <p className="text-red-300 text-sm mt-1 flex items-center">
                <AlertCircle className="w-4 h-4 mr-1" />
                {errors.supabase_url}
              </p>
            )}
            <p className="text-white/60 text-sm mt-1">
              {t('supabaseProjectUrl')} - {t('yourDatabaseDesc')}
            </p>
          </div>

          {/* Supabase Service Key */}
          <div>
            <label className="flex items-center text-white font-medium mb-2">
              <Key className="w-4 h-4 mr-2" />
              {t('supabaseServiceKey')}
              <a
                href="https://supabase.com/dashboard"
                target="_blank"
                rel="noopener noreferrer"
                className="ml-2 text-blue-300 hover:text-blue-200"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            </label>
            <div className="relative">
              <input
                type={showKeys.supabase_service_key ? 'text' : 'password'}
                value={credentials.supabase_service_key}
                onChange={(e) => handleInputChange('supabase_service_key', e.target.value)}
                placeholder="eyJ..."
                className={`w-full p-3 pr-12 bg-white/10 border rounded-lg text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-white/30 ${
                  errors.supabase_service_key ? 'border-red-400' : 'border-white/30'
                }`}
              />
              <button
                type="button"
                onClick={() => toggleShowKey('supabase_service_key')}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-white/70 hover:text-white"
              >
                {showKeys.supabase_service_key ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            {errors.supabase_service_key && (
              <p className="text-red-300 text-sm mt-1 flex items-center">
                <AlertCircle className="w-4 h-4 mr-1" />
                {errors.supabase_service_key}
              </p>
            )}
            <p className="text-white/60 text-sm mt-1">
              {t('supabaseServiceKey')} - {t('storingData')}
            </p>
          </div>

          {/* Setup Instructions */}
          <div className="p-4 bg-blue-500/20 border border-blue-400/30 rounded-lg">
            <h3 className="text-blue-200 font-medium mb-2">{t('quickSetupGuide')}</h3>
            <ol className="text-blue-100 text-sm space-y-1 list-decimal list-inside">
              <li>{t('setupStep1')}</li>
              <li>{t('setupStep2')}</li>
              <li>{t('setupStep3')}</li>
              <li>{t('setupStep4')}</li>
            </ol>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4 pt-4">
            <button
              type="submit"
              className="flex-1 bg-white text-purple-600 py-3 px-6 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              {t('saveCredentials')}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="px-6 py-3 border border-white/30 text-white rounded-lg font-semibold hover:bg-white/10 transition-colors"
            >
              {t('cancel')}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default CredentialsForm;