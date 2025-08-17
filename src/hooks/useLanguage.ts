import { useState, useEffect } from 'react';
import { translations } from '../types/language';

export const useLanguage = () => {
  const [currentLanguage] = useState<string>('en');

  useEffect(() => {
    // Update document language attribute
    document.documentElement.lang = 'en';
    
    // Update document title
    document.title = 'Axie Studio RAG File Processor';
  }, []);

  const t = (key: string): string => {
    return translations['en']?.[key] || key;
  };

  const switchLanguage = () => {
    // No-op since we only have English now
  };

  return {
    currentLanguage: 'en',
    t,
    switchLanguage
  };
};