import { useState, useEffect } from 'react';
import { translations } from '../types/language';

export const useLanguage = () => {
  const [currentLanguage, setCurrentLanguage] = useState<string>(() => {
    // Check localStorage first, then default to Swedish
    const saved = localStorage.getItem('rag-language');
    return saved || 'sv';
  });

  useEffect(() => {
    // Save language preference to localStorage
    localStorage.setItem('rag-language', currentLanguage);
    
    // Update document language attribute
    document.documentElement.lang = currentLanguage;
    
    // Update document title based on language
    const title = currentLanguage === 'sv' ? 'RAG Filbehandlare' : 'RAG File Processor';
    document.title = title;
  }, [currentLanguage]);

  const t = (key: string): string => {
    return translations[currentLanguage]?.[key] || translations['en']?.[key] || key;
  };

  const switchLanguage = (languageCode: string) => {
    setCurrentLanguage(languageCode);
  };

  return {
    currentLanguage,
    t,
    switchLanguage
  };
};