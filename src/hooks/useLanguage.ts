import { useState, useEffect } from 'react';
import { translations } from '../types/language';

// Secure local storage with validation and error handling
const STORAGE_KEY = 'axie-studio-rag-language';
const VALID_LANGUAGES = ['sv', 'en'];

const getStoredLanguage = (): string => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && VALID_LANGUAGES.includes(stored)) {
      return stored;
    }
  } catch (error) {
    console.warn('Failed to read language from localStorage:', error);
  }
  return 'sv'; // Default to Swedish
};

const setStoredLanguage = (language: string): void => {
  try {
    if (VALID_LANGUAGES.includes(language)) {
      localStorage.setItem(STORAGE_KEY, language);
    }
  } catch (error) {
    console.warn('Failed to save language to localStorage:', error);
  }
};

export const useLanguage = () => {
  const [currentLanguage, setCurrentLanguage] = useState<string>(getStoredLanguage);

  useEffect(() => {
    // Save language preference to localStorage securely
    setStoredLanguage(currentLanguage);
    
    // Update document language attribute
    document.documentElement.lang = currentLanguage;
    
    // Update document title based on language
    const title = currentLanguage === 'sv' ? 'Axie Studio RAG Filbehandlare' : 'Axie Studio RAG File Processor';
    document.title = title;
  }, [currentLanguage]);

  const t = (key: string): string => {
    return translations[currentLanguage]?.[key] || translations['en']?.[key] || key;
  };

  const switchLanguage = (languageCode: string) => {
    if (VALID_LANGUAGES.includes(languageCode)) {
      setCurrentLanguage(languageCode);
    }
  };

  return {
    currentLanguage,
    t,
    switchLanguage
  };
};