import React from 'react';
import { languages } from '../types/language';
import { useLanguage } from '../hooks/useLanguage';

const LanguageToggle: React.FC = () => {
  const { currentLanguage, switchLanguage } = useLanguage();

  return (
    <div className="language-toggle">
      <div className="toggle-container">
        {languages.map((language) => (
          <button
            key={language.code}
            onClick={() => switchLanguage(language.code)}
            className={`toggle-option ${currentLanguage === language.code ? 'active' : ''}`}
            aria-label={`Switch to ${language.name}`}
          >
            <span className="flag">{language.flag}</span>
            <span className="name">{language.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default LanguageToggle;