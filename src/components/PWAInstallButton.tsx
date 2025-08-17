import React, { useState, useEffect } from 'react';
import { Download, Smartphone, X } from 'lucide-react';
import { useLanguage } from '../hooks/useLanguage';

interface BeforeInstallPromptEvent extends Event {
  readonly platforms: string[];
  readonly userChoice: Promise<{
    outcome: 'accepted' | 'dismissed';
    platform: string;
  }>;
  prompt(): Promise<void>;
}

const PWAInstallButton: React.FC = () => {
  const { t } = useLanguage();
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [showInstallPrompt, setShowInstallPrompt] = useState(false);
  const [isInstalled, setIsInstalled] = useState(false);
  const [isIOS, setIsIOS] = useState(false);
  const [isStandalone, setIsStandalone] = useState(false);

  useEffect(() => {
    // Check if app is already installed
    const checkIfInstalled = () => {
      const isStandaloneMode = window.matchMedia('(display-mode: standalone)').matches;
      const isIOSStandalone = (window.navigator as any).standalone === true;
      setIsStandalone(isStandaloneMode || isIOSStandalone);
      setIsInstalled(isStandaloneMode || isIOSStandalone);
    };

    // Check if iOS
    const checkIfIOS = () => {
      const isIOSDevice = /iPad|iPhone|iPod/.test(navigator.userAgent);
      setIsIOS(isIOSDevice);
    };

    checkIfInstalled();
    checkIfIOS();

    // Listen for the beforeinstallprompt event
    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e as BeforeInstallPromptEvent);
      setShowInstallPrompt(true);
    };

    // Listen for app installed event
    const handleAppInstalled = () => {
      setIsInstalled(true);
      setShowInstallPrompt(false);
      setDeferredPrompt(null);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, []);

  const handleInstallClick = async () => {
    if (!deferredPrompt) return;

    try {
      await deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;
      
      if (outcome === 'accepted') {
        setIsInstalled(true);
        setShowInstallPrompt(false);
      }
      
      setDeferredPrompt(null);
    } catch (error) {
      console.error('Error during installation:', error);
    }
  };

  const handleDismiss = () => {
    setShowInstallPrompt(false);
  };

  // Don't show if already installed or in standalone mode
  if (isInstalled || isStandalone) {
    return null;
  }

  // iOS install instructions
  if (isIOS && !isStandalone) {
    return (
      <div className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:max-w-sm z-50">
        <div className="glass-effect-dark rounded-xl p-4 border border-white/20">
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center">
              <Smartphone className="w-5 h-5 text-white mr-2" />
              <h3 className="text-white font-medium">Install App</h3>
            </div>
            <button
              onClick={handleDismiss}
              className="text-white/70 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          <p className="text-white/80 text-sm mb-3">
            Install this app on your iPhone: tap the Share button and then "Add to Home Screen".
          </p>
          <div className="flex items-center text-white/60 text-xs">
            <span>📱 Tap</span>
            <span className="mx-2">→</span>
            <span>Share</span>
            <span className="mx-2">→</span>
            <span>"Add to Home Screen"</span>
          </div>
        </div>
      </div>
    );
  }

  // Android/Desktop install button
  if (showInstallPrompt && deferredPrompt) {
    return (
      <div className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:max-w-sm z-50">
        <div className="glass-effect-dark rounded-xl p-4 border border-white/20">
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center">
              <Download className="w-5 h-5 text-white mr-2" />
              <h3 className="text-white font-medium">Install App</h3>
            </div>
            <button
              onClick={handleDismiss}
              className="text-white/70 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          <p className="text-white/80 text-sm mb-4">
            Install Axie Studio RAG for quick access and offline functionality.
          </p>
          <div className="flex gap-2">
            <button
              onClick={handleInstallClick}
              className="flex-1 btn-primary py-2 px-4 rounded-lg font-medium transition-colors text-sm"
            >
              <Download className="w-4 h-4 mr-2 inline" />
              Install Now
            </button>
            <button
              onClick={handleDismiss}
              className="px-4 py-2 btn-secondary rounded-lg font-medium transition-colors text-sm"
            >
              Later
            </button>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default PWAInstallButton;