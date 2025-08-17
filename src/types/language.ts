export interface Language {
  code: string;
  name: string;
  flag: string;
}

export const languages: Language[] = [
  { code: 'en', name: 'English', flag: '🇺🇸' }
];

export interface Translations {
  [key: string]: {
    [key: string]: string;
  };
}

export const translations: Translations = {
  en: {
    // Header
    title: 'Axie Studio RAG File Processor',
    subtitle: 'Upload your documents and create a searchable knowledge base using your own OpenAI and Supabase credentials',
    
    // Credentials
    credentialsConfigured: 'Credentials configured ✓',
    credentialsRequired: 'Credentials required',
    changeCredentials: 'Change Credentials',
    setupCredentials: 'Setup',
    updateCredentials: 'Update',
    
    // File Upload
    uploadFiles: 'Upload Files',
    dragDropFiles: 'Drag & drop files here, or click to browse',
    supportedFormats: 'Supported formats: TXT, PDF, DOC, DOCX, CSV (Max 16MB per file)',
    selectedFiles: 'Selected Files:',
    processFiles: 'Process Files',
    processing: 'Processing...',
    clearAll: 'Clear All',
    fileLimitReached: 'Limit Reached (5 files)',
    fileLimitWarning: 'You can process a maximum of 5 files at once',
    
    // Processing Status
    processingFiles: 'Processing Files',
    pleaseWait: 'Please wait while we process your files and create embeddings...',
    extractingText: 'Extracting Text',
    readingContents: 'Reading file contents',
    creatingEmbeddings: 'Creating Embeddings',
    usingOpenAI: 'Using OpenAI API',
    storingData: 'Storing Data',
    uploadingSupabase: 'Uploading to Supabase',
    processingTime: 'This process may take a few minutes depending on file size and number of files',
    
    // Results
    processingComplete: 'Processing Complete!',
    processingFailed: 'Processing Failed',
    filesProcessed: 'Files Processed',
    chunksCreated: 'Chunks Created',
    uploadedToSupabase: 'Uploaded to Supabase',
    
    // Features
    secure: 'Secure',
    secureDesc: 'Your credentials are never stored on our servers',
    multipleFormats: 'Multiple Formats',
    multipleFormatsDesc: 'Support for TXT, PDF, Word, and CSV files',
    yourDatabase: 'Your Database',
    yourDatabaseDesc: 'Data stored in your own Supabase instance',
    aiPowered: 'AI Powered',
    aiPoweredDesc: 'Smart processing with your OpenAI API',
    
    // Credentials Form
    configureCredentials: 'Configure Credentials',
    credentialsSecure: 'Your credentials are secure',
    credentialsNotStored: 'Credentials are only used for this session and are never stored on our servers. They are sent directly to your own OpenAI and Supabase services.',
    openaiApiKey: 'OpenAI API Key',
    supabaseProjectUrl: 'Supabase Project URL',
    supabaseServiceKey: 'Supabase Service Role Key',
    saveCredentials: 'Save Credentials',
    cancel: 'Cancel',
    
    // Errors and validation
    credentialsBeforeProcessing: 'Please configure your credentials before processing files',
    
    // Quick setup guide
    quickSetupGuide: 'Quick Setup Guide:',
    setupStep1: 'Create an OpenAI account and generate an API key',
    setupStep2: 'Create a Supabase project and enable the pgvector extension in SQL Editor',
    setupStep3: 'Run the SQL schema from the results page to create the documents table',
    setupStep4: 'Copy your project URL and service role key from Supabase settings',
    
    // SQL Setup Modal
    sqlSetup: 'SQL Setup',
    sqlSetupTitle: 'Supabase Database Setup',
    setupInstructions: 'Setup Instructions:',
    sqlStep1: 'Open your Supabase dashboard and navigate to SQL Editor',
    sqlStep2: 'Copy the SQL commands below',
    sqlStep3: 'Paste and run the commands in SQL Editor',
    sqlStep4: 'Verify that the "documents" table was created under Database > Tables',
    openSupabaseDashboard: 'Open Supabase Dashboard',
    sqlCommands: 'SQL Commands:',
    copySQL: 'Copy SQL',
    copied: 'Copied!',
    close: 'Close',
    whatThisCreates: 'What this creates:',
    documentsTable: 'Documents Table',
    documentsTableDesc: 'Stores your documents with vector embeddings',
    vectorIndex: 'Vector Index',
    vectorIndexDesc: 'Enables fast semantic search',
    searchFunction: 'Search Function',
    searchFunctionDesc: 'Ready-to-use function for finding similar documents',
    performanceIndexes: 'Performance Indexes',
    performanceIndexesDesc: 'Optimizes queries on source and date',
    important: 'Important',
    pgvectorWarning: 'Make sure the pgvector extension is enabled in your Supabase instance before running these commands.',
    
    // PWA
    installApp: 'Install App',
    installNow: 'Install Now',
    installLater: 'Later',
    installPrompt: 'Install Axie Studio RAG for quick access and offline functionality.',
    iosInstallInstructions: 'Install this app on your iPhone: tap the Share button and then "Add to Home Screen".',
    
    // Additional PWA translations
    downloadApp: 'Download App',
    installForOffline: 'Install for offline access',
    appInstalled: 'App installed successfully!'
  }
};