export interface Language {
  code: string;
  name: string;
  flag: string;
}

export const languages: Language[] = [
  { code: 'sv', name: 'Svenska', flag: '🇸🇪' },
  { code: 'en', name: 'English', flag: '🇺🇸' }
];

export interface Translations {
  [key: string]: {
    [key: string]: string;
  };
}

export const translations: Translations = {
  sv: {
    // Header
    title: 'Offentlig RAG Filbehandlare',
    subtitle: 'Ladda upp dina dokument och skapa en sökbar kunskapsbas med dina egna OpenAI och Supabase-uppgifter',
    
    // Credentials
    credentialsConfigured: 'Uppgifter konfigurerade ✓',
    credentialsRequired: 'Uppgifter krävs',
    changeCredentials: 'Ändra uppgifter',
    setupCredentials: 'Konfigurera',
    updateCredentials: 'Uppdatera',
    
    // File Upload
    uploadFiles: 'Ladda upp filer',
    dragDropFiles: 'Dra och släpp filer här, eller klicka för att bläddra',
    supportedFormats: 'Format som stöds: TXT, PDF, DOC, DOCX, CSV (Max 16MB per fil)',
    selectedFiles: 'Valda filer:',
    processFiles: 'Bearbeta filer',
    processing: 'Bearbetar...',
    clearAll: 'Rensa alla',
    
    // Processing Status
    processingFiles: 'Bearbetar filer',
    pleaseWait: 'Vänta medan vi bearbetar dina filer och skapar inbäddningar...',
    extractingText: 'Extraherar text',
    readingContents: 'Läser filinnehåll',
    creatingEmbeddings: 'Skapar inbäddningar',
    usingOpenAI: 'Använder OpenAI API',
    storingData: 'Lagrar data',
    uploadingSupabase: 'Laddar upp till Supabase',
    processingTime: 'Denna process kan ta några minuter beroende på filstorlek och antal filer',
    
    // Results
    processingComplete: 'Bearbetning klar!',
    processingFailed: 'Bearbetning misslyckades',
    filesProcessed: 'Filer bearbetade',
    chunksCreated: 'Delar skapade',
    uploadedToSupabase: 'Uppladdade till Supabase',
    
    // Features
    secure: 'Säker',
    secureDesc: 'Dina uppgifter lagras aldrig på våra servrar',
    multipleFormats: 'Flera format',
    multipleFormatsDesc: 'Stöd för TXT, PDF, Word och CSV-filer',
    yourDatabase: 'Din databas',
    yourDatabaseDesc: 'Data lagras i din egen Supabase-instans',
    aiPowered: 'AI-driven',
    aiPoweredDesc: 'Smart bearbetning med din OpenAI API',
    
    // Credentials Form
    configureCredentials: 'Konfigurera uppgifter',
    credentialsSecure: 'Dina uppgifter är säkra',
    credentialsNotStored: 'Uppgifter används endast för denna session och lagras aldrig på våra servrar. De skickas direkt till dina egna OpenAI och Supabase-tjänster.',
    openaiApiKey: 'OpenAI API-nyckel',
    supabaseProjectUrl: 'Supabase projekt-URL',
    supabaseServiceKey: 'Supabase tjänstnyckel',
    saveCredentials: 'Spara uppgifter',
    cancel: 'Avbryt',
    
    // Errors and validation
    credentialsBeforeProcessing: 'Vänligen konfigurera dina uppgifter innan du bearbetar filer',
    
    // Quick setup guide
    quickSetupGuide: 'Snabb installationsguide:',
    setupStep1: 'Skapa ett OpenAI-konto och generera en API-nyckel',
    setupStep2: 'Skapa ett Supabase-projekt och aktivera pgvector-tillägget i SQL Editor',
    setupStep3: 'Kör SQL-schemat från resultatsidan för att skapa dokumenttabellen',
    setupStep4: 'Kopiera din projekt-URL och tjänstnyckel från Supabase-inställningar'
  },
  en: {
    // Header
    title: 'Public RAG File Processor',
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
    setupStep4: 'Copy your project URL and service role key from Supabase settings'
  }
};