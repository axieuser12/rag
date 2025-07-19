import React, { useState, useCallback } from 'react';
import { UploadCloud, FileText, XCircle } from 'lucide-react';

interface FileUploaderProps {
  onFilesSelected: (files: File[]) => void;
  selectedFiles: File[];
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFilesSelected, selectedFiles }) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const newFiles = Array.from(event.target.files);
      onFilesSelected(newFiles);
    }
  };

  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(false);
    if (event.dataTransfer.files) {
      const newFiles = Array.from(event.dataTransfer.files);
      onFilesSelected(newFiles);
    }
  }, [onFilesSelected]);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleRemoveFile = (fileToRemove: File) => {
    const updatedFiles = selectedFiles.filter(file => file !== fileToRemove);
    onFilesSelected(updatedFiles);
  };

  return (
    <div className="border-2 border-dashed border-white/30 rounded-lg p-6 text-center relative">
      <input
        type="file"
        multiple
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        onChange={handleFileChange}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        aria-label="Upload files"
      />
      <div
        className={`flex flex-col items-center justify-center p-4 rounded-lg transition-colors duration-200 ${
          isDragOver ? 'bg-white/10 border-white/50' : 'bg-transparent'
        }`}
      >
        <UploadCloud className="w-12 h-12 text-white/70 mb-3" />
        <p className="text-white/80 text-lg font-medium">Drag & drop files here, or click to browse</p>
        <p className="text-white/60 text-sm mt-1">Supported formats: TXT, PDF, DOC, DOCX, CSV (Max 16MB per file)</p>
      </div>

      {selectedFiles.length > 0 && (
        <div className="mt-6 text-left">
          <h3 className="text-white font-semibold mb-3">Selected Files:</h3>
          <ul className="space-y-2">
            {selectedFiles.map((file, index) => (
              <li key={file.name + file.size + index} className="flex items-center justify-between bg-white/10 p-3 rounded-md">
                <div className="flex items-center">
                  <FileText className="w-5 h-5 text-white/70 mr-2" />
                  <span className="text-white text-sm truncate">{file.name}</span>
                  <span className="text-white/60 text-xs ml-2">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                </div>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); handleRemoveFile(file); }}
                  className="text-white/70 hover:text-red-400 transition-colors"
                  aria-label={`Remove ${file.name}`}
                >
                  <XCircle className="w-5 h-5" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default FileUploader;