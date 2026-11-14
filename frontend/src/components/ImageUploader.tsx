import React, { useState, useRef, useEffect, useCallback } from 'react';

const ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'];
const MAX_SIZE_BYTES = 20 * 1024 * 1024; // 20 MB

interface ImageUploaderProps {
  id: string;
  title: string;
  required?: boolean;
  helperText?: string;
  file: File | null;
  onFileSelect: (file: File) => void;
  onFileRemove: () => void;
  clientError?: string | null;
  setClientError?: (error: string | null) => void;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  id,
  title,
  required = false,
  helperText,
  file,
  onFileSelect,
  onFileRemove,
  clientError,
  setClientError,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Generate thumbnail and read image dimensions in the browser
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      setDimensions(null);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    // Read natural dimensions
    const img = new Image();
    img.onload = () => {
      setDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.onerror = () => {
      // For formats browser might not natively decode (like some TIFF variants)
      setDimensions(null);
    };
    img.src = objectUrl;

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  const validateAndProcessFile = useCallback(
    (selectedFile: File) => {
      if (setClientError) setClientError(null);

      // 1. Extension check
      const dotIdx = selectedFile.name.lastIndexOf('.');
      const ext = dotIdx !== -1 ? selectedFile.name.slice(dotIdx).toLowerCase() : '';
      if (!ALLOWED_EXTENSIONS.includes(ext)) {
        if (setClientError) {
          setClientError(
            `Unsupported extension "${ext}". Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`
          );
        }
        return;
      }

      // 2. Size check
      if (selectedFile.size === 0) {
        if (setClientError) {
          setClientError('The selected file is empty (0 bytes).');
        }
        return;
      }
      if (selectedFile.size > MAX_SIZE_BYTES) {
        if (setClientError) {
          setClientError('File exceeds maximum upload size of 20 MB.');
        }
        return;
      }

      onFileSelect(selectedFile);
    },
    [onFileSelect, setClientError]
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label
          htmlFor={`${id}-input`}
          className="text-sm font-semibold text-slate-800 flex items-center space-x-1"
        >
          <span>{title}</span>
          {required && <span className="text-rose-500 font-bold" aria-hidden="true">*</span>}
        </label>
        {required && (
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Required
          </span>
        )}
      </div>

      {helperText && (
        <p className="text-xs text-slate-500 leading-normal">
          {helperText}
        </p>
      )}

      {clientError && (
        <p className="text-xs text-rose-600 font-medium" role="alert">
          {clientError}
        </p>
      )}

      {!file ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
          tabIndex={0}
          role="button"
          aria-label={`Upload ${title}. Drag and drop or press Enter to browse files`}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 ${
            isDragOver
              ? 'border-indigo-500 bg-indigo-50/50 scale-[1.01]'
              : 'border-slate-300 hover:border-slate-400 bg-slate-50/50'
          }`}
        >
          <input
            id={`${id}-input`}
            ref={fileInputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.tif,.tiff"
            onChange={handleInputChange}
            className="sr-only"
          />

          <div className="flex flex-col items-center justify-center space-y-2">
            <div className="p-3 bg-white border border-slate-200 rounded-full shadow-sm text-slate-600">
              <svg
                className="w-6 h-6 text-indigo-600"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-slate-700">
                <span className="text-indigo-600 hover:text-indigo-700">Click to upload</span> or drag and drop
              </p>
              <p className="text-xs text-slate-500 mt-0.5">
                PNG, JPEG, or TIFF up to 20 MB
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex items-center space-x-4">
          {/* Thumbnail preview */}
          <div className="w-20 h-20 bg-slate-100 rounded-lg border border-slate-200 overflow-hidden flex items-center justify-center shrink-0">
            {previewUrl ? (
              <img
                src={previewUrl}
                alt={`Preview of ${file.name}`}
                className="w-full h-full object-contain"
              />
            ) : (
              <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            )}
          </div>

          {/* Details */}
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-slate-900 truncate" title={file.name}>
              {file.name}
            </p>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-1 text-xs text-slate-500">
              <span>{formatFileSize(file.size)}</span>
              {dimensions ? (
                <span>
                  {dimensions.width} &times; {dimensions.height} px
                </span>
              ) : (
                <span className="italic text-slate-400">Dimensions loading...</span>
              )}
            </div>
          </div>

          {/* Remove Button */}
          <button
            type="button"
            onClick={onFileRemove}
            className="px-3 py-1.5 text-xs font-medium text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-rose-500"
            aria-label={`Remove ${file.name}`}
          >
            Remove
          </button>
        </div>
      )}
    </div>
  );
};
