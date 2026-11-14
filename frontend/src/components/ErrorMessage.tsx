import React from 'react';

interface ErrorMessageProps {
  message: string;
  code?: string;
  onDismiss?: () => void;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({ message, code, onDismiss }) => {
  if (!message) return null;

  return (
    <div
      role="alert"
      aria-live="assertive"
      className="p-4 bg-rose-50/80 border border-rose-200 rounded-lg text-rose-900 transition-all"
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3">
          <svg
            className="w-5 h-5 text-rose-600 mt-0.5 shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
          <div>
            <p className="text-sm font-semibold text-rose-800">
              Upload Rejected
            </p>
            <p className="text-xs text-rose-700 mt-1 leading-relaxed">
              {message}
            </p>
            {code && (
              <span className="inline-block mt-2 font-mono text-[11px] px-2 py-0.5 bg-rose-100 text-rose-800 rounded border border-rose-200">
                Code: {code}
              </span>
            )}
          </div>
        </div>

        {onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            className="text-rose-400 hover:text-rose-600 focus:outline-none focus:ring-2 focus:ring-rose-500 rounded p-1"
            aria-label="Dismiss error message"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
};
