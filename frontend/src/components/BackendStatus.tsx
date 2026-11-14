import { useEffect, useState, useCallback } from 'react';
import { getHealth, HealthResponse } from '../services/api';

type ConnectionState = 'loading' | 'connected' | 'unreachable';

export const BackendStatus = () => {
  const [statusState, setStatusState] = useState<ConnectionState>('loading');
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>('');

  const checkConnection = useCallback(async () => {
    setStatusState('loading');
    setErrorMessage('');
    try {
      const data = await getHealth();
      setHealthData(data);
      setStatusState('connected');
    } catch (err: unknown) {
      setHealthData(null);
      setStatusState('unreachable');
      if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Failed to connect to backend.');
      }
    }
  }, []);

  useEffect(() => {
    checkConnection();
  }, [checkConnection]);

  return (
    <div
      role="status"
      aria-label="Backend Server Status"
      className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm"
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
          Backend Service Status
        </h3>
        <button
          type="button"
          onClick={checkConnection}
          className="text-xs text-indigo-600 hover:text-indigo-800 font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 rounded px-2 py-0.5"
          aria-label="Refresh backend status"
        >
          Refresh
        </button>
      </div>

      {statusState === 'loading' && (
        <div className="flex items-center space-x-2 text-slate-600 text-xs py-1">
          <svg
            className="animate-spin h-3.5 w-3.5 text-indigo-600"
            fill="none"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8v8H4z"
            />
          </svg>
          <span>Connecting to backend...</span>
        </div>
      )}

      {statusState === 'connected' && healthData && (
        <div className="flex items-center justify-between text-xs bg-emerald-50/70 border border-emerald-200 rounded-md p-2.5">
          <div className="flex items-center space-x-2">
            <span className="h-2 w-2 rounded-full bg-emerald-500 inline-block" aria-hidden="true" />
            <span className="font-semibold text-emerald-800">{healthData.service}</span>
          </div>
          <span className="font-mono text-emerald-700 uppercase font-semibold">
            {healthData.status}
          </span>
        </div>
      )}

      {statusState === 'unreachable' && (
        <div className="text-xs bg-rose-50/80 border border-rose-200 rounded-md p-2.5 space-y-2">
          <div className="flex items-center space-x-1.5 text-rose-800 font-semibold">
            <span className="h-2 w-2 rounded-full bg-rose-500 inline-block" aria-hidden="true" />
            <span>Backend Unreachable</span>
          </div>
          <p className="text-[11px] text-rose-700 leading-snug">
            {errorMessage || 'FastAPI server not responding on port 8000.'}
          </p>
          <button
            type="button"
            onClick={checkConnection}
            className="w-full text-center px-2 py-1 bg-rose-600 hover:bg-rose-700 text-white rounded text-[11px] font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-rose-500"
          >
            Retry Connection
          </button>
        </div>
      )}
    </div>
  );
};
