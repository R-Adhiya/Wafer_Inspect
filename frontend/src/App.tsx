import { useEffect, useState, useCallback } from 'react';
import { getHealth, HealthResponse } from './services/api';

type ConnectionState = 'loading' | 'connected' | 'unreachable';

export default function App() {
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
        setErrorMessage('Failed to connect to backend server.');
      }
    }
  }, []);

  useEffect(() => {
    checkConnection();
  }, [checkConnection]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col justify-between font-sans antialiased">
      {/* Header Landmark */}
      <header className="bg-white border-b border-slate-200 py-6 px-6 sm:px-10">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">
            WaferInspect
          </h1>
          <p className="text-sm text-slate-600 mt-1 font-normal">
            Automated Wafer Defect Inspection & Measurement
          </p>
        </div>
      </header>

      {/* Main Content Landmark */}
      <main className="flex-1 max-w-4xl w-full mx-auto p-6 sm:p-10 flex flex-col justify-center">
        <section
          aria-labelledby="backend-status-heading"
          className="bg-white border border-slate-200 rounded-xl p-8 shadow-sm transition-all"
        >
          <div className="flex items-center justify-between border-b border-slate-100 pb-4 mb-6">
            <h2
              id="backend-status-heading"
              className="text-lg font-semibold text-slate-900"
            >
              Backend System Status
            </h2>
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
              Phase 1 Prototype
            </span>
          </div>

          <div role="status" aria-live="polite" className="space-y-4">
            {statusState === 'loading' && (
              <div className="flex items-center space-x-3 py-6 px-4 bg-slate-50 border border-slate-200 rounded-lg text-slate-600">
                <svg
                  className="animate-spin h-5 w-5 text-indigo-600"
                  xmlns="http://www.w3.org/2000/svg"
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
                <span className="text-sm font-medium">
                  Connecting to WaferInspect backend...
                </span>
              </div>
            )}

            {statusState === 'connected' && healthData && (
              <div className="p-5 bg-emerald-50/70 border border-emerald-200 rounded-lg">
                <div className="flex items-center space-x-2 text-emerald-800 mb-3">
                  <span
                    className="h-2.5 w-2.5 rounded-full bg-emerald-500 inline-block"
                    aria-hidden="true"
                  />
                  <span className="text-sm font-semibold">
                    Backend Connected
                  </span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs mt-3 pt-3 border-t border-emerald-100">
                  <div>
                    <span className="text-slate-500 block uppercase font-medium tracking-wide">
                      Service Name
                    </span>
                    <span className="font-mono text-slate-800 text-sm font-semibold">
                      {healthData.service}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block uppercase font-medium tracking-wide">
                      Health Status
                    </span>
                    <span className="font-mono text-emerald-700 text-sm font-semibold uppercase">
                      {healthData.status}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {statusState === 'unreachable' && (
              <div className="p-5 bg-rose-50/70 border border-rose-200 rounded-lg">
                <div className="flex items-center space-x-2 text-rose-800 mb-2">
                  <span
                    className="h-2.5 w-2.5 rounded-full bg-rose-500 inline-block"
                    aria-hidden="true"
                  />
                  <span className="text-sm font-semibold">
                    Backend Unreachable
                  </span>
                </div>
                <p className="text-xs text-rose-700 leading-relaxed">
                  Unable to establish a connection to the WaferInspect backend service. Please verify that the FastAPI backend is running on port 8000.
                </p>
                {errorMessage && (
                  <p className="text-xs font-mono text-rose-600/80 mt-1">
                    Details: {errorMessage}
                  </p>
                )}
                <div className="mt-4 pt-3 border-t border-rose-100 flex justify-end">
                  <button
                    type="button"
                    onClick={checkConnection}
                    className="inline-flex items-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium rounded-lg shadow-sm transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    aria-label="Retry connection to backend service"
                  >
                    Retry Connection
                  </button>
                </div>
              </div>
            )}
          </div>
        </section>
      </main>

      {/* Footer Landmark */}
      <footer className="border-t border-slate-200 bg-white py-4 px-6 text-center">
        <p className="text-xs text-slate-500 font-normal">
          Software prototype using wafer-like grayscale imagery. Not an industrial inspection system.
        </p>
      </footer>
    </div>
  );
}
