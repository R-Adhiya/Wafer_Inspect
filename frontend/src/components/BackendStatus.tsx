import { useEffect, useState } from 'react';
import { CheckCircle2, XCircle, RefreshCw, Server } from 'lucide-react';
import { getHealth } from '../services/api';
import { HealthResponse } from '../types/health';

export const BackendStatus = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getHealth();
      setHealth(data);
    } catch (err: any) {
      setError('Unable to connect to the WaferInspect backend. Make sure the FastAPI server is running.');
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Server className="w-5 h-5 text-slate-400" />
          <h3 className="font-semibold text-slate-200">Backend Connection Status</h3>
        </div>
        <button
          onClick={fetchStatus}
          disabled={loading}
          className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition-colors flex items-center space-x-2 text-xs font-medium"
          title="Refresh Backend Status"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {loading ? (
        <div className="flex items-center space-x-3 p-4 bg-slate-950/50 border border-slate-800 rounded-lg text-slate-400 text-sm">
          <RefreshCw className="w-4 h-4 animate-spin text-sky-400" />
          <span>Checking backend...</span>
        </div>
      ) : error ? (
        <div className="p-4 bg-red-950/30 border border-red-800/50 rounded-lg space-y-2">
          <div className="flex items-center space-x-2 text-red-400 font-semibold text-sm">
            <XCircle className="w-5 h-5 shrink-0" />
            <span>Backend Unavailable</span>
          </div>
          <p className="text-xs text-red-300/80 leading-relaxed pl-7">
            {error}
          </p>
        </div>
      ) : health ? (
        <div className="p-4 bg-emerald-950/30 border border-emerald-800/50 rounded-lg space-y-3">
          <div className="flex items-center space-x-2 text-emerald-400 font-semibold text-sm">
            <CheckCircle2 className="w-5 h-5 shrink-0" />
            <span>Backend Connected</span>
          </div>
          <div className="grid grid-cols-3 gap-2 pl-7 text-xs">
            <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
              <span className="text-slate-400 block text-[10px]">STATUS</span>
              <span className="font-mono text-emerald-400 font-medium">{health.status}</span>
            </div>
            <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
              <span className="text-slate-400 block text-[10px]">SERVICE</span>
              <span className="font-mono text-slate-200 font-medium truncate block">{health.service}</span>
            </div>
            <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
              <span className="text-slate-400 block text-[10px]">VERSION</span>
              <span className="font-mono text-sky-400 font-medium">{health.version}</span>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};
