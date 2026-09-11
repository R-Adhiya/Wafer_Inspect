import { useState } from 'react';
import { Disc, Info, ArrowRight, ShieldCheck, Layers } from 'lucide-react';
import { BackendStatus } from '../components/BackendStatus';

export const DashboardPage = () => {
  const [infoMessage, setInfoMessage] = useState<string | null>(null);

  const handleStartInspection = () => {
    setInfoMessage('Inspection workflow will be available in the next phase.');
  };

  return (
    <div className="flex-1 bg-slate-950 overflow-y-auto p-8 space-y-8">
      {/* Header section */}
      <div className="flex justify-between items-start border-b border-slate-800 pb-6">
        <div>
          <div className="inline-flex items-center space-x-2 px-3 py-1 bg-sky-500/10 border border-sky-500/20 rounded-full text-xs font-semibold text-sky-400 mb-3">
            <Layers className="w-3.5 h-3.5" />
            <span>Phase 1 — Project Foundation</span>
          </div>
          <h1 className="text-3xl font-extrabold text-slate-100 tracking-tight">WaferInspect</h1>
          <p className="text-slate-400 text-sm font-medium mt-1">
            Automated Wafer Defect Inspection & Measurement
          </p>
        </div>

        {/* Primary CTA */}
        <button
          onClick={handleStartInspection}
          className="flex items-center space-x-2 px-5 py-2.5 bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold rounded-lg shadow-lg shadow-sky-500/20 transition-all text-sm"
        >
          <Disc className="w-4 h-4" />
          <span>Start New Inspection</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Info message banner when CTA is clicked */}
      {infoMessage && (
        <div className="p-4 bg-sky-950/40 border border-sky-800/60 rounded-xl text-sky-300 flex items-center justify-between animate-fadeIn">
          <div className="flex items-center space-x-3 text-sm">
            <Info className="w-5 h-5 text-sky-400 shrink-0" />
            <span>{infoMessage}</span>
          </div>
          <button
            onClick={() => setInfoMessage(null)}
            className="text-xs text-sky-400 hover:text-sky-200 font-medium px-2 py-1"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Project Description Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
        <h2 className="text-lg font-bold text-slate-200 mb-2">About WaferInspect</h2>
        <p className="text-slate-400 text-sm leading-relaxed">
          A software prototype demonstrating automated defect detection and measurement using wafer-like grayscale imagery.
        </p>
      </div>

      {/* Status & Architecture Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Backend Connection Card */}
        <BackendStatus />

        {/* Architectural Overview Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl space-y-4">
          <div className="flex items-center space-x-3">
            <ShieldCheck className="w-5 h-5 text-slate-400" />
            <h3 className="font-semibold text-slate-200">System Architecture & Pipeline Roadmap</h3>
          </div>
          <div className="space-y-2 text-xs">
            <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg flex justify-between items-center text-slate-300">
              <span className="font-medium">Phase 1: React + TS Frontend ↔ FastAPI REST + SQLite</span>
              <span className="text-emerald-400 font-semibold px-2 py-0.5 bg-emerald-500/10 border border-emerald-500/20 rounded">Active</span>
            </div>
            <div className="p-3 bg-slate-950/30 border border-slate-800/60 rounded-lg flex justify-between items-center text-slate-500">
              <span>Phase 2: Image Upload & Validation</span>
              <span className="text-slate-500">Upcoming</span>
            </div>
            <div className="p-3 bg-slate-950/30 border border-slate-800/60 rounded-lg flex justify-between items-center text-slate-500">
              <span>Phase 3-6: Preprocessing, CV Detection & Measurement</span>
              <span className="text-slate-500">Upcoming</span>
            </div>
            <div className="p-3 bg-slate-950/30 border border-slate-800/60 rounded-lg flex justify-between items-center text-slate-500">
              <span>Phase 7-8: Interactive Dashboard & Database History</span>
              <span className="text-slate-500">Upcoming</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
