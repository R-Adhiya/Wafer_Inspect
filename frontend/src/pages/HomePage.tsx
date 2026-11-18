import React from 'react';
import { Link } from 'react-router-dom';
import { BackendStatus } from '../components/BackendStatus';

export const HomePage: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-8 py-4">
      <section aria-label="System Health">
        <BackendStatus />
      </section>

      <section
        aria-labelledby="home-hero-heading"
        className="bg-white border border-slate-200 rounded-2xl p-8 sm:p-10 shadow-xs space-y-6"
      >
        <div className="space-y-3">
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
            Classical CV Inspection Prototype
          </span>
          <h2
            id="home-hero-heading"
            className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900"
          >
            Wafer Defect Inspection, Measurement &amp; Spatial Pattern Diagnosis
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed max-w-2xl">
            Upload a grayscale wafer image (optionally paired with a golden reference) to run
            deterministic preprocessing, die-to-die self-reference or golden-image detection,
            half-maximum boundary refinement, severity classification, and Monte Carlo spatial
            pattern diagnosis.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 pt-1">
          <Link
            to="/inspection/new"
            className="inline-flex items-center px-5 py-2.5 rounded-lg text-sm font-semibold bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            Start New Inspection
          </Link>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4 border-t border-slate-100 text-xs text-slate-600">
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1">
            <h3 className="font-bold text-slate-900">1. Dual Detection Paths</h3>
            <p>Golden-image phase-correlated subtraction or automatic die-to-die self-reference.</p>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1">
            <h3 className="font-bold text-slate-900">2. Refined Measurements</h3>
            <p>Half-maximum edge localisation, oriented scratch geometry, and severity overlays.</p>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1">
            <h3 className="font-bold text-slate-900">3. Spatial Pattern Diagnosis</h3>
            <p>Seeded Monte Carlo significance tests for edge rings, donuts, clusters, and scratches.</p>
          </div>
        </div>
      </section>
    </div>
  );
};
