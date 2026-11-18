import React from 'react';

export const Navbar: React.FC = () => {
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-sm shadow-sm">
            WI
          </div>
          <div>
            <h1 className="text-base font-bold text-slate-900 leading-tight">
              WaferInspect
            </h1>
            <p className="text-[11px] text-slate-500 font-medium">
              Automated Wafer Defect Inspection &amp; Measurement
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <span className="text-[11px] font-semibold text-indigo-700 bg-indigo-50 border border-indigo-200/80 px-2.5 py-0.5 rounded-full">
            CV Inspection Pipeline
          </span>
        </div>
      </div>
    </header>
  );
};
