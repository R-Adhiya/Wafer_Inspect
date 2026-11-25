import React from 'react';
import { Link, NavLink } from 'react-router-dom';

export const Navbar: React.FC = () => {
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        <Link
          to="/"
          className="flex items-center space-x-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-sm shadow-sm">
            WI
          </div>
          <div>
            <h1 className="text-base font-bold text-slate-900 leading-tight">
              WaferInspect
            </h1>
            <p className="text-[11px] text-slate-500 font-medium hidden sm:block">
              Automated Wafer Defect Inspection &amp; Measurement
            </p>
          </div>
        </Link>

        <nav aria-label="Primary Navigation" className="flex items-center space-x-2">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                isActive
                  ? 'bg-slate-900 text-white'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`
            }
          >
            Dashboard
          </NavLink>
          <NavLink
            to="/history"
            className={({ isActive }) =>
              `px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                isActive
                  ? 'bg-slate-900 text-white'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`
            }
          >
            History
          </NavLink>
          <NavLink
            to="/experiments/noise"
            className={({ isActive }) =>
              `px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                isActive
                  ? 'bg-slate-900 text-white'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`
            }
          >
            Noise Experiment
          </NavLink>
          <NavLink
            to="/inspection/new"
            className={({ isActive }) =>
              `px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                isActive
                  ? 'bg-indigo-600 text-white'
                  : 'text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200/80'
              }`
            }
          >
            New Inspection
          </NavLink>
        </nav>
      </div>
    </header>
  );
};
