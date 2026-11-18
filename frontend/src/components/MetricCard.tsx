import React from 'react';

export interface MetricCardProps {
  label: string;
  value?: React.ReactNode;
  subtext?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subtext,
  children,
  className = '',
}) => {
  return (
    <div
      className={`bg-white border border-slate-200 rounded-xl p-4 shadow-xs flex flex-col justify-between space-y-1.5 ${className}`}
    >
      <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
        {label}
      </span>
      {value !== undefined && (
        <div className="text-lg font-bold font-mono text-slate-900 leading-tight">
          {value}
        </div>
      )}
      {children}
      {subtext && (
        <div className="text-[11px] text-slate-500 leading-snug">
          {subtext}
        </div>
      )}
    </div>
  );
};
