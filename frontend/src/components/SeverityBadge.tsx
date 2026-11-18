import React from 'react';

export type SeverityLevel = 'NONE' | 'LOW' | 'MEDIUM' | 'HIGH';

export interface SeverityBadgeProps {
  severity: SeverityLevel;
  reason?: string;
  size?: 'sm' | 'md';
}

const SEVERITY_CONFIG: Record<
  SeverityLevel,
  { tag: string; classes: string; label: string }
> = {
  NONE: {
    tag: '0',
    label: 'NONE',
    classes: 'bg-emerald-50 text-emerald-800 border border-emerald-300',
  },
  LOW: {
    tag: 'L',
    label: 'LOW',
    classes: 'bg-blue-50 text-blue-800 border border-dashed border-blue-500',
  },
  MEDIUM: {
    tag: 'M',
    label: 'MEDIUM',
    classes: 'bg-orange-50 text-orange-800 border-2 border-orange-500',
  },
  HIGH: {
    tag: 'H',
    label: 'HIGH',
    classes: 'bg-red-50 text-red-800 border-2 border-red-600 ring-1 ring-red-300',
  },
};

export const SeverityBadge: React.FC<SeverityBadgeProps> = ({
  severity,
  reason,
  size = 'sm',
}) => {
  const cfg = SEVERITY_CONFIG[severity] ?? SEVERITY_CONFIG.NONE;
  const paddingClass =
    size === 'md' ? 'px-2.5 py-1 text-xs' : 'px-2 py-0.5 text-[10px]';

  return (
    <span
      title={reason}
      className={`inline-flex items-center gap-1 rounded font-bold uppercase tracking-wide ${paddingClass} ${cfg.classes}`}
    >
      {severity !== 'NONE' && (
        <span className="font-mono opacity-80">[{cfg.tag}]</span>
      )}
      <span>{cfg.label}</span>
    </span>
  );
};
