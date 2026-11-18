import React from 'react';
import { SeverityBadge, SeverityLevel } from './SeverityBadge';

export interface EmptyStateProps {
  title?: string;
  description?: string;
  severity?: SeverityLevel;
  severityReason?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No defects found',
  description = 'All candidate regions were either below the noise ceiling or filtered out as sensor noise.',
  severity = 'NONE',
  severityReason,
}) => {
  return (
    <div className="p-8 text-center space-y-2.5" role="status">
      <div className="flex justify-center">
        <SeverityBadge severity={severity} reason={severityReason} size="md" />
      </div>
      <p className="text-sm font-semibold text-emerald-800">{title}</p>
      <p className="text-xs text-slate-500 max-w-md mx-auto leading-relaxed">
        {severityReason || description}
      </p>
    </div>
  );
};
