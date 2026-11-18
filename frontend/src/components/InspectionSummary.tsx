import React, { useState } from 'react';
import { InspectionRunResponse } from '../services/api';
import { MetricCard } from './MetricCard';
import { SeverityBadge } from './SeverityBadge';

export interface InspectionSummaryProps {
  result: InspectionRunResponse;
}

export const InspectionSummary: React.FC<InspectionSummaryProps> = ({ result }) => {
  const [showSeverityReason, setShowSeverityReason] = useState(false);

  if (!result.summary) {
    return null;
  }

  const { summary, model, processing_time_s, original_stats } = result;

  return (
    <section aria-label="Inspection Summary" className="space-y-3">
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard
          label="Defects Detected"
          value={summary.defect_count}
          subtext={
            summary.defect_count === 0
              ? 'Clean inspection'
              : `Avg area: ${summary.average_defect_area_px} px`
          }
        />
        <MetricCard
          label="Total Defect Area"
          value={`${summary.total_defect_area_px} px`}
          subtext="Original pixel units"
        />
        <MetricCard
          label="Largest Defect"
          value={`${summary.largest_defect_area_px} px`}
          subtext={summary.defect_count > 0 ? 'Single largest region' : 'No regions'}
        />
        <MetricCard
          label="Severity"
          subtext={
            <button
              type="button"
              onClick={() => setShowSeverityReason((prev) => !prev)}
              aria-expanded={showSeverityReason}
              className="text-[11px] font-medium text-indigo-600 hover:text-indigo-800 underline underline-offset-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 rounded cursor-pointer"
            >
              {showSeverityReason ? 'Hide reason' : 'Why this severity?'}
            </button>
          }
        >
          <div className="pt-0.5">
            <SeverityBadge
              severity={summary.severity}
              reason={summary.severity_reason}
              size="md"
            />
          </div>
        </MetricCard>
        <MetricCard
          label="Processing Time"
          value={`${processing_time_s.toFixed(3)} s`}
          subtext={`${Math.round(processing_time_s * 1000)} ms total`}
        />
        <MetricCard
          label="Detection Method"
          value={
            <span className="text-sm font-sans font-bold text-slate-900">
              {model?.detection_method ?? 'Classical CV Baseline'}
            </span>
          }
          subtext={model ? `${model.name} v${model.version}` : 'WaferInspect-CV v1.0.0'}
        />
      </div>

      {showSeverityReason && (
        <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700 leading-relaxed">
          <strong className="text-slate-900">Prototype Severity Heuristic Basis: </strong>
          {summary.severity_reason}
          <span className="block text-[11px] text-slate-500 mt-1">
            Note: Severity tiers are a documented prototype heuristic based on wafer-area fraction and count escalation, not a fab standard.
          </span>
        </div>
      )}

      <div className="px-3.5 py-2 rounded-lg bg-amber-50/80 border border-amber-200 text-xs text-amber-900 flex flex-wrap items-center justify-between gap-2">
        <span>
          <strong>Calibration:</strong>{' '}
          {summary.pixel_size_um
            ? summary.calibration_status
            : `${summary.calibration_status} (uncalibrated, pixel units)`}
        </span>
        <span className="text-[11px] font-mono text-amber-800">
          Original frame: {original_stats.width} &times; {original_stats.height} px
        </span>
      </div>
    </section>
  );
};
