import React from 'react';
import { InspectionRunResponse } from '../services/api';

export interface InspectionDetailsCardProps {
  result: InspectionRunResponse;
}

export const InspectionDetailsCard: React.FC<InspectionDetailsCardProps> = ({ result }) => {
  const filename = result.test_filename || 'Uploaded Wafer Image';
  const dimensions = `${result.original_stats.width} × ${result.original_stats.height} px`;
  const mode = result.test_mode || 'L (grayscale)';
  const refUsed = Boolean(result.reference_used);
  const detectionPath = result.detection?.path ?? 'unknown';
  const detectionLabel = result.detection?.label ?? 'Classical CV Detection';
  const modelName = result.model?.name ?? 'WaferInspect-CV';
  const modelVersion = result.model?.version ?? '1.0.0';
  const calibrationStatus =
    result.summary?.calibration_status ?? 'uncalibrated: measurements are in image pixels';

  return (
    <section
      aria-labelledby="inspection-details-heading"
      className="bg-white border border-slate-200 rounded-xl shadow-xs overflow-hidden"
    >
      <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
        <h3
          id="inspection-details-heading"
          className="text-xs font-bold text-slate-800 uppercase tracking-wide"
        >
          Inspection Details
        </h3>
        <span className="font-mono text-[11px] text-slate-600 bg-white px-2 py-0.5 rounded border border-slate-200 select-all">
          ID: {result.upload_id}
        </span>
      </div>

      <dl className="p-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
        <div>
          <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Test Image Filename
          </dt>
          <dd className="mt-0.5 font-mono font-semibold text-slate-900 break-all">{filename}</dd>
        </div>

        <div>
          <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Dimensions &amp; Mode
          </dt>
          <dd className="mt-0.5 font-mono text-slate-800">
            {dimensions} ({mode})
          </dd>
        </div>

        <div>
          <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Golden Reference
          </dt>
          <dd className="mt-0.5 text-slate-800">
            {refUsed ? (
              <span className="inline-flex items-center gap-1.5">
                <span className="px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200 font-semibold text-[10px]">
                  Used
                </span>
                {result.reference_filename && (
                  <span className="font-mono text-[11px] text-slate-600 truncate max-w-[160px]">
                    {result.reference_filename}
                  </span>
                )}
              </span>
            ) : (
              <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200 font-semibold text-[10px]">
                No reference (single-wafer)
              </span>
            )}
          </dd>
        </div>

        <div>
          <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Detection Path
          </dt>
          <dd className="mt-0.5 text-slate-800">
            <span className="font-semibold text-slate-900">{detectionLabel}</span>{' '}
            <span className="font-mono text-[11px] text-slate-500">({detectionPath})</span>
          </dd>
        </div>

        <div>
          <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Model &amp; Version
          </dt>
          <dd className="mt-0.5 font-mono text-slate-800">
            {modelName} v{modelVersion}
          </dd>
        </div>

        <div className="sm:col-span-2">
          <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Calibration Status
          </dt>
          <dd className="mt-0.5 text-slate-800">{calibrationStatus}</dd>
        </div>

        <div>
          <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Processing Time
          </dt>
          <dd className="mt-0.5 font-mono font-semibold text-slate-900">
            {result.processing_time_s.toFixed(3)} s ({Math.round(result.processing_time_s * 1000)} ms)
          </dd>
        </div>
      </dl>
    </section>
  );
};
