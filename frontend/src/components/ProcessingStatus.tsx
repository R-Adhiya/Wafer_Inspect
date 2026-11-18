import React, { useEffect, useState } from 'react';
import {
  getInspectionStatus,
  getInspectionStatusById,
  InspectionStatusResponse,
} from '../services/api';
import { ErrorMessage } from './ErrorMessage';

export const DEFAULT_PIPELINE_STAGES: string[] = [
  'preparing_image',
  'preprocessing',
  'detecting',
  'segmenting',
  'filtering_noise',
  'measuring',
  'diagnosing_pattern',
  'generating_report',
];

const STAGE_LABELS: Record<string, string> = {
  preparing_image: 'Preparing Image & Configs',
  preprocessing: 'Preprocessing (Grayscale, Denoise, Contrast)',
  detecting: 'Detecting Anomalies & Thresholding',
  segmenting: 'Segmenting & Merging Fragments',
  filtering_noise: 'Filtering Sensor Noise',
  measuring: 'Refining & Measuring Defects',
  diagnosing_pattern: 'Diagnosing Wafer Spatial Pattern',
  generating_report: 'Generating Overlays & Saving Report',
};

export interface ProcessingStatusProps {
  uploadId?: string;
  inspectionId?: number;
  isActive: boolean;
  pollIntervalMs?: number;
  pollFn?: (uploadId: string) => Promise<InspectionStatusResponse>;
}

export const ProcessingStatus: React.FC<ProcessingStatusProps> = ({
  uploadId = '',
  inspectionId,
  isActive,
  pollIntervalMs = 300,
  pollFn,
}) => {
  const [statusData, setStatusData] = useState<InspectionStatusResponse | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);

  useEffect(() => {
    if (!isActive || (!uploadId && inspectionId == null)) {
      return;
    }

    let cancelled = false;

    const fetchStatus = async () => {
      try {
        const res = pollFn
          ? await pollFn(uploadId)
          : inspectionId != null
            ? await getInspectionStatusById(inspectionId)
            : await getInspectionStatus(uploadId);
        if (!cancelled) {
          setStatusData(res);
          setPollError(res.status === 'failed' ? res.error || 'Inspection failed.' : null);
        }
      } catch (err: unknown) {
        if (!cancelled) {
          const msg = err instanceof Error ? err.message : 'Unable to poll status.';
          setPollError(msg);
        }
      }
    };

    void fetchStatus();
    const timer = window.setInterval(() => {
      void fetchStatus();
    }, pollIntervalMs);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [uploadId, inspectionId, isActive, pollIntervalMs, pollFn]);

  const stages = statusData?.stages?.length ? statusData.stages : DEFAULT_PIPELINE_STAGES;
  const completedSet = new Set(statusData?.completed_stages ?? []);
  const currentStage = statusData?.current_stage ?? stages[0];
  const isFailed = statusData?.status === 'failed';
  const isCompleted = statusData?.status === 'completed';

  return (
    <div
      role="status"
      aria-live="polite"
      aria-atomic="false"
      className="p-5 bg-white border border-slate-200 rounded-xl shadow-xs space-y-4"
    >
      <div className="flex items-center justify-between gap-3 border-b border-slate-100 pb-3">
        <div className="flex items-center space-x-2.5">
          {!isCompleted && !isFailed && (
            <svg
              className="animate-spin h-4 w-4 text-indigo-600 shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
          )}
          <h4 className="text-sm font-bold text-slate-900">
            {isFailed
              ? 'Inspection Pipeline Failed'
              : isCompleted
                ? 'Inspection Pipeline Completed'
                : 'Executing Computer Vision Pipeline'}
          </h4>
        </div>
        <span className="text-[11px] font-mono text-slate-500">
          Stage: {currentStage}
        </span>
      </div>

      {isFailed && (
        <ErrorMessage
          title="Pipeline Execution Failed"
          message={statusData?.error || pollError || 'An error occurred during inspection.'}
        />
      )}

      {!isFailed && pollError && (
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-1.5">
          Status update delayed: {pollError}
        </p>
      )}

      <ol className="space-y-2" aria-label="Inspection pipeline stages">
        {stages.map((stage) => {
          const isDone = isCompleted || completedSet.has(stage);
          const isCurrent = !isDone && !isFailed && currentStage === stage;
          const stateLabel = isDone ? 'done' : isCurrent ? 'current' : 'pending';

          return (
            <li
              key={stage}
              data-stage={stage}
              data-stage-state={stateLabel}
              className={`flex items-center justify-between px-3 py-2 rounded-lg text-xs border transition-colors ${
                isDone
                  ? 'bg-emerald-50/70 border-emerald-200 text-emerald-900'
                  : isCurrent
                    ? 'bg-indigo-50 border-indigo-300 text-indigo-950 font-semibold'
                    : 'bg-slate-50/60 border-slate-200/80 text-slate-500'
              }`}
            >
              <span className="flex items-center space-x-2.5">
                <span
                  aria-hidden="true"
                  className={`inline-flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-bold ${
                    isDone
                      ? 'bg-emerald-600 text-white'
                      : isCurrent
                        ? 'bg-indigo-600 text-white animate-pulse'
                        : 'bg-slate-200 text-slate-500'
                  }`}
                >
                  {isDone ? '✓' : isCurrent ? '•' : ''}
                </span>
                <span>{STAGE_LABELS[stage] ?? stage}</span>
              </span>
              <span className="font-mono text-[10px] uppercase tracking-wider">
                {stateLabel}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
};
