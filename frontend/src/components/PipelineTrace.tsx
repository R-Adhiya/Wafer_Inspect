import React, { useState } from 'react';
import {
  getRunImageUrl,
  RejectedRegion,
  RunImageKind,
  StageCount,
} from '../services/api';
import { useInspectionSelection } from '../context/InspectionSelectionContext';

export interface PipelineTraceProps {
  uploadId: string;
  stageCounts: StageCount[];
  rejectedRegions: RejectedRegion[];
  finalDefectCount: number;
}

interface TraceStep {
  id: string;
  label: string;
  imageKind: RunImageKind;
  countLabel: string;
}

function findStageCount(stageCounts: StageCount[], ...names: string[]): number | null {
  for (const name of names) {
    const match = stageCounts.find((s) => s.stage === name);
    if (match) return match.candidate_count;
  }
  return null;
}

export const PipelineTrace: React.FC<PipelineTraceProps> = ({
  uploadId,
  stageCounts,
  rejectedRegions,
  finalDefectCount,
}) => {
  const { activeTab, setActiveTab } = useInspectionSelection();
  const [showRejected, setShowRejected] = useState(true);

  const rawCount = findStageCount(stageCounts, 'raw_threshold');
  const filteredCandidateCount = findStageCount(
    stageCounts,
    'after_fragment_merge',
    'after_morphology'
  );
  const finalCount = findStageCount(stageCounts, 'final') ?? finalDefectCount;

  const steps: TraceStep[] = [
    { id: 'original', label: 'Original', imageKind: 'original', countLabel: 'Input' },
    { id: 'preprocessed', label: 'Preprocessed', imageKind: 'preprocessed', countLabel: 'Denoised' },
    { id: 'difference', label: 'Difference', imageKind: 'difference', countLabel: 'Residual' },
    {
      id: 'mask',
      label: 'Threshold Mask',
      imageKind: 'mask',
      countLabel: rawCount != null ? `${rawCount} raw` : 'Masked',
    },
    {
      id: 'filtered_mask',
      label: 'Filtered Candidates',
      imageKind: 'filtered_mask',
      countLabel:
        filteredCandidateCount != null ? `${filteredCandidateCount} cand.` : 'Filtered',
    },
    {
      id: 'final_overlay',
      label: 'Final Defects',
      imageKind: 'final_overlay',
      countLabel: `${finalCount} final`,
    },
  ];

  const funnelNotes = stageCounts.filter((s) => Boolean(s.note));

  return (
    <section
      aria-labelledby="pipeline-trace-heading"
      className="bg-white border border-slate-200 rounded-xl shadow-xs overflow-hidden"
    >
      <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3
            id="pipeline-trace-heading"
            className="text-xs font-bold text-slate-900 uppercase tracking-wide"
          >
            Pipeline Interpretability Trace
          </h3>
          <p className="text-[11px] text-slate-500 mt-0.5">
            Click any stage thumbnail to inspect it in the viewer above. Shows candidate reduction from raw threshold to final defects.
          </p>
        </div>
      </div>

      {/* Horizontal Stepper with Thumbnails & Real Counts */}
      <div className="p-4 overflow-x-auto">
        <ol className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 min-w-[540px]">
          {steps.map((step, idx) => {
            const isCurrent = activeTab === step.imageKind;
            return (
              <li key={step.id}>
                <button
                  type="button"
                  onClick={() => setActiveTab(step.imageKind)}
                  aria-current={isCurrent ? 'step' : undefined}
                  className={`w-full text-left rounded-lg border p-2 transition-all focus:outline-none focus:ring-2 focus:ring-indigo-500 cursor-pointer ${
                    isCurrent
                      ? 'border-indigo-500 bg-indigo-50/70 ring-1 ring-indigo-300'
                      : 'border-slate-200 bg-slate-50/50 hover:border-slate-300 hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[10px] font-bold text-slate-500 font-mono">
                      {idx + 1}.
                    </span>
                    <span
                      className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                        isCurrent
                          ? 'bg-indigo-600 text-white'
                          : 'bg-white text-slate-700 border border-slate-200'
                      }`}
                    >
                      {step.countLabel}
                    </span>
                  </div>
                  <div className="aspect-square w-full bg-slate-900/10 rounded overflow-hidden flex items-center justify-center mb-1.5">
                    <img
                      src={getRunImageUrl(uploadId, step.imageKind)}
                      alt={`${step.label} stage thumbnail`}
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <div className="text-[11px] font-semibold text-slate-800 truncate">
                    {step.label}
                  </div>
                </button>
              </li>
            );
          })}
        </ol>
      </div>

      {/* Funnel Counts & Notes */}
      {stageCounts.length > 0 && (
        <div className="px-4 pb-4 space-y-2 border-t border-slate-100 pt-3">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Candidate Funnel:
            </span>
            {stageCounts.map((sc, i) => (
              <span key={sc.stage} className="inline-flex items-center gap-1.5">
                <span className="px-2 py-0.5 rounded bg-slate-100 border border-slate-200 font-mono text-[11px] text-slate-800">
                  {sc.stage}: <strong className="text-indigo-700">{sc.candidate_count}</strong>
                </span>
                {i < stageCounts.length - 1 && (
                  <span className="text-slate-400" aria-hidden="true">
                    &rarr;
                  </span>
                )}
              </span>
            ))}
          </div>

          {funnelNotes.length > 0 && (
            <ul className="space-y-1 text-xs text-slate-600">
              {funnelNotes.map((sc) => (
                <li
                  key={sc.stage}
                  className="px-2.5 py-1.5 rounded bg-amber-50/70 border border-amber-200/80 text-amber-900"
                >
                  <strong className="font-mono">{sc.stage}:</strong> {sc.note}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Rejected Candidate Regions */}
      <div className="border-t border-slate-200">
        <button
          type="button"
          onClick={() => setShowRejected((prev) => !prev)}
          aria-expanded={showRejected}
          className="w-full px-4 py-2.5 bg-slate-50 hover:bg-slate-100/80 transition-colors flex items-center justify-between text-left focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500"
        >
          <span className="text-xs font-bold text-slate-700 uppercase tracking-wide">
            Rejected Candidate Regions ({rejectedRegions.length})
          </span>
          <span className="text-xs font-medium text-slate-500">
            {showRejected ? 'Hide' : 'Show'}
          </span>
        </button>

        {showRejected && (
          <div className="border-t border-slate-100">
            {rejectedRegions.length === 0 ? (
              <p className="px-4 py-3 text-xs text-slate-500">
                No candidate regions were rejected during noise filtering.
              </p>
            ) : (
              <ul className="divide-y divide-slate-100 text-xs">
                {rejectedRegions.map((rr) => (
                  <li
                    key={rr.id}
                    className="px-4 py-2.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2"
                  >
                    <div className="flex items-center space-x-3 font-mono">
                      <span className="font-semibold text-slate-700">#{rr.id}</span>
                      <span className="text-slate-500">{rr.area_px} px</span>
                      <span className="text-slate-500">contrast {rr.mean_contrast}</span>
                    </div>
                    <span className="text-slate-600">{rr.reason}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </section>
  );
};
