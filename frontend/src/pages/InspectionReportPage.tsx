import React, { useCallback, useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  ApiError,
  getInspectionById,
  getInspectionImages,
  getInspectionResult,
  InspectionRunResponse,
} from '../services/api';
import { InspectionSelectionProvider } from '../context/InspectionSelectionContext';
import { ComparisonViewer } from '../components/ComparisonViewer';
import { DefectTable } from '../components/DefectTable';
import { ErrorMessage } from '../components/ErrorMessage';
import { InspectionDetailsCard } from '../components/InspectionDetailsCard';
import { InspectionSummary } from '../components/InspectionSummary';
import { PatternCard } from '../components/PatternCard';
import { PipelineTrace } from '../components/PipelineTrace';
import { ProcessingStatus } from '../components/ProcessingStatus';

export const InspectionReportPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();

  const [result, setResult] = useState<InspectionRunResponse | null>(null);
  const [imageUrls, setImageUrls] = useState<Record<string, string> | undefined>(undefined);
  const [isLoading, setIsLoading] = useState(true);
  const [isPendingRun, setIsPendingRun] = useState(false);
  const [error, setError] = useState<{ message: string; code?: string } | null>(null);
  const [showStepLog, setShowStepLog] = useState(false);

  const numericId = id && /^\d+$/.test(id) ? Number(id) : null;

  const loadReport = useCallback(async () => {
    if (!id) {
      setError({ message: 'Missing inspection ID in URL.', code: 'INVALID_ID' });
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);
    setIsPendingRun(false);

    try {
      if (numericId != null) {
        const [data, imgs] = await Promise.all([
          getInspectionById(numericId),
          getInspectionImages(numericId).catch(() => null),
        ]);
        setResult(data);
        setImageUrls(imgs?.images);
      } else {
        // Legacy UUID URL compatibility
        const data = await getInspectionResult(id);
        setResult(data);
        if (data.inspection_id != null) {
          const imgs = await getInspectionImages(data.inspection_id).catch(() => null);
          setImageUrls(imgs?.images);
        }
      }
    } catch (err: unknown) {
      const apiErr = err as ApiError;
      if (apiErr.status === 409) {
        setIsPendingRun(true);
      } else {
        setError({
          message: apiErr.message || 'Unable to load inspection report.',
          code: apiErr.code,
        });
      }
    } finally {
      setIsLoading(false);
    }
  }, [id, numericId]);

  useEffect(() => {
    void loadReport();
  }, [loadReport]);

  const handleDownloadJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `waferinspect-report-${result.inspection_id ?? result.upload_id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (isLoading) {
    return (
      <div
        role="status"
        aria-live="polite"
        className="max-w-5xl mx-auto p-10 bg-white border border-slate-200 rounded-xl text-center space-y-3"
      >
        <svg
          className="animate-spin h-7 w-7 text-indigo-600 mx-auto"
          fill="none"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
        </svg>
        <p className="text-sm font-semibold text-slate-800">Loading inspection report...</p>
      </div>
    );
  }

  if (isPendingRun && id) {
    return (
      <div className="max-w-3xl mx-auto space-y-6">
        <ProcessingStatus
          inspectionId={numericId ?? undefined}
          uploadId={numericId == null ? id : ''}
          isActive
        />
        <div className="flex items-center justify-between gap-3">
          <button
            type="button"
            onClick={() => void loadReport()}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            Refresh Report
          </button>
          <Link
            to="/inspection/new"
            className="text-xs font-semibold text-slate-600 hover:text-slate-900 underline"
          >
            Run another inspection
          </Link>
        </div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="max-w-3xl mx-auto space-y-4">
        <ErrorMessage
          title="Unable to Load Inspection Report"
          message={error?.message || 'Inspection report not found.'}
          code={error?.code}
        />
        <div className="flex items-center gap-3">
          <Link
            to="/history"
            className="inline-flex items-center px-4 py-2 rounded-lg text-xs font-semibold bg-white border border-slate-300 text-slate-800 hover:bg-slate-100"
          >
            Back to History
          </Link>
          <Link
            to="/inspection/new"
            className="inline-flex items-center px-4 py-2 rounded-lg text-xs font-semibold bg-slate-900 hover:bg-slate-800 text-white"
          >
            Start New Inspection
          </Link>
        </div>
      </div>
    );
  }

  return (
    <InspectionSelectionProvider initialTab="final_overlay">
      <div className="space-y-6 max-w-6xl mx-auto pb-12">
        {/* Top Report Header & Actions */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
              Inspection Report {result.inspection_id != null ? `#${result.inspection_id}` : ''}
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Database-backed report for{' '}
              <span className="font-mono text-slate-700">
                {result.test_filename || result.upload_id}
              </span>
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <Link
              to="/history"
              className="px-3.5 py-2 rounded-lg text-xs font-semibold bg-white hover:bg-slate-100 text-slate-700 border border-slate-300 shadow-xs transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              History
            </Link>
            <button
              type="button"
              onClick={handleDownloadJson}
              className="px-4 py-2 rounded-lg text-xs font-semibold bg-white hover:bg-slate-100 text-slate-800 border border-slate-300 shadow-xs transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 cursor-pointer"
            >
              Download Report (JSON)
            </button>
            <Link
              to="/inspection/new"
              className="px-4 py-2 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white shadow-xs transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Run Another Inspection
            </Link>
          </div>
        </div>

        {/* 1. Inspection Details Card */}
        <InspectionDetailsCard result={result} />

        {/* 2. Inspection Summary Metric Row */}
        <InspectionSummary result={result} />

        {/* 3. Spatial Pattern Card */}
        {result.pattern && <PatternCard pattern={result.pattern} />}

        {/* 4. Interactive ComparisonViewer with SVG DefectOverlay */}
        <ComparisonViewer
          uploadId={result.upload_id}
          imageWidth={result.original_stats.width}
          imageHeight={result.original_stats.height}
          regions={result.regions}
          imageUrls={imageUrls}
        />

        {/* 5. Pipeline Interpretability Trace */}
        <PipelineTrace
          uploadId={result.upload_id}
          stageCounts={result.stage_counts}
          rejectedRegions={result.rejected_regions}
          finalDefectCount={result.regions.length}
          imageUrls={imageUrls}
        />

        {/* 6. Sortable & Filterable Defect Table */}
        <DefectTable
          regions={result.regions}
          inspectionSeverity={result.summary?.severity ?? 'NONE'}
          severityReason={result.summary?.severity_reason}
        />

        {/* Collapsible Preprocessing Step Audit Log */}
        {result.step_log && result.step_log.length > 0 && (
          <section
            aria-label="Preprocessing Execution Log"
            className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-xs"
          >
            <button
              type="button"
              onClick={() => setShowStepLog((prev) => !prev)}
              aria-expanded={showStepLog}
              className="w-full px-4 py-3 bg-slate-50 hover:bg-slate-100/80 transition-colors flex items-center justify-between text-left focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500"
            >
              <span className="text-xs font-bold text-slate-800 uppercase tracking-wide">
                Preprocessing Execution Step Log ({result.step_log.length} steps)
              </span>
              <span className="text-xs font-medium text-slate-500">
                {showStepLog ? 'Hide' : 'Show'}
              </span>
            </button>

            {showStepLog && (
              <div className="overflow-x-auto border-t border-slate-200">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-100/70 border-b border-slate-200 text-slate-600 uppercase text-[10px] font-bold tracking-wider">
                      <th className="py-2 px-4">Step</th>
                      <th className="py-2 px-3">Applied</th>
                      <th className="py-2 px-4">Decision / Reason</th>
                      <th className="py-2 px-3 text-right">Duration</th>
                      <th className="py-2 px-3 text-right">Mean</th>
                      <th className="py-2 px-3 text-right">Std</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {result.step_log.map((item, idx) => (
                      <tr key={idx} className="hover:bg-slate-50/80">
                        <td className="py-2.5 px-4 font-mono font-medium text-slate-800">
                          {item.step}
                        </td>
                        <td className="py-2.5 px-3">
                          {item.applied ? (
                            <span className="inline-flex px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
                              Yes
                            </span>
                          ) : (
                            <span className="inline-flex px-2 py-0.5 rounded text-[10px] font-bold bg-slate-100 text-slate-500 border border-slate-200">
                              Skipped
                            </span>
                          )}
                        </td>
                        <td className="py-2.5 px-4 text-slate-600 max-w-md">{item.reason}</td>
                        <td className="py-2.5 px-3 font-mono text-right text-slate-700">
                          {item.duration_ms} ms
                        </td>
                        <td className="py-2.5 px-3 font-mono text-right text-slate-700">
                          {item.mean}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-right text-slate-700">
                          {item.std}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}
      </div>
    </InspectionSelectionProvider>
  );
};
