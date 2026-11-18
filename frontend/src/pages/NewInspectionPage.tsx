import { Fragment, useState } from 'react';
import { ImageUploader } from '../components/ImageUploader';
import { ErrorMessage } from '../components/ErrorMessage';
import { MetricCard } from '../components/MetricCard';
import { SeverityBadge } from '../components/SeverityBadge';
import {
  uploadInspection,
  runInspection,
  getRunImageUrl,
  RunImageKind,
  UploadInspectionResponse,
  InspectionRunResponse,
} from '../services/api';

const IMAGE_TABS: { key: RunImageKind; label: string }[] = [
  { key: 'final_overlay', label: 'Final Overlay' },
  { key: 'original', label: 'Original' },
  { key: 'preprocessed', label: 'Preprocessed' },
  { key: 'difference', label: 'Difference' },
  { key: 'mask', label: 'Mask' },
  { key: 'filtered_mask', label: 'Filtered Mask' },
  { key: 'bboxes', label: 'Bounding Boxes' },
  { key: 'contours', label: 'Contours' },
];

export interface NewInspectionPageProps {
  initialUploadResult?: UploadInspectionResponse | null;
  initialRunResult?: InspectionRunResponse | null;
}

export const NewInspectionPage = ({
  initialUploadResult = null,
  initialRunResult = null,
}: NewInspectionPageProps = {}) => {
  const [testImage, setTestImage] = useState<File | null>(null);
  const [referenceImage, setReferenceImage] = useState<File | null>(null);
  const [testClientError, setTestClientError] = useState<string | null>(null);
  const [refClientError, setRefClientError] = useState<string | null>(null);

  // Upload state
  const [isUploading, setIsUploading] = useState(false);
  const [backendError, setBackendError] = useState<{ message: string; code?: string } | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadInspectionResponse | null>(initialUploadResult);

  // Pipeline Run state
  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState<{ message: string; code?: string } | null>(null);
  const [runResult, setRunResult] = useState<InspectionRunResponse | null>(initialRunResult);

  // Active view tab for inspection images
  const [activeTab, setActiveTab] = useState<RunImageKind>('final_overlay');
  const [showRejected, setShowRejected] = useState(false);
  const [showSeverityReason, setShowSeverityReason] = useState(false);
  const [expandedRegionIds, setExpandedRegionIds] = useState<Record<number, boolean>>({});

  const toggleRegionWhyFlagged = (id: number) => {
    setExpandedRegionIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!testImage) {
      setTestClientError('Test image is required.');
      return;
    }

    setTestClientError(null);
    setRefClientError(null);
    setBackendError(null);
    setIsUploading(true);

    try {
      const result = await uploadInspection(testImage, referenceImage);
      setUploadResult(result);
      setRunResult(null);
      setRunError(null);
    } catch (err: unknown) {
      const error = err as Error & { code?: string };
      setBackendError({
        message: error.message || 'An unexpected error occurred during upload.',
        code: error.code,
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleRunPipeline = async () => {
    if (!uploadResult) return;

    setIsRunning(true);
    setRunError(null);

    try {
      const result = await runInspection(uploadResult.upload_id);
      setRunResult(result);
      setShowRejected(false);
      setShowSeverityReason(false);
      setExpandedRegionIds({});
      // Default to combined final overlay view when available
      if (result.final_overlay_image_url) {
        setActiveTab('final_overlay');
      } else if (result.bboxes_image_url) {
        setActiveTab('bboxes');
      } else if (result.difference_image_url) {
        setActiveTab('difference');
      } else {
        setActiveTab('preprocessed');
      }
    } catch (err: unknown) {
      const error = err as Error & { code?: string };
      setRunError({
        message: error.message || 'An unexpected error occurred while executing the inspection pipeline.',
        code: error.code,
      });
    } finally {
      setIsRunning(false);
    }
  };

  const handleReset = () => {
    setTestImage(null);
    setReferenceImage(null);
    setTestClientError(null);
    setRefClientError(null);
    setBackendError(null);
    setUploadResult(null);
    setRunResult(null);
    setRunError(null);
    setShowRejected(false);
    setShowSeverityReason(false);
    setExpandedRegionIds({});
    setActiveTab('final_overlay');
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-12">
      {/* Header section */}
      <div>
        <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
          Wafer Inspection Workflow
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          Upload test and golden reference wafer imagery, execute CV preprocessing, and perform classical anomaly detection.
        </p>
      </div>

      {/* Global backend/pipeline error alerts */}
      {backendError && (
        <ErrorMessage
          message={backendError.message}
          code={backendError.code}
          onDismiss={() => setBackendError(null)}
        />
      )}

      {runError && (
        <ErrorMessage
          message={runError.message}
          code={runError.code}
          onDismiss={() => setRunError(null)}
        />
      )}

      {/* Reference dimension warning banner */}
      {uploadResult?.reference_warning && (
        <div
          role="alert"
          className="p-4 rounded-xl border border-amber-200 bg-amber-50 text-amber-900 flex items-start space-x-3 text-sm"
        >
          <span className="font-bold text-amber-700">Notice:</span>
          <span>{uploadResult.reference_warning}</span>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* 1. Upload Section (shown when no successful upload yet) */}
      {/* ------------------------------------------------------------- */}
      {!uploadResult ? (
        <section
          aria-labelledby="upload-section-heading"
          className="bg-white border border-slate-200 rounded-xl p-6 sm:p-8 shadow-sm"
        >
          <h3 id="upload-section-heading" className="text-base font-semibold text-slate-900 mb-6">
            Step 1: Upload Inspection Imagery
          </h3>

          <form onSubmit={handleUpload} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Test Image Uploader */}
              <ImageUploader
                id="test-image-uploader"
                title="Test Wafer Image"
                file={testImage}
                onFileSelect={(f) => {
                  setTestImage(f);
                  setTestClientError(null);
                }}
                onFileRemove={() => setTestImage(null)}
                required
                clientError={testClientError}
                setClientError={setTestClientError}
                helperText="Raw sensor image of the wafer under inspection"
              />

              {/* Reference Image Uploader (Optional) */}
              <ImageUploader
                id="reference-image-uploader"
                title="Golden Reference Image"
                file={referenceImage}
                onFileSelect={(f) => {
                  setReferenceImage(f);
                  setRefClientError(null);
                }}
                onFileRemove={() => setReferenceImage(null)}
                clientError={refClientError}
                setClientError={setRefClientError}
                helperText="Known-good defect-free golden wafer image for differential comparison"
              />
            </div>

            <div className="flex justify-end pt-2">
              <button
                type="submit"
                disabled={isUploading || !testImage}
                className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-slate-900 shadow-sm ${
                  isUploading || !testImage
                    ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                    : 'bg-slate-900 hover:bg-slate-800 text-white cursor-pointer'
                }`}
              >
                {isUploading ? (
                  <span className="flex items-center space-x-2">
                    <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                    </svg>
                    <span>Uploading &amp; Verifying...</span>
                  </span>
                ) : (
                  'Upload & Verify Image'
                )}
              </button>
            </div>
          </form>
        </section>
      ) : (
        /* ------------------------------------------------------------- */
        /* 2. Uploaded State: Ready to Run Pipeline */
        /* ------------------------------------------------------------- */
        <section
          aria-labelledby="confirmed-inspection-heading"
          className="bg-white border border-slate-200 rounded-xl p-6 sm:p-8 shadow-sm space-y-6"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
            <div>
              <div className="flex items-center space-x-2">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" aria-hidden="true" />
                <h3 id="confirmed-inspection-heading" className="text-base font-semibold text-slate-900">
                  Image Upload Verified
                </h3>
              </div>
              <p className="text-xs text-slate-500 mt-1">
                Server validated decoding integrity, mode compatibility, and dimension constraints.
              </p>
            </div>

            <button
              type="button"
              onClick={handleReset}
              className="self-start sm:self-auto px-3.5 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400"
            >
              Start Over / New Image
            </button>
          </div>

          {/* Upload ID Pill */}
          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider">
              Upload Session ID
            </span>
            <span className="font-mono text-xs text-slate-800 bg-white px-2.5 py-1 rounded border border-slate-200 select-all">
              {uploadResult.upload_id}
            </span>
          </div>

          {/* Trigger "Run Inspection" CTA Banner */}
          {!runResult && (
            <div className="p-5 bg-indigo-50/70 border border-indigo-200 rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h4 className="text-sm font-bold text-indigo-950">
                  Step 2: Execute Preprocessing &amp; Defect Detection Pipeline
                </h4>
                <p className="text-xs text-indigo-700 mt-0.5">
                  Standardizes imagery, aligns reference via phase correlation, computes differential residuals, and isolates candidate defects.
                </p>
              </div>

              <button
                type="button"
                onClick={handleRunPipeline}
                disabled={isRunning}
                className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 shadow-sm shrink-0 ${
                  isRunning
                    ? 'bg-indigo-400 text-white cursor-wait'
                    : 'bg-indigo-600 hover:bg-indigo-700 text-white cursor-pointer'
                }`}
              >
                {isRunning ? (
                  <span className="flex items-center space-x-2">
                    <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                    </svg>
                    <span>Processing Pipeline...</span>
                  </span>
                ) : (
                  'Run Inspection'
                )}
              </button>
            </div>
          )}

          {/* Active Processing Indicator */}
          {isRunning && (
            <div
              role="status"
              aria-live="polite"
              className="p-6 bg-slate-50 border border-slate-200 rounded-xl flex flex-col items-center justify-center space-y-3 text-center"
            >
              <svg className="animate-spin h-8 w-8 text-indigo-600" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              <div>
                <p className="text-sm font-semibold text-slate-800">
                  Executing Computer Vision Inspection Pipeline
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Filtering noise, registering golden reference coordinates, computing difference residuals, and applying morphology...
                </p>
              </div>
            </div>
          )}

          {/* ------------------------------------------------------------- */}
          {/* 3. Pipeline Results Section: Diagnostics, Tabs, Stage Counts  */}
          {/* ------------------------------------------------------------- */}
          {runResult && (
            <div className="space-y-8 pt-4">
              {/* Summary Metrics Row (Phase 6) */}
              {runResult.summary && (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                    <MetricCard
                      label="Defects Detected"
                      value={runResult.summary.defect_count}
                      subtext={
                        runResult.summary.defect_count === 0
                          ? 'Clean inspection'
                          : `Avg area: ${runResult.summary.average_defect_area_px} px`
                      }
                    />
                    <MetricCard
                      label="Total Defect Area"
                      value={`${runResult.summary.total_defect_area_px} px`}
                      subtext="Original pixel units"
                    />
                    <MetricCard
                      label="Largest Defect"
                      value={`${runResult.summary.largest_defect_area_px} px`}
                      subtext={
                        runResult.summary.defect_count > 0
                          ? 'Single largest region'
                          : 'No regions'
                      }
                    />
                    <MetricCard
                      label="Severity"
                      subtext={
                        <button
                          type="button"
                          onClick={() => setShowSeverityReason((prev) => !prev)}
                          className="text-[11px] font-medium text-indigo-600 hover:text-indigo-800 underline underline-offset-2 cursor-pointer"
                        >
                          {showSeverityReason ? 'Hide reason' : 'Why this severity?'}
                        </button>
                      }
                    >
                      <div className="pt-0.5">
                        <SeverityBadge
                          severity={runResult.summary.severity}
                          reason={runResult.summary.severity_reason}
                          size="md"
                        />
                      </div>
                    </MetricCard>
                    <MetricCard
                      label="Processing Time"
                      value={`${runResult.processing_time_s.toFixed(3)} s`}
                      subtext={`${Math.round(runResult.processing_time_s * 1000)} ms total`}
                    />
                    <MetricCard
                      label="Detection Method"
                      value={
                        <span className="text-sm font-sans font-bold text-slate-900">
                          {runResult.model?.detection_method ?? 'Classical CV Baseline'}
                        </span>
                      }
                      subtext={
                        runResult.model
                          ? `${runResult.model.name} v${runResult.model.version}`
                          : 'WaferInspect-CV v1.0.0'
                      }
                    />
                  </div>

                  {/* Expandable Severity Reason Panel */}
                  {showSeverityReason && (
                    <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700 leading-relaxed">
                      <strong className="text-slate-900">Prototype Severity Heuristic Basis: </strong>
                      {runResult.summary.severity_reason}
                      <span className="block text-[11px] text-slate-500 mt-1">
                        Note: Severity tiers are a documented prototype heuristic based on wafer-area fraction and count escalation, not a fab standard.
                      </span>
                    </div>
                  )}

                  {/* Calibration Status Notice */}
                  <div className="px-3.5 py-2 rounded-lg bg-amber-50/80 border border-amber-200 text-xs text-amber-900 flex flex-wrap items-center justify-between gap-2">
                    <span>
                      <strong>Calibration:</strong>{' '}
                      {runResult.summary.pixel_size_um
                        ? runResult.summary.calibration_status
                        : `${runResult.summary.calibration_status} (uncalibrated, pixel units)`}
                    </span>
                    <span className="text-[11px] font-mono text-amber-800">
                      Original frame: {runResult.original_stats.width} &times; {runResult.original_stats.height} px
                    </span>
                  </div>
                </div>
              )}

              {/* Detection Overview Banner */}
              {runResult.detection && (
                <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center space-x-2">
                      <span
                        className={`inline-flex items-center px-2.5 py-1 rounded-md text-xs font-bold uppercase tracking-wide border ${
                          runResult.detection.path === 'golden'
                            ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
                            : runResult.detection.path === 'self_reference'
                              ? 'bg-indigo-50 text-indigo-800 border-indigo-300'
                              : 'bg-amber-50 text-amber-800 border-amber-300'
                        }`}
                      >
                        {runResult.detection.label}
                      </span>
                      <span className="text-xs font-mono text-slate-600">
                        Method: {runResult.detection.threshold_method.toUpperCase()} (T={runResult.detection.threshold_value})
                      </span>
                    </div>

                    <div className="text-xs text-slate-500">
                      Wafer Center: ({runResult.detection.wafer_center[0]}, {runResult.detection.wafer_center[1]}) | Radius: {runResult.detection.wafer_radius} px
                    </div>
                  </div>

                  <p className="text-xs text-slate-600">
                    <strong className="text-slate-800">Threshold Diagnostic:</strong> {runResult.detection.threshold_reason}
                  </p>

                  {runResult.detection.path === 'self_reference' && runResult.detection.die_pitch && (
                    <div className="flex flex-wrap gap-4 text-xs font-mono text-indigo-900 bg-indigo-50/60 border border-indigo-100 rounded-lg px-3 py-2">
                      <span>Die Pitch: ({runResult.detection.die_pitch[0]}, {runResult.detection.die_pitch[1]}) px</span>
                      {runResult.detection.peak_strength != null && (
                        <span>Peak Strength: {runResult.detection.peak_strength}</span>
                      )}
                      {runResult.detection.coverage != null && (
                        <span>Coverage: {(runResult.detection.coverage * 100).toFixed(1)}%</span>
                      )}
                      {runResult.detection.unverifiable_fraction != null && (
                        <span>Unverifiable Border: {(runResult.detection.unverifiable_fraction * 100).toFixed(1)}%</span>
                      )}
                    </div>
                  )}

                  {runResult.detection.path === 'golden' && runResult.detection.alignment_info?.applied && (
                    <p className="text-xs text-slate-600 font-mono">
                      Phase Correlation Alignment: shift=({runResult.detection.alignment_info.shift_x}, {runResult.detection.alignment_info.shift_y}) px, response={runResult.detection.alignment_info.response}
                    </p>
                  )}

                  {runResult.detection.warnings && runResult.detection.warnings.length > 0 && (
                    <div className="space-y-1">
                      {runResult.detection.warnings.map((w, idx) => (
                        <p key={idx} className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded px-2.5 py-1.5">
                          <strong>Warning:</strong> {w}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Stage Count Funnel Table */}
              {((runResult.stage_counts && runResult.stage_counts.length > 0) ||
                (runResult.detection && runResult.detection.stage_counts.length > 0)) && (
                <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-xs">
                  <div className="p-3.5 bg-slate-50 border-b border-slate-200">
                    <h5 className="text-xs font-bold text-slate-800 uppercase tracking-wide">
                      Candidate Filtering Funnel (Raw to Final)
                    </h5>
                  </div>
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="bg-slate-100/70 border-b border-slate-200 text-slate-600 uppercase text-[10px] font-bold tracking-wider">
                        <th className="py-2 px-4">Inspection Stage</th>
                        <th className="py-2 px-4 text-right">Candidate Components</th>
                        <th className="py-2 px-4">Note</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {(runResult.stage_counts?.length
                        ? runResult.stage_counts
                        : runResult.detection?.stage_counts || []
                      ).map((sc, i) => (
                        <tr key={i} className="hover:bg-slate-50/80">
                          <td className="py-2.5 px-4 font-mono font-medium text-slate-800">
                            {sc.stage}
                          </td>
                          <td className="py-2.5 px-4 text-right font-mono font-semibold text-indigo-700">
                            {sc.candidate_count}
                          </td>
                          <td className="py-2.5 px-4 text-xs text-slate-500">
                            {sc.note ? (
                              <span className="text-amber-700 font-medium">{sc.note}</span>
                            ) : (
                              '—'
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Inspection Imagery Tabs */}
              <div className="space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-200 pb-3">
                  <h4 className="text-sm font-bold text-slate-900 uppercase tracking-wide">
                    Inspection Imagery Views
                  </h4>

                  {/* Tabs navigation */}
                  <div className="flex flex-wrap gap-1 bg-slate-100 p-1 rounded-lg">
                    {IMAGE_TABS.map((tab) => (
                      <button
                        key={tab.key}
                        type="button"
                        onClick={() => setActiveTab(tab.key)}
                        className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                          activeTab === tab.key
                            ? 'bg-white text-slate-900 shadow-xs'
                            : 'text-slate-500 hover:text-slate-900'
                        }`}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Active Tab Image Display */}
                <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-xs">
                  <div className="p-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-800 uppercase tracking-wide">
                      Viewing: {IMAGE_TABS.find((t) => t.key === activeTab)?.label ?? activeTab}
                    </span>
                    <span className="text-[11px] font-mono text-slate-600 bg-white px-2 py-0.5 rounded border border-slate-200">
                      {runResult.preprocessed_stats.width} &times; {runResult.preprocessed_stats.height} px
                    </span>
                  </div>

                  <div className="h-96 bg-slate-900/5 flex items-center justify-center p-4">
                    <img
                      src={getRunImageUrl(runResult.upload_id, activeTab)}
                      alt={`Inspection ${activeTab} view`}
                      className="max-h-full max-w-full object-contain rounded shadow-xs"
                    />
                  </div>

                  {/* Visible Severity Overlay Legend (colour + line style/thickness for colour-vision accessibility) */}
                  <div className="px-4 py-2.5 bg-slate-50 border-t border-slate-200 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-slate-700">
                    <span className="font-bold uppercase tracking-wider text-[10px] text-slate-500">
                      Severity Overlay Legend:
                    </span>
                    <span className="inline-flex items-center gap-2">
                      <span className="inline-block w-5 h-0 border-t-2 border-dashed border-blue-600" aria-hidden="true" />
                      <span className="font-mono font-semibold text-blue-800">[L] LOW</span>
                      <span className="text-slate-500">(thin dashed blue)</span>
                    </span>
                    <span className="inline-flex items-center gap-2">
                      <span className="inline-block w-5 h-0 border-t-2 border-solid border-orange-600" aria-hidden="true" />
                      <span className="font-mono font-semibold text-orange-800">[M] MEDIUM</span>
                      <span className="text-slate-500">(medium solid orange)</span>
                    </span>
                    <span className="inline-flex items-center gap-2">
                      <span className="inline-block w-5 h-1 border-y-2 border-red-600" aria-hidden="true" />
                      <span className="font-mono font-semibold text-red-800">[H] HIGH</span>
                      <span className="text-slate-500">(thick double red)</span>
                    </span>
                  </div>
                </div>
              </div>

              {/* Accepted Regions Measurement Table (or Clear Zero-Defects State with Severity NONE) */}
              <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-xs">
                <div className="p-3.5 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
                  <h5 className="text-xs font-bold text-slate-800 uppercase tracking-wide">
                    Defect Measurements ({runResult.regions?.length ?? 0})
                  </h5>
                  <span className="text-[11px] text-slate-500">
                    All geometric values reported in original image pixels
                  </span>
                </div>

                {!runResult.regions || runResult.regions.length === 0 ? (
                  <div className="p-8 text-center space-y-2">
                    <div className="flex justify-center">
                      <SeverityBadge
                        severity={runResult.summary?.severity ?? 'NONE'}
                        reason={runResult.summary?.severity_reason}
                        size="md"
                      />
                    </div>
                    <p className="text-sm font-semibold text-emerald-800">
                      No defects found
                    </p>
                    <p className="text-xs text-slate-500 max-w-md mx-auto">
                      {runResult.summary?.severity_reason ??
                        'All candidate regions were either below the noise ceiling or filtered out as sensor noise.'}
                    </p>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="bg-slate-100/70 border-b border-slate-200 text-slate-600 uppercase text-[10px] font-bold tracking-wider">
                          <th className="py-2.5 px-3">ID</th>
                          <th className="py-2.5 px-3">Severity</th>
                          <th className="py-2.5 px-3 text-right">Area (px)</th>
                          <th className="py-2.5 px-2.5 text-right">Width</th>
                          <th className="py-2.5 px-2.5 text-right">Height</th>
                          <th className="py-2.5 px-3 text-right">Length</th>
                          <th className="py-2.5 px-3">Centroid (x, y)</th>
                          <th className="py-2.5 px-3 text-right">Perimeter</th>
                          <th className="py-2.5 px-3 text-right">CNR</th>
                          <th className="py-2.5 px-3">Why Flagged</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {runResult.regions.map((region) => {
                          const isExpanded = Boolean(expandedRegionIds[region.id]);
                          return (
                            <Fragment key={region.id}>
                              <tr className="hover:bg-slate-50/80">
                                <td className="py-2.5 px-3 font-mono font-semibold text-slate-900">
                                  #{region.id}
                                </td>
                                <td className="py-2.5 px-3">
                                  <SeverityBadge severity={region.severity ?? 'LOW'} />
                                </td>
                                <td className="py-2.5 px-3 font-mono text-right text-slate-800">
                                  {region.area_px}
                                </td>
                                <td className="py-2.5 px-2.5 font-mono text-right text-slate-700">
                                  {region.width_px ?? region.bbox[2]}
                                </td>
                                <td className="py-2.5 px-2.5 font-mono text-right text-slate-700">
                                  {region.height_px ?? region.bbox[3]}
                                </td>
                                <td className="py-2.5 px-3 font-mono text-right text-slate-700">
                                  {region.length_px}
                                </td>
                                <td className="py-2.5 px-3 font-mono text-slate-700">
                                  ({region.centroid[0]}, {region.centroid[1]})
                                </td>
                                <td className="py-2.5 px-3 font-mono text-right text-slate-700">
                                  {region.perimeter_px}
                                </td>
                                <td className="py-2.5 px-3 font-mono text-right text-slate-800">
                                  {region.contrast_to_noise}x
                                </td>
                                <td className="py-2.5 px-3">
                                  <button
                                    type="button"
                                    onClick={() => toggleRegionWhyFlagged(region.id)}
                                    aria-expanded={isExpanded}
                                    className="text-xs font-medium text-indigo-600 hover:text-indigo-800 underline underline-offset-2 cursor-pointer"
                                  >
                                    {isExpanded ? 'Hide' : 'Explain'}
                                  </button>
                                </td>
                              </tr>
                              {isExpanded && (
                                <tr className="bg-slate-50/90">
                                  <td colSpan={10} className="px-4 py-3 text-xs text-slate-700 space-y-1.5">
                                    <p className="font-medium text-slate-900">
                                      {region.why_flagged}
                                    </p>
                                    <div className="flex flex-wrap gap-x-5 gap-y-1 text-[11px] font-mono text-slate-600">
                                      <span>Thickness: {region.thickness_px} px</span>
                                      <span>Angle: {region.angle_deg}&deg;</span>
                                      <span>Elongation: {region.elongation}</span>
                                      <span>Eq. Diam: {region.equivalent_diameter_px} px</span>
                                      <span>Polarity: {region.polarity}</span>
                                      <span>Peak Contrast: {region.peak_contrast}</span>
                                      <span>Edge: {region.touches_edge ? 'Rim' : 'Interior'}</span>
                                      {region.area_um2 != null && (
                                        <span>Calibrated Area: {region.area_um2} &mu;m&sup2;</span>
                                      )}
                                    </div>
                                  </td>
                                </tr>
                              )}
                            </Fragment>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              {/* Collapsible Rejected Regions List */}
              <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-xs">
                <button
                  type="button"
                  onClick={() => setShowRejected((prev) => !prev)}
                  aria-expanded={showRejected}
                  className="w-full p-3.5 bg-slate-50 hover:bg-slate-100/80 transition-colors flex items-center justify-between text-left"
                >
                  <span className="text-xs font-bold text-slate-700 uppercase tracking-wide">
                    Rejected Candidate Regions ({runResult.rejected_regions?.length ?? 0})
                  </span>
                  <span className="text-xs font-medium text-slate-500">
                    {showRejected ? 'Hide details' : 'Show details'}
                  </span>
                </button>

                {showRejected && (
                  <div className="border-t border-slate-200">
                    {!runResult.rejected_regions || runResult.rejected_regions.length === 0 ? (
                      <p className="p-4 text-xs text-slate-500">
                        No candidate regions were rejected during noise filtering.
                      </p>
                    ) : (
                      <ul className="divide-y divide-slate-100 text-xs">
                        {runResult.rejected_regions.map((rr) => (
                          <li key={rr.id} className="p-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                            <div className="flex items-center space-x-3">
                              <span className="font-mono font-semibold text-slate-700">#{rr.id}</span>
                              <span className="font-mono text-slate-500">{rr.area_px} px</span>
                            </div>
                            <span className="text-slate-600">{rr.reason}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </div>

              {/* Step Execution Log Table */}
              <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-xs">
                <div className="p-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                  <div>
                    <h5 className="text-sm font-bold text-slate-900">
                      Pipeline Execution Step Log
                    </h5>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Deterministic audit log recorded during execution for reproducibility.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleRunPipeline}
                    disabled={isRunning}
                    className="px-3 py-1 text-xs font-medium text-indigo-600 hover:text-indigo-800 bg-white border border-slate-200 rounded shadow-xs"
                  >
                    Re-run Pipeline
                  </button>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="bg-slate-100/70 border-b border-slate-200 text-slate-600 uppercase text-[10px] font-bold tracking-wider">
                        <th className="py-2.5 px-4">Step</th>
                        <th className="py-2.5 px-3">Applied</th>
                        <th className="py-2.5 px-4">Decision / Reason</th>
                        <th className="py-2.5 px-3 text-right">Duration</th>
                        <th className="py-2.5 px-3 text-right">Mean</th>
                        <th className="py-2.5 px-3 text-right">Std</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 font-sans">
                      {runResult.step_log.map((item, index) => (
                        <tr key={index} className="hover:bg-slate-50/80 transition-colors">
                          <td className="py-3 px-4 font-mono font-medium text-slate-800">
                            {item.step}
                          </td>
                          <td className="py-3 px-3">
                            {item.applied ? (
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
                                Yes
                              </span>
                            ) : (
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-slate-100 text-slate-500 border border-slate-200">
                                Skipped
                              </span>
                            )}
                          </td>
                          <td className="py-3 px-4 text-slate-600 text-xs leading-relaxed max-w-md">
                            {item.reason}
                          </td>
                          <td className="py-3 px-3 font-mono text-right text-slate-700">
                            {item.duration_ms} ms
                          </td>
                          <td className="py-3 px-3 font-mono text-right text-slate-700">
                            {item.mean}
                          </td>
                          <td className="py-3 px-3 font-mono text-right text-slate-700">
                            {item.std}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
};
