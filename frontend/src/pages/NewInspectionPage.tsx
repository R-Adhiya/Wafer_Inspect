import { useState } from 'react';
import { ImageUploader } from '../components/ImageUploader';
import { ErrorMessage } from '../components/ErrorMessage';
import {
  uploadInspection,
  runInspection,
  getRunImageUrl,
  RunImageKind,
  UploadInspectionResponse,
  InspectionRunResponse,
} from '../services/api';

const IMAGE_TABS: { key: RunImageKind; label: string }[] = [
  { key: 'original', label: 'Original' },
  { key: 'preprocessed', label: 'Preprocessed' },
  { key: 'difference', label: 'Difference' },
  { key: 'mask', label: 'Mask' },
  { key: 'filtered_mask', label: 'Filtered Mask' },
  { key: 'bboxes', label: 'Bounding Boxes' },
  { key: 'contours', label: 'Contours' },
];

export const NewInspectionPage = () => {
  const [testImage, setTestImage] = useState<File | null>(null);
  const [referenceImage, setReferenceImage] = useState<File | null>(null);
  const [testClientError, setTestClientError] = useState<string | null>(null);
  const [refClientError, setRefClientError] = useState<string | null>(null);

  // Upload state
  const [isUploading, setIsUploading] = useState(false);
  const [backendError, setBackendError] = useState<{ message: string; code?: string } | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadInspectionResponse | null>(null);

  // Pipeline Run state
  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState<{ message: string; code?: string } | null>(null);
  const [runResult, setRunResult] = useState<InspectionRunResponse | null>(null);

  // Active view tab for inspection images
  const [activeTab, setActiveTab] = useState<RunImageKind>('bboxes');
  const [showRejected, setShowRejected] = useState(false);

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
      // Default to bounding boxes view when available
      if (result.bboxes_image_url) {
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
    setActiveTab('bboxes');
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
                  'Upload &amp; Verify Image'
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
              {/* Detection Overview Banner */}
              {runResult.detection && (
                <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center space-x-2">
                      <span
                        className={`inline-flex items-center px-2.5 py-1 rounded-md text-xs font-bold uppercase tracking-wide border ${
                          runResult.detection.path === 'golden'
                            ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
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

                  {runResult.detection.alignment_info?.applied && (
                    <p className="text-xs text-slate-600 font-mono">
                      Phase Correlation Alignment: shift=({runResult.detection.alignment_info.shift_x}, {runResult.detection.alignment_info.shift_y}) px, response={runResult.detection.alignment_info.response}
                    </p>
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
                </div>
              </div>

              {/* Accepted Regions Table (or Clear Zero-Defects State) */}
              <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-xs">
                <div className="p-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                  <h5 className="text-xs font-bold text-slate-800 uppercase tracking-wide">
                    Accepted Defect Regions ({runResult.regions?.length ?? 0})
                  </h5>
                </div>

                {!runResult.regions || runResult.regions.length === 0 ? (
                  <div className="p-8 text-center space-y-1.5">
                    <p className="text-sm font-semibold text-emerald-800">
                      No defects found
                    </p>
                    <p className="text-xs text-slate-500">
                      All candidate regions were either below the noise ceiling or filtered out as sensor noise.
                    </p>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="bg-slate-100/70 border-b border-slate-200 text-slate-600 uppercase text-[10px] font-bold tracking-wider">
                          <th className="py-2.5 px-4">ID</th>
                          <th className="py-2.5 px-3 text-right">Area (px)</th>
                          <th className="py-2.5 px-4">BBox (x, y, w, h)</th>
                          <th className="py-2.5 px-4">Centroid (x, y)</th>
                          <th className="py-2.5 px-3">Polarity</th>
                          <th className="py-2.5 px-3 text-right">Mean Contrast</th>
                          <th className="py-2.5 px-3">Edge</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {runResult.regions.map((region) => (
                          <tr key={region.id} className="hover:bg-slate-50/80">
                            <td className="py-2.5 px-4 font-mono font-semibold text-slate-900">
                              #{region.id}
                            </td>
                            <td className="py-2.5 px-3 font-mono text-right text-slate-800">
                              {region.area_px}
                            </td>
                            <td className="py-2.5 px-4 font-mono text-slate-700">
                              ({region.bbox[0]}, {region.bbox[1]}, {region.bbox[2]}, {region.bbox[3]})
                            </td>
                            <td className="py-2.5 px-4 font-mono text-slate-700">
                              ({region.centroid[0]}, {region.centroid[1]})
                            </td>
                            <td className="py-2.5 px-3">
                              <span
                                className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${
                                  region.polarity === 'bright'
                                    ? 'bg-sky-50 text-sky-800 border-sky-200'
                                    : 'bg-slate-100 text-slate-700 border-slate-300'
                                }`}
                              >
                                {region.polarity}
                              </span>
                            </td>
                            <td className="py-2.5 px-3 font-mono text-right text-slate-700">
                              {region.mean_contrast}
                            </td>
                            <td className="py-2.5 px-3 text-xs text-slate-600">
                              {region.touches_edge ? (
                                <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-50 text-amber-800 border border-amber-200">
                                  Rim
                                </span>
                              ) : (
                                <span className="text-slate-400">Interior</span>
                              )}
                            </td>
                          </tr>
                        ))}
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
