import { useState } from 'react';
import { ImageUploader } from '../components/ImageUploader';
import { ErrorMessage } from '../components/ErrorMessage';
import {
  uploadInspection,
  runInspection,
  getRunImageUrl,
  UploadInspectionResponse,
  InspectionRunResponse,
} from '../services/api';

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

  const handleUpload = async () => {
    if (!testImage) return;

    setIsUploading(true);
    setBackendError(null);
    setRunError(null);
    setRunResult(null);

    try {
      const response = await uploadInspection(testImage, referenceImage);
      setUploadResult(response);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setBackendError({
          message: err.message,
          code: (err as unknown as { code?: string }).code,
        });
      } else {
        setBackendError({
          message: 'An unknown error occurred while uploading inspection files.',
        });
      }
    } finally {
      setIsUploading(false);
    }
  };

  const handleRunPipeline = async () => {
    if (!uploadResult) return;

    setIsRunning(true);
    setRunError(null);

    try {
      const response = await runInspection(uploadResult.upload_id);
      setRunResult(response);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setRunError({
          message: err.message,
          code: (err as unknown as { code?: string }).code,
        });
      } else {
        setRunError({
          message: 'An unknown error occurred while running the preprocessing pipeline.',
        });
      }
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
    setIsRunning(false);
    setRunError(null);
    setRunResult(null);
  };

  return (
    <div className="space-y-8">
      {/* Intro Heading */}
      <div>
        <h2 className="text-xl font-bold tracking-tight text-slate-900">
          Wafer Defect Inspection &amp; Preprocessing
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          Upload wafer surface imagery and execute the computer vision preprocessing pipeline.
        </p>
      </div>

      {/* Backend Upload Error Alert */}
      {backendError && (
        <ErrorMessage
          message={backendError.message}
          code={backendError.code}
          onDismiss={() => setBackendError(null)}
        />
      )}

      {/* Pipeline Run Error Alert */}
      {runError && (
        <ErrorMessage
          message={runError.message}
          code={runError.code}
          onDismiss={() => setRunError(null)}
        />
      )}

      {/* ------------------------------------------------------------- */}
      {/* 1. Upload Form Section (shown before upload) */}
      {/* ------------------------------------------------------------- */}
      {!uploadResult ? (
        <section
          aria-labelledby="upload-card-heading"
          className="bg-white border border-slate-200 rounded-xl p-6 sm:p-8 shadow-sm space-y-6"
        >
          <h3 id="upload-card-heading" className="text-base font-semibold text-slate-800 border-b border-slate-100 pb-3">
            Step 1: Select Inspection Imagery
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Test Wafer Image (Required) */}
            <ImageUploader
              id="test-wafer-uploader"
              title="Test Wafer Image"
              required={true}
              helperText="Target wafer imagery for surface defect inspection (.png, .jpg, .tif up to 20MB)."
              file={testImage}
              onFileSelect={(f) => {
                setTestImage(f);
                setBackendError(null);
              }}
              onFileRemove={() => {
                setTestImage(null);
                setBackendError(null);
              }}
              clientError={testClientError}
              setClientError={setTestClientError}
            />

            {/* Reference / Golden Image (Optional) */}
            <ImageUploader
              id="reference-wafer-uploader"
              title="Reference / Golden Image (Optional)"
              required={false}
              helperText="A reference image improves comparison-based detection when available."
              file={referenceImage}
              onFileSelect={(f) => {
                setReferenceImage(f);
                setBackendError(null);
              }}
              onFileRemove={() => {
                setReferenceImage(null);
                setBackendError(null);
              }}
              clientError={refClientError}
              setClientError={setRefClientError}
            />
          </div>

          {/* Action Row */}
          <div className="pt-4 border-t border-slate-100 flex items-center justify-between">
            <span className="text-xs text-slate-400">
              * Test wafer image is required to initiate inspection
            </span>

            <button
              type="button"
              onClick={handleUpload}
              disabled={!testImage || isUploading}
              className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 shadow-sm ${
                !testImage || isUploading
                  ? 'bg-slate-100 text-slate-400 border border-slate-200 cursor-not-allowed'
                  : 'bg-indigo-600 hover:bg-indigo-700 text-white cursor-pointer'
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
                  Image Upload Verified (Ready for Preprocessing)
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

          {/* Trigger "Run Inspection" CTA Banner (if not already run) */}
          {!runResult && (
            <div className="p-5 bg-indigo-50/70 border border-indigo-200 rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h4 className="text-sm font-bold text-indigo-950">
                  Step 2: Execute Preprocessing Pipeline
                </h4>
                <p className="text-xs text-indigo-700 mt-0.5">
                  Runs grayscale standardization, aspect-preserving resize, noise reduction, and adaptive CLAHE contrast analysis.
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
                  Executing Computer Vision Preprocessing Pipeline
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Evaluating illumination uniformity, calculating spatial contrast, and generating normalized outputs...
                </p>
              </div>
            </div>
          )}

          {/* ------------------------------------------------------------- */}
          {/* 3. Pipeline Results Section: Side-by-Side Images & Step Log */}
          {/* ------------------------------------------------------------- */}
          {runResult && (
            <div className="space-y-8 pt-4">
              <div className="border-t border-slate-100 pt-6">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-base font-bold text-slate-900">
                    Image Preprocessing Comparison
                  </h4>
                  <span className="text-xs font-mono px-2 py-0.5 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded">
                    Status: Pipeline Complete
                  </span>
                </div>

                {/* Side-by-Side Images */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Original Image */}
                  <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-xs">
                    <div className="p-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-800 uppercase tracking-wide">
                        Original Input
                      </span>
                      <span className="text-[11px] font-mono text-slate-600 bg-white px-2 py-0.5 rounded border border-slate-200">
                        {runResult.original_stats.width} &times; {runResult.original_stats.height} px
                      </span>
                    </div>

                    <div className="h-64 bg-slate-900/5 flex items-center justify-center p-3 border-b border-slate-100">
                      <img
                        src={getRunImageUrl(runResult.upload_id, 'original')}
                        alt="Original wafer test image"
                        className="max-h-full max-w-full object-contain rounded shadow-xs"
                      />
                    </div>

                    <div className="p-3.5 bg-white grid grid-cols-2 gap-3 text-xs">
                      <div>
                        <span className="text-slate-400 block uppercase text-[10px] font-bold">Mean Intensity</span>
                        <span className="font-mono text-slate-800 font-semibold">{runResult.original_stats.mean}</span>
                      </div>
                      <div>
                        <span className="text-slate-400 block uppercase text-[10px] font-bold">Std Deviation</span>
                        <span className="font-mono text-slate-800 font-semibold">{runResult.original_stats.std}</span>
                      </div>
                    </div>
                  </div>

                  {/* Preprocessed Image */}
                  <div className="border border-indigo-200 rounded-xl overflow-hidden bg-white shadow-xs">
                    <div className="p-3 bg-indigo-50/70 border-b border-indigo-200 flex items-center justify-between">
                      <span className="text-xs font-bold text-indigo-950 uppercase tracking-wide">
                        Preprocessed Output (8-bit)
                      </span>
                      <span className="text-[11px] font-mono text-indigo-700 bg-white px-2 py-0.5 rounded border border-indigo-200">
                        {runResult.preprocessed_stats.width} &times; {runResult.preprocessed_stats.height} px
                      </span>
                    </div>

                    <div className="h-64 bg-slate-900/5 flex items-center justify-center p-3 border-b border-slate-100">
                      <img
                        src={getRunImageUrl(runResult.upload_id, 'preprocessed')}
                        alt="Preprocessed wafer inspection image"
                        className="max-h-full max-w-full object-contain rounded shadow-xs"
                      />
                    </div>

                    <div className="p-3.5 bg-white grid grid-cols-2 gap-3 text-xs">
                      <div>
                        <span className="text-slate-400 block uppercase text-[10px] font-bold">Mean Intensity</span>
                        <span className="font-mono text-indigo-900 font-semibold">{runResult.preprocessed_stats.mean}</span>
                      </div>
                      <div>
                        <span className="text-slate-400 block uppercase text-[10px] font-bold">Std Deviation</span>
                        <span className="font-mono text-indigo-900 font-semibold">{runResult.preprocessed_stats.std}</span>
                      </div>
                    </div>
                  </div>
                </div>
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
