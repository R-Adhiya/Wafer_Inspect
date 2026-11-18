import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ImageUploader } from '../components/ImageUploader';
import { ErrorMessage } from '../components/ErrorMessage';
import { ProcessingStatus } from '../components/ProcessingStatus';
import {
  createSampleUpload,
  getSamples,
  runInspection,
  SampleItem,
  uploadInspection,
  UploadInspectionResponse,
} from '../services/api';

export const NewInspectionPage: React.FC = () => {
  const navigate = useNavigate();

  const [testImage, setTestImage] = useState<File | null>(null);
  const [referenceImage, setReferenceImage] = useState<File | null>(null);
  const [testClientError, setTestClientError] = useState<string | null>(null);
  const [refClientError, setRefClientError] = useState<string | null>(null);

  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [selectedSample, setSelectedSample] = useState<string>('');
  const [isLoadingSample, setIsLoadingSample] = useState(false);

  const [isUploading, setIsUploading] = useState(false);
  const [backendError, setBackendError] = useState<{ message: string; code?: string } | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadInspectionResponse | null>(null);

  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState<{ message: string; code?: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    getSamples()
      .then((res) => {
        if (!cancelled) {
          setSamples(res.samples);
          if (res.samples.length > 0) {
            setSelectedSample(res.samples[0].name);
          }
        }
      })
      .catch(() => {
        // Non-blocking: manual upload still works even if sample catalog is unavailable
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleLoadSample = async () => {
    if (!selectedSample) return;
    setBackendError(null);
    setRunError(null);
    setIsLoadingSample(true);
    try {
      const result = await createSampleUpload(selectedSample);
      setUploadResult(result);
    } catch (err: unknown) {
      const error = err as Error & { code?: string };
      setBackendError({
        message: error.message || 'Unable to generate synthetic demonstration sample.',
        code: error.code,
      });
    } finally {
      setIsLoadingSample(false);
    }
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
    setRunError(null);
    setIsUploading(true);

    try {
      const result = await uploadInspection(testImage, referenceImage);
      setUploadResult(result);
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
      const res = await runInspection(uploadResult.inspection_id);
      const targetId = res.inspection_id ?? uploadResult.inspection_id;
      navigate(`/inspection/${targetId}`);
    } catch (err: unknown) {
      const error = err as Error & { code?: string };
      setRunError({
        message:
          error.message ||
          'An unexpected error occurred while executing the inspection pipeline.',
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
    setRunError(null);
  };

  const activeSampleMeta = samples.find((s) => s.name === selectedSample);

  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-12">
      <div>
        <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
          New Wafer Inspection
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          Upload a test wafer image (and an optional golden reference), or try a deterministic
          synthetic demonstration wafer.
        </p>
      </div>

      {backendError && (
        <ErrorMessage
          title="Upload Rejected"
          message={backendError.message}
          code={backendError.code}
          onDismiss={() => setBackendError(null)}
        />
      )}

      {runError && (
        <ErrorMessage
          title="Inspection Pipeline Failed"
          message={runError.message}
          code={runError.code}
          onDismiss={() => setRunError(null)}
        />
      )}

      {uploadResult?.reference_warning && (
        <div
          role="alert"
          className="p-4 rounded-xl border border-amber-200 bg-amber-50 text-amber-900 flex items-start space-x-3 text-sm"
        >
          <span className="font-bold text-amber-700">Notice:</span>
          <span>{uploadResult.reference_warning}</span>
        </div>
      )}

      {!uploadResult ? (
        <div className="space-y-6">
          {/* Synthetic Demonstration Wafer Selector */}
          {samples.length > 0 && (
            <section
              aria-labelledby="demo-sample-heading"
              className="bg-indigo-50/60 border border-indigo-200 rounded-xl p-5 space-y-3"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 id="demo-sample-heading" className="text-sm font-bold text-indigo-950">
                    Try a Synthetic Demo Wafer
                  </h3>
                  <p className="text-xs text-indigo-800 mt-0.5">
                    Synthetic demonstration image — generated deterministically and routed through the exact upload validator.
                  </p>
                </div>
                <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-white text-indigo-700 border border-indigo-200">
                  Synthetic demonstration image
                </span>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center gap-3">
                <label htmlFor="synthetic-sample-select" className="sr-only">
                  Select a synthetic demonstration wafer
                </label>
                <select
                  id="synthetic-sample-select"
                  value={selectedSample}
                  onChange={(e) => setSelectedSample(e.target.value)}
                  className="flex-1 px-3 py-2 text-xs bg-white border border-indigo-300 rounded-lg text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  {samples.map((s) => (
                    <option key={s.name} value={s.name}>
                      {s.title} ({s.image_size}×{s.image_size}{s.has_reference ? ', golden pair' : ''})
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  disabled={isLoadingSample || !selectedSample}
                  onClick={() => void handleLoadSample()}
                  className="px-4 py-2 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white shadow-xs shrink-0 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 cursor-pointer"
                >
                  {isLoadingSample ? 'Generating Sample...' : 'Load Synthetic Demo'}
                </button>
              </div>
              {activeSampleMeta && (
                <p className="text-xs text-indigo-900/80">{activeSampleMeta.description}</p>
              )}
            </section>
          )}

          {/* Manual File Upload Section */}
          <section
            aria-labelledby="upload-section-heading"
            className="bg-white border border-slate-200 rounded-xl p-6 sm:p-8 shadow-sm"
          >
            <h3 id="upload-section-heading" className="text-base font-semibold text-slate-900 mb-6">
              Step 1: Upload Inspection Imagery
            </h3>

            <form onSubmit={handleUpload} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
                  helperText="Optional known-good golden wafer image for differential comparison"
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
                  {isUploading ? 'Uploading & Verifying...' : 'Upload & Verify Image'}
                </button>
              </div>
            </form>
          </section>
        </div>
      ) : (
        <section
          aria-labelledby="confirmed-inspection-heading"
          className="bg-white border border-slate-200 rounded-xl p-6 sm:p-8 shadow-sm space-y-6"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
            <div>
              <div className="flex items-center space-x-2">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" aria-hidden="true" />
                <h3 id="confirmed-inspection-heading" className="text-base font-semibold text-slate-900">
                  Image Upload Verified (Inspection #{uploadResult.inspection_id})
                </h3>
                {uploadResult.disclaimer && (
                  <span className="px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200 text-[10px] font-bold">
                    {uploadResult.disclaimer}
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-500 mt-1">
                {uploadResult.test_image.original_filename} ({uploadResult.test_image.width} &times;{' '}
                {uploadResult.test_image.height} px, {uploadResult.test_image.mode})
                {uploadResult.reference_image
                  ? ` + Golden Reference (${uploadResult.reference_image.original_filename})`
                  : ' — Single-wafer mode (die-to-die self-reference)'}
              </p>
            </div>

            <button
              type="button"
              onClick={handleReset}
              disabled={isRunning}
              className="self-start sm:self-auto px-3.5 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 disabled:opacity-50 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400"
            >
              Start Over / New Image
            </button>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider">
              Inspection Record
            </span>
            <span className="font-mono text-xs text-slate-800 bg-white px-2.5 py-1 rounded border border-slate-200 select-all">
              #{uploadResult.inspection_id} ({uploadResult.upload_id})
            </span>
          </div>

          {!isRunning ? (
            <div className="p-5 bg-indigo-50/70 border border-indigo-200 rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h4 className="text-sm font-bold text-indigo-950">
                  Step 2: Run Inspection &amp; Persist Report
                </h4>
                <p className="text-xs text-indigo-700 mt-0.5">
                  Executes preprocessing, defect detection, noise filtering, half-maximum measurement
                  refinement, severity classification, and spatial pattern diagnosis.
                </p>
              </div>

              <button
                type="button"
                onClick={handleRunPipeline}
                className="px-5 py-2.5 rounded-lg text-sm font-semibold bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm shrink-0 transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 cursor-pointer"
              >
                Run Inspection
              </button>
            </div>
          ) : (
            <ProcessingStatus
              inspectionId={uploadResult.inspection_id}
              uploadId={uploadResult.upload_id}
              isActive={isRunning}
            />
          )}
        </section>
      )}
    </div>
  );
};
