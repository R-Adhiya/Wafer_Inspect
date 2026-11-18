import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ImageUploader } from '../components/ImageUploader';
import { ErrorMessage } from '../components/ErrorMessage';
import { ProcessingStatus } from '../components/ProcessingStatus';
import {
  uploadInspection,
  runInspection,
  UploadInspectionResponse,
} from '../services/api';

export const NewInspectionPage: React.FC = () => {
  const navigate = useNavigate();

  const [testImage, setTestImage] = useState<File | null>(null);
  const [referenceImage, setReferenceImage] = useState<File | null>(null);
  const [testClientError, setTestClientError] = useState<string | null>(null);
  const [refClientError, setRefClientError] = useState<string | null>(null);

  const [isUploading, setIsUploading] = useState(false);
  const [backendError, setBackendError] = useState<{ message: string; code?: string } | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadInspectionResponse | null>(null);

  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState<{ message: string; code?: string } | null>(null);

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
      await runInspection(uploadResult.upload_id);
      navigate(`/inspection/${encodeURIComponent(uploadResult.upload_id)}`);
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

  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-12">
      <div>
        <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
          New Wafer Inspection
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          Upload a test wafer image (and an optional golden reference), verify decoding integrity,
          and execute the full classical CV inspection pipeline.
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
                  Image Upload Verified
                </h3>
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
              Upload Session ID
            </span>
            <span className="font-mono text-xs text-slate-800 bg-white px-2.5 py-1 rounded border border-slate-200 select-all">
              {uploadResult.upload_id}
            </span>
          </div>

          {!isRunning ? (
            <div className="p-5 bg-indigo-50/70 border border-indigo-200 rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h4 className="text-sm font-bold text-indigo-950">
                  Step 2: Run Inspection &amp; Generate Report
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
            <ProcessingStatus uploadId={uploadResult.upload_id} isActive={isRunning} />
          )}
        </section>
      )}
    </div>
  );
};
