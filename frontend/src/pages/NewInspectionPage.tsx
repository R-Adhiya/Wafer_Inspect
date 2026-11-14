import { useState } from 'react';
import { ImageUploader } from '../components/ImageUploader';
import { ErrorMessage } from '../components/ErrorMessage';
import { uploadInspection, getUploadImageUrl, UploadInspectionResponse } from '../services/api';

export const NewInspectionPage = () => {
  const [testImage, setTestImage] = useState<File | null>(null);
  const [referenceImage, setReferenceImage] = useState<File | null>(null);
  const [testClientError, setTestClientError] = useState<string | null>(null);
  const [refClientError, setRefClientError] = useState<string | null>(null);

  const [isUploading, setIsUploading] = useState(false);
  const [backendError, setBackendError] = useState<{ message: string; code?: string } | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadInspectionResponse | null>(null);

  const handleRunInspection = async () => {
    if (!testImage) return;

    setIsUploading(true);
    setBackendError(null);

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

  const handleReset = () => {
    setTestImage(null);
    setReferenceImage(null);
    setTestClientError(null);
    setRefClientError(null);
    setBackendError(null);
    setUploadResult(null);
  };

  return (
    <div className="space-y-8">
      {/* Intro Heading */}
      <div>
        <h2 className="text-xl font-bold tracking-tight text-slate-900">
          New Inspection
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          Select or drop a test wafer image to begin automated defect inspection and analysis.
        </p>
      </div>

      {/* Backend Rejection Error Alert */}
      {backendError && (
        <ErrorMessage
          message={backendError.message}
          code={backendError.code}
          onDismiss={() => setBackendError(null)}
        />
      )}

      {/* Upload Form Card */}
      {!uploadResult ? (
        <section
          aria-labelledby="upload-card-heading"
          className="bg-white border border-slate-200 rounded-xl p-6 sm:p-8 shadow-sm space-y-6"
        >
          <h3 id="upload-card-heading" className="text-base font-semibold text-slate-800 border-b border-slate-100 pb-3">
            Select Inspection Imagery
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* 1. Test Wafer Image (Required) */}
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

            {/* 2. Reference / Golden Image (Optional) */}
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
              onClick={handleRunInspection}
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
                'Run Inspection'
              )}
            </button>
          </div>
        </section>
      ) : (
        /* Server Confirmation / Preview State */
        <section
          aria-labelledby="confirmed-inspection-heading"
          className="bg-white border border-slate-200 rounded-xl p-6 sm:p-8 shadow-sm space-y-6"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
            <div>
              <div className="flex items-center space-x-2">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" aria-hidden="true" />
                <h3 id="confirmed-inspection-heading" className="text-base font-semibold text-slate-900">
                  Image Upload Verified &amp; Stored
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
              Upload Another Wafer
            </button>
          </div>

          {/* Upload ID Pill */}
          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider">
              Upload Identifier
            </span>
            <span className="font-mono text-xs text-slate-800 bg-white px-2.5 py-1 rounded border border-slate-200 select-all">
              {uploadResult.upload_id}
            </span>
          </div>

          {/* Reference Warning if applicable */}
          {uploadResult.reference_warning && (
            <div className="p-4 bg-amber-50/80 border border-amber-200 rounded-lg text-amber-900 text-xs flex items-start space-x-2.5">
              <svg className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <div>
                <p className="font-semibold text-amber-800">Reference Alignment Notice</p>
                <p className="mt-0.5 text-amber-700">{uploadResult.reference_warning}</p>
              </div>
            </div>
          )}

          {/* Confirmed Details & Preview Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Test Image Card */}
            <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-xs">
              <div className="p-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                <span className="text-xs font-bold text-slate-800 uppercase tracking-wide">
                  Test Wafer Image
                </span>
                <span className="text-[11px] font-mono text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                  Confirmed
                </span>
              </div>

              {/* Preview image served from backend GET endpoint */}
              <div className="h-56 bg-slate-900/5 flex items-center justify-center p-3 border-b border-slate-100">
                <img
                  src={getUploadImageUrl(uploadResult.upload_id, 'test')}
                  alt="Test Wafer backend preview"
                  className="max-h-full max-w-full object-contain rounded shadow-xs"
                />
              </div>

              {/* Server-Confirmed Metadata Table */}
              <div className="p-4 grid grid-cols-2 gap-3 text-xs">
                <div>
                  <span className="text-slate-400 block uppercase text-[10px] font-bold">Filename</span>
                  <span className="font-mono text-slate-800 font-medium truncate block" title={uploadResult.test_image.original_filename}>
                    {uploadResult.test_image.original_filename}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block uppercase text-[10px] font-bold">Format</span>
                  <span className="font-mono text-slate-800 font-medium">{uploadResult.test_image.detected_format}</span>
                </div>
                <div>
                  <span className="text-slate-400 block uppercase text-[10px] font-bold">Mode</span>
                  <span className="font-mono text-slate-800 font-medium">{uploadResult.test_image.mode}</span>
                </div>
                <div>
                  <span className="text-slate-400 block uppercase text-[10px] font-bold">Dimensions</span>
                  <span className="font-mono text-slate-800 font-medium">
                    {uploadResult.test_image.width} &times; {uploadResult.test_image.height} px
                  </span>
                </div>
                <div className="col-span-2">
                  <span className="text-slate-400 block uppercase text-[10px] font-bold">Size</span>
                  <span className="font-mono text-slate-800 font-medium">
                    {uploadResult.test_image.file_size_bytes.toLocaleString()} bytes
                  </span>
                </div>
              </div>
            </div>

            {/* Reference Image Card (if uploaded) */}
            {uploadResult.reference_image && (
              <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-xs">
                <div className="p-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-800 uppercase tracking-wide">
                    Reference / Golden Image
                  </span>
                  <span className="text-[11px] font-mono text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                    Confirmed
                  </span>
                </div>

                {/* Preview image served from backend GET endpoint */}
                <div className="h-56 bg-slate-900/5 flex items-center justify-center p-3 border-b border-slate-100">
                  <img
                    src={getUploadImageUrl(uploadResult.upload_id, 'reference')}
                    alt="Reference Wafer backend preview"
                    className="max-h-full max-w-full object-contain rounded shadow-xs"
                  />
                </div>

                {/* Server-Confirmed Metadata Table */}
                <div className="p-4 grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <span className="text-slate-400 block uppercase text-[10px] font-bold">Filename</span>
                    <span className="font-mono text-slate-800 font-medium truncate block" title={uploadResult.reference_image.original_filename}>
                      {uploadResult.reference_image.original_filename}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400 block uppercase text-[10px] font-bold">Format</span>
                    <span className="font-mono text-slate-800 font-medium">{uploadResult.reference_image.detected_format}</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block uppercase text-[10px] font-bold">Mode</span>
                    <span className="font-mono text-slate-800 font-medium">{uploadResult.reference_image.mode}</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block uppercase text-[10px] font-bold">Dimensions</span>
                    <span className="font-mono text-slate-800 font-medium">
                      {uploadResult.reference_image.width} &times; {uploadResult.reference_image.height} px
                    </span>
                  </div>
                  <div className="col-span-2">
                    <span className="text-slate-400 block uppercase text-[10px] font-bold">Size</span>
                    <span className="font-mono text-slate-800 font-medium">
                      {uploadResult.reference_image.file_size_bytes.toLocaleString()} bytes
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
};
