export interface HealthResponse {
  status: string;
  service: string;
}

export interface ImageMetaResponse {
  original_filename: string;
  detected_format: string;
  mode: string;
  width: number;
  height: number;
  file_size_bytes: number;
}

export interface UploadInspectionResponse {
  upload_id: string;
  test_image: ImageMetaResponse;
  reference_image?: ImageMetaResponse | null;
  reference_warning?: string | null;
}

export interface ApiErrorDetail {
  code?: string;
  message: string;
}

const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Fetches the health status of the backend API.
 * Single API access point; no fetch calls anywhere else in the application.
 */
export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`, {
    method: 'GET',
    headers: {
      Accept: 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
  }

  const data: HealthResponse = await response.json();
  return data;
}

/**
 * Uploads wafer inspection images (test_image required, reference_image optional)
 * to the backend inspection upload endpoint.
 */
export async function uploadInspection(
  testImage: File,
  referenceImage?: File | null
): Promise<UploadInspectionResponse> {
  const formData = new FormData();
  formData.append('test_image', testImage);
  if (referenceImage) {
    formData.append('reference_image', referenceImage);
  }

  const response = await fetch(`${API_BASE_URL}/inspection/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorCode: string | undefined = undefined;
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`;

    try {
      const errJson = await response.json();
      if (errJson && errJson.detail) {
        if (typeof errJson.detail === 'object') {
          errorCode = errJson.detail.code;
          errorMessage = errJson.detail.message || errorMessage;
        } else {
          errorMessage = String(errJson.detail);
        }
      }
    } catch {
      // JSON decode failure fallback
    }

    const err = new Error(errorMessage);
    (err as unknown as { code?: string }).code = errorCode;
    throw err;
  }

  return response.json();
}

/**
 * Constructs preview URL for uploaded images served by backend.
 */
export function getUploadImageUrl(uploadId: string, kind: 'test' | 'reference'): string {
  return `${API_BASE_URL}/inspection/upload/${encodeURIComponent(uploadId)}/${kind}`;
}
