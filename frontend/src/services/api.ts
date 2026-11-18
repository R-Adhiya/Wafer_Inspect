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

export interface ImageStats {
  width: number;
  height: number;
  mean: number;
  std: number;
}

export interface StepLogItem {
  step: string;
  parameters: Record<string, any>;
  applied: boolean;
  reason: string;
  duration_ms: number;
  mean: number;
  std: number;
}

export interface StageCount {
  stage: string;
  candidate_count: number;
  note?: string | null;
}

export interface DetectionSummary {
  path: 'golden' | 'self_reference' | 'no_reference';
  label: string;
  threshold_method: string;
  threshold_value: number;
  threshold_reason: string;
  wafer_center: [number, number];
  wafer_radius: number;
  is_full_frame: boolean;
  alignment_info: Record<string, any>;
  stage_counts: StageCount[];
  warnings: string[];
  die_pitch?: [number, number] | null;
  peak_strength?: number | null;
  coverage?: number | null;
  unverifiable_fraction?: number | null;
}

export interface AcceptedRegion {
  id: number;
  bbox: [number, number, number, number];
  area_px: number;
  centroid: [number, number];
  polarity: 'bright' | 'dark';
  mean_contrast: number;
  peak_contrast: number;
  touches_edge: boolean;
  reason?: string;
  original_bbox?: [number, number, number, number] | null;
  original_centroid?: [number, number] | null;
}

export interface RejectedRegion {
  id: number;
  area_px: number;
  mean_contrast: number;
  reason: string;
}

export type RunImageKind =
  | 'original'
  | 'preprocessed'
  | 'difference'
  | 'mask'
  | 'filtered_mask'
  | 'bboxes'
  | 'contours';

export interface InspectionRunResponse {
  upload_id: string;
  step_log: StepLogItem[];
  original_stats: ImageStats;
  preprocessed_stats: ImageStats;
  original_image_url: string;
  preprocessed_image_url: string;
  difference_image_url?: string | null;
  mask_image_url?: string | null;
  filtered_mask_image_url?: string | null;
  bboxes_image_url?: string | null;
  contours_image_url?: string | null;
  detection?: DetectionSummary | null;
  stage_counts: StageCount[];
  regions: AcceptedRegion[];
  rejected_regions: RejectedRegion[];
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

/**
 * Executes CV preprocessing on an uploaded wafer inspection session.
 */
export async function runInspection(
  uploadId: string,
  configOverrides?: Record<string, any>
): Promise<InspectionRunResponse> {
  const response = await fetch(`${API_BASE_URL}/inspection/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({
      upload_id: uploadId,
      config: configOverrides || null,
    }),
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
 * Constructs preview URL for inspection run images
 * (original, preprocessed, difference, mask, filtered_mask, bboxes, contours).
 */
export function getRunImageUrl(uploadId: string, kind: RunImageKind): string {
  return `${API_BASE_URL}/inspection/run/${encodeURIComponent(uploadId)}/image/${kind}`;
}
