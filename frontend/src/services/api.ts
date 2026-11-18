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

export type DefectSeverity = 'LOW' | 'MEDIUM' | 'HIGH';
export type InspectionSeverity = 'NONE' | DefectSeverity;

export type PatternType =
  | 'INSUFFICIENT_DATA'
  | 'EDGE_RING'
  | 'CENTER_CLUSTER'
  | 'DONUT'
  | 'LOCAL_CLUSTER'
  | 'SCRATCH_LINE'
  | 'RANDOM_SCATTER';

export interface RadialBinMetric {
  bin: string;
  r_min: number;
  r_max: number;
  observed: number;
  expected_random: number;
}

export interface PatternResult {
  type: PatternType;
  confidence: number;
  p_value?: number | null;
  explanation: string;
  metrics: {
    n?: number;
    wafer_center_orig?: [number, number];
    wafer_radius_orig?: number;
    edge_fraction?: number;
    center_fraction?: number;
    donut_fraction?: number;
    largest_angular_gap_deg?: number;
    line_explained_variance?: number;
    line_span_ratio?: number;
    nn_ratio?: number;
    radial_bins?: RadialBinMetric[];
    [key: string]: any;
  };
  supporting_defect_ids: number[];
  warnings: string[];
}

export interface AcceptedRegion {
  id: number;
  bbox: [number, number, number, number];
  width_px: number;
  height_px: number;
  area_px: number;
  detected_area_px?: number;
  centroid: [number, number];
  perimeter_px: number;
  equivalent_diameter_px: number;
  length_px: number;
  thickness_px: number;
  angle_deg: number;
  elongation: number;
  polarity: 'bright' | 'dark';
  mean_contrast: number;
  peak_contrast: number;
  contrast_to_noise: number;
  touches_edge: boolean;
  detected_by: 'golden' | 'self_reference' | 'background';
  why_flagged: string;
  severity: DefectSeverity;
  area_um2?: number | null;
  length_um?: number | null;
  thickness_um?: number | null;
  contour?: [number, number][];
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

export interface InspectionSummary {
  defect_count: number;
  total_defect_area_px: number;
  largest_defect_area_px: number;
  average_defect_area_px: number;
  severity: InspectionSeverity;
  severity_reason: string;
  calibration_status: string;
  pixel_size_um?: number | null;
  pattern?: PatternResult | null;
}

export interface ModelInfo {
  name: string;
  version: string;
  detection_method: string;
}

export type RunImageKind =
  | 'original'
  | 'preprocessed'
  | 'difference'
  | 'mask'
  | 'filtered_mask'
  | 'bboxes'
  | 'contours'
  | 'final_overlay'
  | 'pattern_overlay';

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
  final_overlay_image_url?: string | null;
  pattern_overlay_image_url?: string | null;
  detection?: DetectionSummary | null;
  stage_counts: StageCount[];
  regions: AcceptedRegion[];
  rejected_regions: RejectedRegion[];
  summary?: InspectionSummary | null;
  model?: ModelInfo | null;
  pattern?: PatternResult | null;
  processing_time_s: number;
  stage_timings_ms: Record<string, number>;
  test_filename?: string | null;
  test_mode?: string | null;
  reference_used?: boolean;
  reference_filename?: string | null;
}

export type RunLifecycleStatus = 'not_started' | 'running' | 'completed' | 'failed';

export interface InspectionStatusResponse {
  upload_id: string;
  status: RunLifecycleStatus;
  current_stage: string;
  completed_stages: string[];
  stage_history: string[];
  stages: string[];
  error?: string | null;
  updated_at?: string | null;
}

export interface ApiError extends Error {
  code?: string;
  status?: number;
}

export interface ApiErrorDetail {
  code?: string;
  message: string;
}

const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Maps a non-OK backend HTTP response into a structured Error with optional `code` and `status`.
 */
export async function parseApiError(response: Response): Promise<ApiError> {
  let errorCode: string | undefined = undefined;
  let errorMessage = `HTTP ${response.status}: ${response.statusText || 'Request failed'}`;

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
    // Non-JSON response body fallback
  }

  const err: ApiError = new Error(errorMessage);
  err.code = errorCode;
  err.status = response.status;
  return err;
}

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
    throw await parseApiError(response);
  }

  return response.json();
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
    throw await parseApiError(response);
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
 * Executes CV inspection pipeline on an uploaded wafer session.
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
    throw await parseApiError(response);
  }

  return response.json();
}

/**
 * Fetches persisted inspection result for a completed run (GET /inspection/run/{upload_id}/result).
 */
export async function getInspectionResult(uploadId: string): Promise<InspectionRunResponse> {
  const response = await fetch(`${API_BASE_URL}/inspection/run/${encodeURIComponent(uploadId)}/result`, {
    method: 'GET',
    headers: {
      Accept: 'application/json',
    },
  });

  if (!response.ok) {
    throw await parseApiError(response);
  }

  return response.json();
}

/**
 * Polls live single-process stage status for an inspection run (GET /inspection/run/{upload_id}/status).
 */
export async function getInspectionStatus(uploadId: string): Promise<InspectionStatusResponse> {
  const response = await fetch(`${API_BASE_URL}/inspection/run/${encodeURIComponent(uploadId)}/status`, {
    method: 'GET',
    headers: {
      Accept: 'application/json',
    },
  });

  if (!response.ok) {
    throw await parseApiError(response);
  }

  return response.json();
}

/**
 * Constructs preview URL for inspection run images.
 */
export function getRunImageUrl(uploadId: string, kind: RunImageKind): string {
  return `${API_BASE_URL}/inspection/run/${encodeURIComponent(uploadId)}/image/${kind}`;
}
