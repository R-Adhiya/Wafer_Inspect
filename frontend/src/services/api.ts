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
  inspection_id: number;
  upload_id: string;
  test_image: ImageMetaResponse;
  reference_image?: ImageMetaResponse | null;
  reference_warning?: string | null;
  disclaimer?: string | null;
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
  inspection_id?: number | null;
  upload_id: string;
  status?: string;
  created_at?: string | null;
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

export type RunLifecycleStatus = 'not_started' | 'pending' | 'running' | 'completed' | 'failed';

export interface InspectionStatusResponse {
  inspection_id?: number | null;
  upload_id: string;
  status: RunLifecycleStatus;
  current_stage: string;
  completed_stages: string[];
  stage_history: string[];
  stages: string[];
  error?: string | null;
  updated_at?: string | null;
}

export interface HistoryItem {
  id: number;
  upload_id: string;
  status: string;
  original_filename: string;
  image_width: number;
  image_height: number;
  has_reference: boolean;
  defect_count: number;
  total_defect_area: number;
  largest_defect_area: number;
  severity?: InspectionSeverity | null;
  pattern_type?: PatternType | null;
  pattern_confidence?: number | null;
  detection_path?: string | null;
  detection_label?: string | null;
  detection_method?: string | null;
  model_name?: string | null;
  model_version?: string | null;
  processing_time?: number | null;
  created_at: string;
}

export interface HistoryListResponse {
  items: HistoryItem[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface HistoryQueryParams {
  search?: string;
  severity?: string;
  sort?: 'created_at' | 'defect_count' | 'severity' | 'processing_time';
  order?: 'asc' | 'desc';
  page?: number;
  page_size?: number;
}

export interface InspectionImagesResponse {
  inspection_id: number;
  upload_id: string;
  images: Record<string, string>;
}

export interface DeleteInspectionResponse {
  deleted: boolean;
  inspection_id: number;
  upload_id: string;
}

export interface DashboardStatsResponse {
  total_inspections: number;
  total_defects: number;
  high_severity_inspections: number;
  average_defect_area: number;
  recent_inspections: HistoryItem[];
  recent_defect_counts: {
    inspection_id: number;
    filename: string;
    defect_count: number;
    severity: string;
    created_at?: string | null;
  }[];
  severity_distribution: { severity: string; count: number }[];
  defect_area_histogram: { bin: string; count: number }[];
  pattern_distribution: { pattern_type: string; count: number }[];
}

export interface SampleItem {
  name: string;
  title: string;
  description: string;
  category: string;
  has_reference: boolean;
  image_size: number;
  disclaimer: string;
}

export interface SampleListResponse {
  disclaimer: string;
  samples: SampleItem[];
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
 * Executes CV inspection pipeline on an uploaded wafer session (accepts integer inspection_id or UUID upload_id).
 */
export async function runInspection(
  identifier: number | string,
  configOverrides?: Record<string, any>
): Promise<InspectionRunResponse> {
  const payload =
    typeof identifier === 'number'
      ? { inspection_id: identifier, config: configOverrides || null }
      : { upload_id: identifier, config: configOverrides || null };

  const response = await fetch(`${API_BASE_URL}/inspection/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw await parseApiError(response);
  }

  return response.json();
}

/**
 * Fetches full inspection report reconstructed from the database (GET /inspection/{id}).
 */
export async function getInspectionById(inspectionId: number): Promise<InspectionRunResponse> {
  const response = await fetch(`${API_BASE_URL}/inspection/${inspectionId}`, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    throw await parseApiError(response);
  }
  return response.json();
}

/**
 * Fetches available image artifact URLs for an inspection (GET /inspection/{id}/images).
 */
export async function getInspectionImages(inspectionId: number): Promise<InspectionImagesResponse> {
  const response = await fetch(`${API_BASE_URL}/inspection/${inspectionId}/images`, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    throw await parseApiError(response);
  }
  return response.json();
}

/**
 * Polls live single-process stage status by integer inspection ID (GET /inspection/{id}/status).
 */
export async function getInspectionStatusById(inspectionId: number): Promise<InspectionStatusResponse> {
  const response = await fetch(`${API_BASE_URL}/inspection/${inspectionId}/status`, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    throw await parseApiError(response);
  }
  return response.json();
}

/**
 * Lists paginated, searchable, filterable, and sortable inspection history.
 */
export async function getInspectionHistory(
  params: HistoryQueryParams = {}
): Promise<HistoryListResponse> {
  const qs = new URLSearchParams();
  if (params.search && params.search.trim()) qs.set('search', params.search.trim());
  if (params.severity && params.severity !== 'ALL') qs.set('severity', params.severity);
  if (params.sort) qs.set('sort', params.sort);
  if (params.order) qs.set('order', params.order);
  if (params.page != null) qs.set('page', String(params.page));
  if (params.page_size != null) qs.set('page_size', String(params.page_size));

  const queryStr = qs.toString();
  const url = `${API_BASE_URL}/inspection/history${queryStr ? `?${queryStr}` : ''}`;
  const response = await fetch(url, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    throw await parseApiError(response);
  }
  return response.json();
}

/**
 * Deletes an inspection and its stored upload/result artifacts (DELETE /inspection/{id}).
 */
export async function deleteInspection(inspectionId: number): Promise<DeleteInspectionResponse> {
  const response = await fetch(`${API_BASE_URL}/inspection/${inspectionId}`, {
    method: 'DELETE',
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    throw await parseApiError(response);
  }
  return response.json();
}

/**
 * Fetches SQL-aggregated dashboard KPIs and chart data (GET /dashboard/stats).
 */
export async function getDashboardStats(): Promise<DashboardStatsResponse> {
  const response = await fetch(`${API_BASE_URL}/dashboard/stats`, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    throw await parseApiError(response);
  }
  return response.json();
}

/**
 * Lists synthetic demonstration wafer samples (GET /samples).
 */
export async function getSamples(): Promise<SampleListResponse> {
  const response = await fetch(`${API_BASE_URL}/samples`, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    throw await parseApiError(response);
  }
  return response.json();
}

/**
 * Generates and uploads a synthetic demonstration sample through the standard upload path (POST /samples/{name}).
 */
export async function createSampleUpload(name: string): Promise<UploadInspectionResponse> {
  const response = await fetch(`${API_BASE_URL}/samples/${encodeURIComponent(name)}`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    throw await parseApiError(response);
  }
  return response.json();
}

/**
 * Resolves a backend-relative URL (such as `/inspection/5/image/final_overlay`) against API_BASE_URL.
 */
export function resolveApiUrl(pathOrUrl: string): string {
  if (pathOrUrl.startsWith('http://') || pathOrUrl.startsWith('https://')) {
    return pathOrUrl;
  }
  return `${API_BASE_URL}${pathOrUrl.startsWith('/') ? '' : '/'}${pathOrUrl}`;
}

/**
 * Fetches persisted inspection result by UUID upload_id (compatibility helper).
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
