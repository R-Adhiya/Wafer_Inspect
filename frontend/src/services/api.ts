export interface HealthResponse {
  status: string;
  service: string;
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
