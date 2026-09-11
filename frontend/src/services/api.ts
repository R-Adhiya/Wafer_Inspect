import { HealthResponse } from '../types/health';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Fetches the backend operational status and version.
 * Centralized API call wrapper for health check endpoint.
 */
export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Server responded with HTTP status ${response.status}`);
  }

  const data: HealthResponse = await response.json();
  return data;
}
