/**
 * API utility functions for backend communication.
 * 
 * Provides a clean, reusable interface for making HTTP requests to the backend.
 * Handles base URL configuration, JSON serialization, and basic error handling.
 */

/**
 * Base URL for API requests.
 * Uses VITE_API_URL environment variable if set, otherwise defaults to http://localhost:8000
 */
const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * API error class for handling API-specific errors.
 */
export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public response?: unknown
  ) {
    super(message);
    this.name = 'APIError';
  }
}

/**
 * Options for API requests.
 */
interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean>;
}

/**
 * Builds a full URL from a path and optional query parameters.
 */
function buildURL(path: string, params?: Record<string, string | number | boolean>): string {
  const url = new URL(path, baseURL);
  
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.append(key, String(value));
    });
  }
  
  return url.toString();
}

/**
 * Handles API response, throwing APIError for non-OK responses.
 */
async function handleResponse<T>(response: Response): Promise<T> {
  const contentType = response.headers.get('content-type');
  const isJson = contentType?.includes('application/json');
  
  let data: unknown;
  
  if (isJson) {
    data = await response.json();
  } else {
    const text = await response.text();
    data = text || null;
  }
  
  if (!response.ok) {
    // Extract error message from response
    let errorMessage = `API request failed with status ${response.status}`;
    
    if (data && typeof data === 'object') {
      const errorData = data as { detail?: string; message?: string; error?: string };
      errorMessage = errorData.detail || errorData.message || errorData.error || errorMessage;
    } else if (typeof data === 'string') {
      errorMessage = data;
    }
    
    throw new APIError(
      errorMessage,
      response.status,
      data
    );
  }
  
  return data as T;
}

/**
 * Makes a GET request to the API.
 * 
 * @param path - API endpoint path (e.g., '/api/health' or '/health')
 * @param options - Optional request options including query parameters
 * @returns Promise resolving to the response data
 * 
 * @example
 * ```ts
 * const data = await api.get('/api/autotrading/settings');
 * const dataWithParams = await api.get('/api/trades', { params: { limit: 10 } });
 * ```
 */
export async function get<T = unknown>(
  path: string,
  options: RequestOptions = {}
): Promise<T> {
  const { params, ...fetchOptions } = options;
  const url = buildURL(path, params);
  
  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      ...fetchOptions.headers,
    },
    ...fetchOptions,
  });
  
  return handleResponse<T>(response);
}

/**
 * Makes a POST request to the API.
 * 
 * @param path - API endpoint path
 * @param body - Request body (will be JSON stringified)
 * @param options - Optional request options
 * @returns Promise resolving to the response data
 * 
 * @example
 * ```ts
 * const result = await api.post('/api/autotrading/settings', { enabled: true });
 * ```
 */
export async function post<T = unknown>(
  path: string,
  body?: unknown,
  options: RequestOptions = {}
): Promise<T> {
  const { params, ...fetchOptions } = options;
  const url = buildURL(path, params);
  
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...fetchOptions.headers,
    },
    body: body ? JSON.stringify(body) : undefined,
    ...fetchOptions,
  });
  
  return handleResponse<T>(response);
}

/**
 * Makes a PUT request to the API.
 * 
 * @param path - API endpoint path
 * @param body - Request body (will be JSON stringified)
 * @param options - Optional request options
 * @returns Promise resolving to the response data
 * 
 * @example
 * ```ts
 * const result = await api.put('/api/autotrading/settings', { enabled: false });
 * ```
 */
export async function put<T = unknown>(
  path: string,
  body?: unknown,
  options: RequestOptions = {}
): Promise<T> {
  const { params, ...fetchOptions } = options;
  const url = buildURL(path, params);
  
  const response = await fetch(url, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      ...fetchOptions.headers,
    },
    body: body ? JSON.stringify(body) : undefined,
    ...fetchOptions,
  });
  
  return handleResponse<T>(response);
}

/**
 * Makes a DELETE request to the API.
 * 
 * @param path - API endpoint path
 * @param options - Optional request options
 * @returns Promise resolving to the response data
 * 
 * @example
 * ```ts
 * await api.del('/api/trades/123');
 * ```
 */
export async function del<T = unknown>(
  path: string,
  options: RequestOptions = {}
): Promise<T> {
  const { params, ...fetchOptions } = options;
  const url = buildURL(path, params);
  
  const response = await fetch(url, {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json',
      ...fetchOptions.headers,
    },
    ...fetchOptions,
  });
  
  return handleResponse<T>(response);
}

/**
 * Default export with all API methods.
 */
const api = {
  get,
  post,
  put,
  delete: del,
  baseURL,
};

export default api;

