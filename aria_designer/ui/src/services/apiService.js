import { DESIGNER_API_BASE } from '../config';

async function handleResponse(response) {
  if (!response.ok) {
    let errorData;
    try {
      errorData = await response.json();
    } catch {
      errorData = { error: `HTTP ${response.status}: ${response.statusText}` };
    }
    const detail = errorData.detail;
    const msg = (detail && typeof detail === 'object')
      ? (detail.message || JSON.stringify(detail))
      : (detail || errorData.error || errorData.message || `HTTP ${response.status}`);
    throw new Error(msg);
  }
  return response;
}

export const apiCall = (endpoint, options = {}) => {
  const url = endpoint.startsWith('http') ? endpoint : `${DESIGNER_API_BASE}${endpoint}`;
  return fetch(url, { ...options, headers: { ...options.headers } }).then(handleResponse);
};
