import { DESIGNER_API_BASE } from '../config';

async function handleResponse(response) {
  if (!response.ok) {
    let errorData;
    try {
      errorData = await response.json();
    } catch {
      errorData = { error: `HTTP ${response.status}: ${response.statusText}` };
    }
    throw new Error(
      errorData.detail ||
      errorData.error ||
      errorData.message ||
      `HTTP ${response.status}`
    );
  }
  return response;
}

export const apiCall = (endpoint, options = {}) => {
  const url = endpoint.startsWith('http') ? endpoint : `${DESIGNER_API_BASE}${endpoint}`;
  return fetch(url, { ...options, headers: { ...options.headers } }).then(handleResponse);
};
