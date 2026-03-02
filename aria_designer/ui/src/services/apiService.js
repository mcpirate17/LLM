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
  const config = {
    ...options,
    headers: {
      ...options.headers,
    },
  };
  
  if (options.body && typeof options.body === 'object' && !(options.body instanceof FormData)) {
    config.headers['Content-Type'] = 'application/json';
    config.body = JSON.stringify(options.body);
  }

  return fetch(url, config).then(handleResponse);
};

export const apiService = {
  get: (endpoint) => apiCall(endpoint, { method: 'GET' }),
  post: (endpoint, body) => apiCall(endpoint, { method: 'POST', body }),
  put: (endpoint, body) => apiCall(endpoint, { method: 'PUT', body }),
  delete: (endpoint) => apiCall(endpoint, { method: 'DELETE' }),
};

export default apiService;
