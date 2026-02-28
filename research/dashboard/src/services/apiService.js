/**
 * Centralized API Service for the AI Scientist Dashboard.
 * Handles all REST communication with the backend.
 */

const API_BASE = process.env.REACT_APP_API_URL || '';

async function handleResponse(response) {
  if (!response.ok) {
    let errorData;
    try {
      errorData = await response.json();
    } catch {
      errorData = { error: `HTTP ${response.status}: ${response.statusText}` };
    }
    throw new Error(errorData.error || `HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Generic request helper to reduce boilerplate.
 */
export const apiCall = (endpoint, options = {}) => {
  const url = endpoint.startsWith('http') ? endpoint : `${API_BASE}${endpoint}`;
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

  return fetch(url, config);
};

const get = (endpoint) => apiCall(endpoint, { method: 'GET' }).then(handleResponse);
const post = (endpoint, body) => apiCall(endpoint, { method: 'POST', body }).then(handleResponse);

export const apiService = {
  // Experiments
  getExperiments: (limit = 100) => get(`/api/experiments?n=${limit}`),
  getExperiment: (id) => get(`/api/experiments/${id}`),
  getExperimentAnalysis: (id) => get(`/api/experiments/${id}/analysis`),
  startExperiment: (config) => post(`/api/experiments/start`, config),
  stopExperiment: (id) => post(`/api/experiments/${id}/stop`),
  rerunExperiment: (id) => post(`/api/experiments/${id}/rerun`),

  // Programs
  getPrograms: (limit = 100) => get(`/api/programs?limit=${limit}`),
  getProgram: (id) => get(`/api/programs/${id}`),
  getProgramLineage: (id) => get(`/api/programs/${id}/lineage`),
  getLiveFeed: (experimentId, n = 100) => get(`/api/live-feed?experiment_id=${encodeURIComponent(experimentId)}&n=${n}`),
  getTrainingCurve: (id) => get(`/api/programs/${id}/training-curve`),
  getReproducibilityManifest: (id) => get(`/api/reproducibility-manifest/${id}`),
  getDecisionPacket: (id) => get(`/api/decision-packet/${id}`),

  // Analytics & Trends
  getTrends: () => get(`/api/trends/context`),
  getDashboardSummary: () => get(`/api/dashboard`),
  getLeaderboard: (params = '') => get(`/api/leaderboard${params}`),
  getReferences: () => get(`/api/references`),
  getRegressionVsBaseline: (limit = 200) => get(`/api/analytics/regression-vs-baseline?limit=${limit}`),

  // Knowledge & Campaigns
  getCampaigns: () => get(`/api/campaigns`),
  getCampaignHypotheses: (id) => get(`/api/campaigns/${id}/hypotheses`),
  getCampaignDecisions: (id) => get(`/api/campaigns/${id}/decisions`),

  // Diagnostics
  getFingerprintDiagnostics: () => get(`/api/diagnostics/fingerprint`),
  getReportCacheDiagnostics: () => get(`/api/diagnostics/report-cache`),
  getAriaCycleStatus: () => get(`/api/aria/cycle-status`),
  getAriaChatGuardrails: (window = 200) => get(`/api/aria/chat/guardrails?window=${window}`),
  getHealerTasks: (limit = 5) => get(`/api/healer/tasks?limit=${limit}`),

  // Designer Integration
  ensureDesignerRunning: (forceRestart = false) => post(`/api/designer/ensure-running`, { force_restart: !!forceRestart }),
  touchDesigner: (reason = 'dashboard') => post(`/api/designer/touch`, { reason }),
  stopDesigner: () => post(`/api/designer/stop`),

  // Native runner profiling
  getNativeProfile: () => get(`/api/native-profile/v2/data`),
  toggleNativeProfiling: (enable) => post(`/api/native-profile/v2/enable`, { enable: !!enable }),
};

export default apiService;
