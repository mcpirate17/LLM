/**
 * Centralized API Service for the AI Scientist Dashboard.
 * Handles all REST communication with the backend.
 */

function _resolveApiBase() {
  const env = process.env.REACT_APP_API_URL;
  if (env) return env;
  return '';
}
const API_BASE = _resolveApiBase();
const DEFAULT_TIMEOUT_MS = 30000;

function buildTimeoutError(timeoutMs) {
  return new Error(
    `Request timed out after ${Math.round(timeoutMs / 1000)}s. The operation may still complete in the background; refresh to confirm.`
  );
}

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
  const { timeoutMs: timeoutOpt, ...requestOptions } = options;
  const url = endpoint.startsWith('http') ? endpoint : `${API_BASE}${endpoint}`;
  const timeoutMs = Number.isFinite(Number(timeoutOpt))
    ? Number(timeoutOpt)
    : DEFAULT_TIMEOUT_MS;
  const controller = requestOptions.signal ? null : new AbortController();
  const config = {
    ...requestOptions,
    headers: {
      ...requestOptions.headers,
    },
    signal: requestOptions.signal || controller?.signal,
  };

  if (requestOptions.body && typeof requestOptions.body === 'object' && !(requestOptions.body instanceof FormData)) {
    config.headers['Content-Type'] = 'application/json';
    config.body = JSON.stringify(requestOptions.body);
  }

  let timeoutId = null;
  if (controller && timeoutMs > 0) {
    timeoutId = setTimeout(() => controller.abort('timeout'), timeoutMs);
  }
  return fetch(url, config)
    .catch((error) => {
      if (
        controller
        && controller.signal.aborted
        && (error?.name === 'AbortError' || error instanceof TypeError)
      ) {
        throw buildTimeoutError(timeoutMs);
      }
      throw error;
    })
    .finally(() => {
      if (timeoutId) clearTimeout(timeoutId);
    });
};

const get = (endpoint) => apiCall(endpoint, { method: 'GET' }).then(handleResponse);
const post = (endpoint, body) => apiCall(endpoint, { method: 'POST', body }).then(handleResponse);
export const postJson = (endpoint, body, options = {}) =>
  apiCall(endpoint, { method: 'POST', body, ...options });
export const putJson = (endpoint, body, options = {}) =>
  apiCall(endpoint, { method: 'PUT', body, ...options });

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
  // Score-stability validation reruns — queue N tasks for the runner.
  queueValidationRerun: (id, body = {}) =>
    post(`/api/programs/${id}/queue-validation-rerun`, body),
  getPendingReruns: (id) => get(`/api/programs/${id}/pending-reruns`),
  cancelPendingRerun: (id, taskId) =>
    post(`/api/programs/${id}/pending-reruns/${taskId}/cancel`),
  // Bulk auto rerun (striking-distance rule).
  previewQueueRerunAuto: (params = '') =>
    get(`/api/leaderboard/queue-rerun-preview${params}`),
  applyQueueRerunAuto: (body) =>
    post(`/api/leaderboard/queue-rerun-apply`, body),
  // Manually drain one pending validation rerun (without starting
  // continuous mode).  Returns 409 if an experiment is already running.
  drainPendingValidationRerun: () =>
    post(`/api/runner/drain-pending-validation-rerun`),
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
  getEfficiencyFrontier: () => get(`/api/analytics/efficiency-frontier`),

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

  // Native runner profiling
  getNativeProfile: () => get(`/api/native-profile/v2/data`),
  toggleNativeProfiling: (enable) => post(`/api/native-profile/v2/enable`, { enable: !!enable }),
};

export default apiService;
