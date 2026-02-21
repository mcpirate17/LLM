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

export const apiService = {
  // Experiments
  getExperiments: (limit = 100) => 
    fetch(`${API_BASE}/api/experiments?limit=${limit}`).then(handleResponse),
  
  getExperiment: (id) => 
    fetch(`${API_BASE}/api/experiments/${id}`).then(handleResponse),
  
  getExperimentAnalysis: (id) => 
    fetch(`${API_BASE}/api/experiments/${id}/analysis`).then(handleResponse),
  
  startExperiment: (config) => 
    fetch(`${API_BASE}/api/experiments/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    }).then(handleResponse),
  
  stopExperiment: (id) => 
    fetch(`${API_BASE}/api/experiments/${id}/stop`, { method: 'POST' }).then(handleResponse),
  
  rerunExperiment: (id) => 
    fetch(`${API_BASE}/api/experiments/${id}/rerun`, { method: 'POST' }).then(handleResponse),

  // Programs
  getPrograms: (limit = 100) => 
    fetch(`${API_BASE}/api/programs?limit=${limit}`).then(handleResponse),
  
  getProgram: (id) => 
    fetch(`${API_BASE}/api/programs/${id}`).then(handleResponse),

  getProgramLineage: (id) =>
    fetch(`${API_BASE}/api/programs/${id}/lineage`).then(handleResponse),
  
  getLiveFeed: (experimentId, n = 100) =>
    fetch(`${API_BASE}/api/live-feed?experiment_id=${encodeURIComponent(experimentId)}&n=${n}`).then(handleResponse),

  getTrainingCurve: (id) => 
    fetch(`${API_BASE}/api/programs/${id}/training-curve`).then(handleResponse),

  getReproducibilityManifest: (id) =>
    fetch(`${API_BASE}/api/reproducibility-manifest/${id}`).then(handleResponse),

  getDecisionPacket: (id) =>
    fetch(`${API_BASE}/api/decision-packet/${id}`).then(handleResponse),

  // Analytics & Trends
  getTrends: () => 
    fetch(`${API_BASE}/api/trends/context`).then(handleResponse),
  
  getDashboardSummary: () => 
    fetch(`${API_BASE}/api/dashboard`).then(handleResponse),
  
  getLeaderboard: (params = '') => 
    fetch(`${API_BASE}/api/leaderboard${params}`).then(handleResponse),

  getRegressionVsBaseline: (limit = 200) =>
    fetch(`${API_BASE}/api/analytics/regression-vs-baseline?limit=${limit}`).then(handleResponse),

  // Knowledge & Campaigns
  getCampaigns: () => 
    fetch(`${API_BASE}/api/campaigns`).then(handleResponse),
  
  getCampaignHypotheses: (id) => 
    fetch(`${API_BASE}/api/campaigns/${id}/hypotheses`).then(handleResponse),
  
  getCampaignDecisions: (id) => 
    fetch(`${API_BASE}/api/campaigns/${id}/decisions`).then(handleResponse),

  getCampaignHypotheses: (id) => 
    fetch(`${API_BASE}/api/campaigns/${id}/hypotheses`).then(handleResponse),

  // Diagnostics
  getFingerprintDiagnostics: () => 
    fetch(`${API_BASE}/api/diagnostics/fingerprint`).then(handleResponse),
  
  getReportCacheDiagnostics: () => 
    fetch(`${API_BASE}/api/diagnostics/report-cache`).then(handleResponse),

  getAriaCycleStatus: () =>
    fetch(`${API_BASE}/api/aria/cycle-status`).then(handleResponse),

  getAriaChatGuardrails: (window = 200) =>
    fetch(`${API_BASE}/api/aria/chat/guardrails?window=${window}`).then(handleResponse),

  getHealerTasks: (limit = 5) =>
    fetch(`${API_BASE}/api/healer/tasks?limit=${limit}`).then(handleResponse),

  ensureDesignerRunning: (forceRestart = false) =>
    fetch(`${API_BASE}/api/designer/ensure-running`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force_restart: !!forceRestart }),
    }).then(handleResponse),

  touchDesigner: (reason = 'dashboard') =>
    fetch(`${API_BASE}/api/designer/touch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason }),
    }).then(handleResponse),

  stopDesigner: () =>
    fetch(`${API_BASE}/api/designer/stop`, { method: 'POST' }).then(handleResponse),
};

export default apiService;
