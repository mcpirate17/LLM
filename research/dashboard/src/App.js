import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import AriaAvatar from './components/AriaAvatar';
import AriaStatus from './components/AriaStatus';
import SummaryCards from './components/SummaryCards';
import ExperimentList from './components/ExperimentList';
import ExperimentDetail from './components/ExperimentDetail';
import TopPrograms from './components/TopPrograms';
import ProgramDetail from './components/ProgramDetail';
import InsightsPanel from './components/InsightsPanel';
import LabNotebook from './components/LabNotebook';
import MetricsChart from './components/MetricsChart';
import ControlPanel from './components/ControlPanel';
import LiveFeed from './components/LiveFeed';
import TrendCharts, { ExperimentDataTab } from './components/TrendCharts';
import PerfDashboard from './components/PerfDashboard';
import LearningPanel from './components/LearningPanel';
import CycleTimeline from './components/CycleTimeline';
import ResearchReport from './components/ResearchReport';
import HelpPanel from './components/HelpPanel';
import Leaderboard from './components/Leaderboard';
import Discoveries from './components/Discoveries';
import CampaignView from './components/CampaignView';
import KnowledgeBase from './components/KnowledgeBase';
import CompareView from './components/CompareView';
import StrategyAdvisor from './components/StrategyAdvisor';
import GlobalParetoChart from './components/GlobalParetoChart';
import ActionQueue from './components/ActionQueue';
import AriaChatPanel from './components/AriaChatPanel';
import ArchitectureDrawer from './components/ArchitectureDrawer';
import StatusBar from './components/StatusBar';
import NativeProfilePanel from './components/NativeProfilePanel';
import { EventBusProvider } from './hooks/useEventBus';
import { AriaDataProvider, useAriaData } from './hooks/useAriaData';
import apiService, { apiCall } from './services/apiService';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || '';
const DEFAULT_EXPERIMENTS_PAGE_SIZE = 200;
const INVESTIGATION_QUEUE_KEY = 'aria_investigation_queue_v1';
const AUTO_REPAIR_SHOW_COMPLETED_KEY = 'aria_auto_repair_show_completed_v1';
const OVERRIDE_INELIGIBLE_ALWAYS_KEY = 'aria_override_ineligible_always_v1';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error", error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="card" style={{ border: '1px solid var(--accent-red)', padding: 20 }}>
          <h3 style={{ color: 'var(--accent-red)' }}>Component crashed</h3>
          <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>{this.state.error?.message}</p>
          <button className="refresh-btn" onClick={() => this.setState({ hasError: false })}>Retry</button>
        </div>
      );
    }
    return this.props.children;
  }
}

function normalizeQueue(items) {
  if (!Array.isArray(items)) return [];
  const seen = new Set();
  const normalized = [];
  for (const item of items) {
    const resultId = item?.resultId;
    if (!resultId || seen.has(resultId)) continue;
    seen.add(resultId);
    normalized.push({
      resultId,
      fingerprint: item?.fingerprint || null,
      source: item?.source || 'unknown',
      architectureFamily: item?.architectureFamily || null,
      intent: item?.intent === 'validation' ? 'validation' : 'investigation',
    });
  }
  return normalized;
}

function buildCandidateEligibility(entry) {
  if (!entry || typeof entry !== 'object') {
    return {
      investigationEligible: false,
      validationEligible: false,
      queueEligible: false,
      queueReason: 'missing_candidate_data',
    };
  }

  const tier = typeof entry.tier === 'string' ? entry.tier.toLowerCase() : '';
  const hasInvestigationEvidence = entry.investigation_loss_ratio != null || entry.investigation_robustness != null;
  const investigationEligible = tier === 'screening';
  const validationEligible = tier === 'investigation' && Boolean(entry.investigation_passed);

  let queueReason = null;
  if (!investigationEligible && !validationEligible) {
    if (tier === 'screening' && hasInvestigationEvidence) {
      queueReason = 'already_investigated_unchanged';
    } else if (tier === 'investigation' && !entry.investigation_passed) {
      queueReason = 'not_investigation_passed';
    } else if (tier === 'validation' || tier === 'breakthrough') {
      queueReason = 'already_promoted';
    } else {
      queueReason = 'not_progression_eligible';
    }
  }

  return {
    investigationEligible,
    validationEligible,
    queueEligible: investigationEligible || validationEligible,
    queueReason,
  };
}

function buildEligibilityByResultId(entries) {
  const map = {};
  for (const entry of Array.isArray(entries) ? entries : []) {
    const resultId = entry?.result_id;
    if (!resultId) continue;
    map[resultId] = buildCandidateEligibility(entry);
  }
  return map;
}

function queueReasonLabel(reason) {
  if (reason === 'already_investigated_unchanged') {
    return 'Candidate already has investigation evidence and is unchanged.';
  }
  if (reason === 'not_investigation_passed') {
    return 'Candidate is investigation-tier but has not passed robustness gate.';
  }
  if (reason === 'already_promoted') {
    return 'Candidate is already in validation/breakthrough tier.';
  }
  if (reason === 'not_progression_eligible') {
    return 'Candidate is not currently eligible for investigation/validation progression.';
  }
  return 'Candidate is not eligible for this queue action.';
}

function resolveQueueIntent(candidate, eligibility) {
  if (candidate?.intent === 'investigation' || candidate?.intent === 'validation') {
    return candidate.intent;
  }
  if (candidate?.validationEligible || eligibility?.validationEligible) {
    return 'validation';
  }
  if (candidate?.investigationEligible || eligibility?.investigationEligible) {
    return 'investigation';
  }
  return null;
}

function isTerminalAgentStatus(status) {
  const normalized = String(status || '').toLowerCase();
  return normalized === 'completed' || normalized === 'failed';
}

function mergeAutoRepairTask(existing, incoming, fallbackSource = 'start') {
  if (!incoming || !incoming.task_id) return null;
  return {
    ...(existing || {}),
    ...incoming,
    source: incoming.source || existing?.source || fallbackSource,
    status: incoming.status || existing?.status || 'queued',
    updated_at: incoming.updated_at || Date.now() / 1000,
  };
}

const LOG_SUB_TABS = [
  { key: 'notebook', label: 'Notebook' },
  { key: 'cycles', label: 'Cycles' },
];

const ANALYTICS_SUB_TABS = [
  { key: 'trends', label: 'Trends' },
  { key: 'data', label: 'Data' },
  { key: 'insights', label: 'Insights' },
  { key: 'learning', label: 'Learning' },
];

function ReferenceBaselinesPanel() {
  const [refs, setRefs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function fetchRefs() {
      try {
        const res = await apiCall(`/api/discoveries?sort=composite_score&limit=200&view=ranked`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        const entries = (json.entries || []).filter(e => e.is_reference);
        if (!cancelled) setRefs(entries);
      } catch {
        // silently fail — references panel is supplementary
      }
      if (!cancelled) setLoading(false);
    }
    fetchRefs();
    const iv = setInterval(fetchRefs, 30000);
    return () => { cancelled = true; clearInterval(iv); };
  }, []);

  if (loading) return null;
  if (refs.length === 0) return null;

  const metricKeys = [
    { key: 'screening_loss_ratio', label: 'Loss Ratio', fmt: v => v?.toFixed(4) },
    { key: 'moe_routing_efficiency', label: 'MoE Eff', fmt: v => v?.toFixed(3) },
    { key: 'arch_quality_score', label: 'Arch Q', fmt: v => v?.toFixed(3) },
    { key: 'screening_novelty', label: 'Novelty', fmt: v => v?.toFixed(4) },
    { key: 'composite_score', label: 'Score', fmt: v => v?.toFixed(4) },
    { key: 'validation_baseline_ratio', label: 'vs Baseline', fmt: v => v ? v.toFixed(2) + 'x' : '--' },
    { key: 'param_efficiency', label: 'Param Eff', fmt: v => v ? v.toFixed(1) : '--' },
    { key: 'quant_int8_retention', label: 'Quant Ret', fmt: v => v ? (v * 100).toFixed(1) + '%' : '--' },
    { key: 'robustness_noise_score', label: 'Noise', fmt: v => v?.toFixed(2) },
    { key: 'init_sensitivity_std', label: 'Init Std', fmt: v => v?.toFixed(4) },
  ];

  return (
    <div className="card" style={{
      padding: 16, marginTop: 16,
      border: '1px solid var(--accent-purple)',
      background: 'rgba(188, 140, 255, 0.04)',
    }}>
      <div style={{
        fontSize: 13, fontWeight: 700, marginBottom: 12,
        color: 'var(--accent-purple)', textTransform: 'uppercase', letterSpacing: 0.5,
      }}>
        Reference Architecture Baselines
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table table-compact">
          <thead>
            <tr style={{ borderBottom: '2px solid var(--border)' }}>
              <th style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--text-muted)', fontWeight: 600 }}>Architecture</th>
              {metricKeys.map(m => (
                <th key={m.key} style={{ textAlign: 'right', padding: '6px 8px', color: 'var(--text-muted)', fontWeight: 600 }}>{m.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {refs.sort((a, b) => (a.screening_loss_ratio || 99) - (b.screening_loss_ratio || 99)).map(ref => (
              <tr key={ref.entry_id || ref.result_id} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '8px', fontWeight: 600, color: 'var(--accent-purple)' }}>
                  {ref.reference_name || ref.architecture_desc || 'Reference'}
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 400, marginTop: 2 }}>
                    {ref.tags?.split(',').filter(t => t !== 'reference').join(', ')}
                  </div>
                </td>
                {metricKeys.map(m => {
                  const val = ref[m.key];
                  return (
                    <td key={m.key} style={{ textAlign: 'right', padding: '8px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                      {val != null ? m.fmt(Number(val)) : '--'}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 8 }}>
        Aria-generated architectures should aim to beat these baselines. Target: 3-5x lower loss ratio than best reference.
      </div>
    </div>
  );
}

function AnalyticsTab({ data, tabData, tabErrors, onSelectExperiment, onRerunExperiment, onFillGapsExperiment, onNavigateStrategy, onStartExperiment }) {
  const [analyticsView, setAnalyticsView] = useState('trends');
  return (
    <>
      <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
        {ANALYTICS_SUB_TABS.map(st => (
          <button
            key={st.key}
            className="refresh-btn"
            style={{
              fontSize: 12,
              padding: '4px 12px',
              fontWeight: analyticsView === st.key ? 700 : 400,
              background: analyticsView === st.key ? 'var(--accent-blue)' : 'transparent',
              color: analyticsView === st.key ? '#fff' : 'var(--text-secondary)',
              borderColor: analyticsView === st.key ? 'var(--accent-blue)' : 'var(--border)',
            }}
            onClick={() => setAnalyticsView(st.key)}
          >
            {st.label}
          </button>
        ))}
      </div>
      {analyticsView === 'trends' && (
        <ErrorBoundary>
          <TrendCharts onSelectExperiment={onSelectExperiment} />
        </ErrorBoundary>
      )}
      {analyticsView === 'data' && (
        <ErrorBoundary>
          <ExperimentDataTab
            onSelectExperiment={onSelectExperiment}
            onRerunExperiment={onRerunExperiment}
            onFillGapsExperiment={onFillGapsExperiment}
            onStartExperiment={onStartExperiment}
          />
        </ErrorBoundary>
      )}
      {analyticsView === 'insights' && (
        <ErrorBoundary>
          <div>
            {tabErrors.insights && (
              <div className="error-banner" style={{ marginBottom: 12 }}>
                Fresh insights fetch failed ({tabErrors.insights}); showing dashboard snapshot.
              </div>
            )}
            <InsightsPanel insights={tabData.insights || data?.insights} />
          </div>
        </ErrorBoundary>
      )}
      {analyticsView === 'learning' && (
        <ErrorBoundary>
          <LearningPanel onNavigateStrategy={onNavigateStrategy} onStartExperiment={onStartExperiment} />
        </ErrorBoundary>
      )}
      {/* Reference baselines pinned at bottom of all analytics views */}
      <ErrorBoundary>
        <ReferenceBaselinesPanel />
      </ErrorBoundary>
    </>
  );
}

function LogTab({ entries, entriesError, onSelectExperiment }) {
  const [logView, setLogView] = useState('notebook');
  return (
    <>
      <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
        {LOG_SUB_TABS.map(st => (
          <button
            key={st.key}
            className="refresh-btn"
            style={{
              fontSize: 12,
              padding: '4px 12px',
              fontWeight: logView === st.key ? 700 : 400,
              background: logView === st.key ? 'var(--accent-blue)' : 'transparent',
              color: logView === st.key ? '#fff' : 'var(--text-secondary)',
              borderColor: logView === st.key ? 'var(--accent-blue)' : 'var(--border)',
            }}
            onClick={() => setLogView(st.key)}
          >
            {st.label}
          </button>
        ))}
      </div>
      {logView === 'notebook' && (
        <>
          {entriesError && (
            <div className="error-banner" style={{ marginBottom: 12 }}>
              Fresh notebook fetch failed ({entriesError}); showing dashboard snapshot.
            </div>
          )}
          <LabNotebook entries={entries} onSelectExperiment={onSelectExperiment} />
        </>
      )}
      {logView === 'cycles' && <CycleTimeline />}
    </>
  );
}

function QuickAnalyticsPreview({ deltas, learningTrajectory, summary, onOpenAnalytics }) {
  const trend = learningTrajectory?.trend;
  const trendLabel = trend === 'improving'
    ? 'Improving'
    : trend === 'declining'
      ? 'Declining'
      : trend === 'plateaued'
        ? 'Plateaued'
        : 'Insufficient data';
  const trendColor = trend === 'improving'
    ? 'var(--accent-green)'
    : trend === 'declining'
      ? 'var(--accent-red, #e74c3c)'
      : 'var(--accent-yellow)';

  const deltaLoss = deltas?.best_loss;
  const deltaNovelty = deltas?.best_novelty;
  const deltaStage1 = deltas?.stage1;
  const deltaPrograms = deltas?.programs;

  const routingEntropy = summary?.avg_routing_entropy;
  const depthSavings = summary?.avg_depth_savings;
  const avgThroughput = summary?.avg_throughput_tok_s;
  const avgRecursionSavings = summary?.avg_recursion_savings;
  const avgRoutingRetention = summary?.avg_routing_token_retention;

  return (
    <div className="card" style={{ marginTop: 12 }}>
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
        <span>Quick Analytics</span>
        <button className="refresh-btn" onClick={onOpenAnalytics} style={{ fontSize: 11 }}>
          Open Analytics
        </button>
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
        <div>
          Trend: <strong style={{ color: trendColor }}>{trendLabel}</strong>
          {learningTrajectory?.slope != null && (
            <span style={{ color: trendColor }}> · {learningTrajectory.slope > 0 ? '+' : ''}{(learningTrajectory.slope * 100).toFixed(2)}%/exp</span>
          )}
        </div>
        <div>
          Δ Loss: {deltaLoss != null ? `${deltaLoss > 0 ? '+' : ''}${deltaLoss.toFixed(4)}` : 'n/a'} ·
          Δ Novelty: {deltaNovelty != null ? `${deltaNovelty > 0 ? '+' : ''}${deltaNovelty.toFixed(3)}` : 'n/a'}
        </div>
        <div>
          Δ S1 survivors: {deltaStage1 != null ? `${deltaStage1 > 0 ? '+' : ''}${deltaStage1}` : 'n/a'} ·
          Δ Programs: {deltaPrograms != null ? `${deltaPrograms > 0 ? '+' : ''}${deltaPrograms}` : 'n/a'}
        </div>
        <div>
          Avg throughput: {avgThroughput != null ? `${Math.round(avgThroughput).toLocaleString()} tok/s` : 'n/a'} ·
          Routing entropy: {routingEntropy != null ? routingEntropy.toFixed(2) : 'n/a'} ·
          Token retention: {avgRoutingRetention != null ? `${(avgRoutingRetention * 100).toFixed(1)}%` : 'n/a'}
        </div>
        <div>
          Depth savings: {depthSavings != null ? `${(depthSavings * 100).toFixed(1)}%` : 'n/a'} ·
          Recursion savings: {avgRecursionSavings != null ? `${(avgRecursionSavings * 100).toFixed(1)}%` : 'n/a'}
        </div>
      </div>
    </div>
  );
}

function App() {
  const [isRunning, setIsRunning] = useState(false);
  return (
    <EventBusProvider apiBase={API_BASE}>
      <AriaDataProvider apiBase={API_BASE} isRunning={isRunning}>
        <AppContent onRunningChange={setIsRunning} />
      </AriaDataProvider>
    </EventBusProvider>
  );
}

const NAV_CATEGORIES = {
  workbench: {
    label: 'Workbench',
    tabs: ['command', 'experiments', 'discoveries', 'comparison'],
  },
  knowledge: {
    label: 'Knowledge',
    tabs: ['reports', 'trends', 'log'],
  },
  diagnostics: {
    label: 'Diagnostics',
    tabs: ['perf'],
  }
};

function AppContent({ onRunningChange }) {
  const TAB_LABELS = {
    command: 'Command',
    trends: 'Analytics',
    experiments: 'Experiments',
    discoveries: 'Discoveries',
    comparison: 'Comparison',
    perf: 'Optimization',
    reports: 'Reports',
    log: 'Log',
  };
  const TAB_TIPS = {
    command: 'Control center — start/stop experiments, see live status (1)',
    trends: 'Analytics: trends, learning signals, and diagnostic charts (2)',
    experiments: 'Browse all experiments and their results (3)',
    discoveries: 'Best architectures found so far, ranked by tier (4)',
    comparison: 'Side-by-side architecture comparison (5)',
    perf: 'System performance and optimization metrics (6)',
    reports: 'Publishable findings, campaigns, and knowledge base (7)',
    log: 'Raw notebook entries and cycle timeline (8)',
  };
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('command');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [dashboardUpdatedAt, setDashboardUpdatedAt] = useState(null);
  const [overviewActivityTab, setOverviewActivityTab] = useState('recent');
  const [showHelp, setShowHelp] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [tabData, setTabData] = useState({
    experiments: null,
    programs: null,
    entries: null,
    insights: null,
  });
  const [tabErrors, setTabErrors] = useState({
    experiments: null,
    programs: null,
    entries: null,
    insights: null,
  });
  const [experimentsPageSize, setExperimentsPageSize] = useState(DEFAULT_EXPERIMENTS_PAGE_SIZE);
  const [experimentsHasMore, setExperimentsHasMore] = useState(true);
  const [experimentsLoadingMore, setExperimentsLoadingMore] = useState(false);

  // Drill-down state
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [selectedProgram, setSelectedProgram] = useState(null);

  // Action error state (replaces alert())
  const [actionError, setActionError] = useState(null);
  const [blockedConfig, setBlockedConfig] = useState(null);
  const [autoRepairTasks, setAutoRepairTasks] = useState([]);
  const [showCompletedAutoRepairTasks, setShowCompletedAutoRepairTasks] = useState(() => {
    try {
      return window.localStorage.getItem(AUTO_REPAIR_SHOW_COMPLETED_KEY) === '1';
    } catch {
      return false;
    }
  });
  const [overrideIneligibleAlways, setOverrideIneligibleAlways] = useState(() => {
    try {
      return window.localStorage.getItem(OVERRIDE_INELIGIBLE_ALWAYS_KEY) === '1';
    } catch {
      return false;
    }
  });

  // Architecture designer drawer
  const [designerSession, setDesignerSession] = useState({ open: false, resultId: null });
  const openDesignerBlank = useCallback(() => {
    setDesignerSession({ open: true, resultId: null });
  }, []);
  const openDesignerForResult = useCallback((rid) => {
    setDesignerSession({ open: true, resultId: rid || null });
  }, []);
  const closeDesigner = useCallback(() => {
    setDesignerSession({ open: false, resultId: null });
  }, []);

  // Cross-view navigation state
  const [leaderboardHighlight, setLeaderboardHighlight] = useState(null);
  const [selectedCampaignId, setSelectedCampaignId] = useState(null);
  const [controlPanelPrefill, setControlPanelPrefill] = useState(null);
  const [activeOverviewStrategy, setActiveOverviewStrategy] = useState(null);
  const [ariaCycle, setAriaCycle] = useState(null);
  const [cycleControlBusy, setCycleControlBusy] = useState(false);
  const [fingerprintLookup, setFingerprintLookup] = useState('');
  const [fingerprintLookupBusy, setFingerprintLookupBusy] = useState(false);
  const [fingerprintLookupError, setFingerprintLookupError] = useState(null);
  const [allowAdvancedStartOverride, setAllowAdvancedStartOverride] = useState(false);
  const [autonomousMode, setAutonomousMode] = useState(false);
  const [healerTasks, setHealerTasks] = useState([]);
  const advancedDetailsRef = useRef(null);
  const [investigationQueue, setInvestigationQueue] = useState(() => {
    try {
      const stored = window.localStorage.getItem(INVESTIGATION_QUEUE_KEY);
      return normalizeQueue(stored ? JSON.parse(stored) : []);
    } catch {
      return [];
    }
  });

  const [comparisonList, setComparisonList] = useState([]);

  const handleAddToComparison = useCallback((resultId) => {
    setComparisonList(prev => {
      if (prev.includes(resultId)) return prev;
      if (prev.length >= 5) {
        setActionError("Max 5 candidates for comparison.");
        return prev;
      }
      return [...prev, resultId];
    });
  }, []);

  const handleRemoveFromComparison = useCallback((resultId) => {
    setComparisonList(prev => prev.filter(id => id !== resultId));
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(INVESTIGATION_QUEUE_KEY, JSON.stringify(investigationQueue));
    } catch {
      // Ignore localStorage failures.
    }
  }, [investigationQueue]);

  useEffect(() => {
    try {
      window.localStorage.setItem(
        AUTO_REPAIR_SHOW_COMPLETED_KEY,
        showCompletedAutoRepairTasks ? '1' : '0',
      );
    } catch {}
  }, [showCompletedAutoRepairTasks]);

  useEffect(() => {
    try {
      window.localStorage.setItem(
        OVERRIDE_INELIGIBLE_ALWAYS_KEY,
        overrideIneligibleAlways ? '1' : '0',
      );
    } catch {}
  }, [overrideIneligibleAlways]);

  const fetchDashboard = useCallback(async () => {
    try {
      const [json, cycleJson, healerJson] = await Promise.all([
        apiService.getDashboardSummary(),
        apiService.getAriaCycleStatus().catch(() => null),
        apiService.getHealerTasks(5).catch(() => []),
      ]);
      
      setData(json);
      setDashboardUpdatedAt(Date.now());
      if (setSummary && json?.summary) setSummary(json.summary);
      setError(null);
      
      if (cycleJson && !cycleJson.error) {
        setAriaCycle(cycleJson);
      }
      if (Array.isArray(healerJson)) {
        setHealerTasks(healerJson);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setInitialLoading(false);
    }
  }, []);

  // Sync isRunning up to App for AriaDataProvider polling speed
  useEffect(() => {
    if (onRunningChange) onRunningChange(Boolean(data?.is_running));
  }, [data?.is_running, onRunningChange]);

  useEffect(() => {
    if (
      ariaCycle
      && typeof ariaCycle.continuous_active === 'boolean'
    ) {
      const autonomousActive = Boolean(ariaCycle.continuous_active) && !Boolean(ariaCycle.cycle_paused);
      setAutonomousMode(autonomousActive);
    }
  }, [ariaCycle]);

  // Global keyboard shortcuts
  useEffect(() => {
    const TAB_KEYS = ['command', 'trends', 'experiments', 'discoveries', 'perf', 'reports', 'log'];
    const handler = (e) => {
      // Ignore when typing in inputs/textareas/selects
      const tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable) return;

      if (e.key === '?') { e.preventDefault(); setShowHelp(h => !h); return; }
      if (e.key === 'Escape') {
        if (showHelp) { setShowHelp(false); return; }
        if (showChat) { setShowChat(false); return; }
        if (showSettings) { setShowSettings(false); return; }
        if (designerSession.open) { closeDesigner(); return; }
        if (selectedProgram) { setSelectedProgram(null); return; }
        return;
      }
      const num = parseInt(e.key, 10);
      if (num >= 1 && num <= TAB_KEYS.length) {
        e.preventDefault();
        setActiveTab(TAB_KEYS[num - 1]);
        setSelectedExperiment(null);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [showHelp, showChat, showSettings, selectedProgram, designerSession.open, closeDesigner]);

  // Shared analytics from AriaDataProvider
  const {
    learningTrajectory,
    leaderboardEntries,
    fingerprintDiagnostics,
    setSummary,
    refreshSharedData,
  } = useAriaData() || {};

  const eligibilityByResultId = useMemo(
    () => buildEligibilityByResultId(leaderboardEntries || []),
    [leaderboardEntries],
  );

  useEffect(() => {
    fetchDashboard();
  }, [fetchDashboard]);

  // Faster refresh when experiment is running
  const refreshInterval = data?.is_running ? 3000 : 10000;

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchDashboard, refreshInterval);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchDashboard, refreshInterval]);

  useEffect(() => {
    if (data?.is_running && overviewActivityTab !== 'live') {
      setOverviewActivityTab('live');
    }
  }, [data?.is_running, overviewActivityTab]);

  useEffect(() => {
    setAllowAdvancedStartOverride(false);
  }, [activeOverviewStrategy?.id]);

  const fetchTabFreshData = useCallback(async (tab, options = {}) => {
    const append = options.append === true;
    const requestedOffset = Number(options.offset || 0);
    const endpoints = {
      experiments: null,
      programs: '/api/programs?n=50&sort=novelty_score',
      entries: '/api/entries?n=50',
      insights: '/api/insights',
    };

    if (tab === 'experiments') {
      const offset = append ? requestedOffset : 0;
      const endpoint = `/api/experiments?n=${experimentsPageSize}&offset=${offset}`;

      try {
        if (append) {
          setExperimentsLoadingMore(true);
        }
        const res = await apiCall(endpoint);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        const page = Array.isArray(json) ? json : [];
        setTabData(prev => ({
          ...prev,
          experiments: append ? [...(Array.isArray(prev.experiments) ? prev.experiments : []), ...page] : page,
        }));
        setTabErrors(prev => ({ ...prev, experiments: null }));
        setExperimentsHasMore(page.length === experimentsPageSize);
      } catch (err) {
        setTabErrors(prev => ({ ...prev, experiments: err.message }));
      } finally {
        if (append) {
          setExperimentsLoadingMore(false);
        }
      }
      return;
    }

    const endpoint = endpoints[tab];
    if (!endpoint) return;

    try {
      const res = await apiCall(endpoint);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setTabData(prev => ({ ...prev, [tab]: Array.isArray(json) ? json : [] }));
      setTabErrors(prev => ({ ...prev, [tab]: null }));
    } catch (err) {
      setTabErrors(prev => ({ ...prev, [tab]: err.message }));
    }
  }, [experimentsPageSize]);

  useEffect(() => {
    if (activeTab === 'experiments') {
      setExperimentsHasMore(true);
      fetchTabFreshData('experiments');
    } else if (activeTab === 'discoveries') {
      fetchTabFreshData('programs');
    } else if (activeTab === 'log') {
      fetchTabFreshData('entries');
    } else if (activeTab === 'trends') {
      fetchTabFreshData('insights');
    }
  }, [activeTab, fetchTabFreshData]);

  const handleLoadMoreExperiments = useCallback(() => {
    if (experimentsLoadingMore || !experimentsHasMore) return;
    const currentCount = Array.isArray(tabData.experiments) ? tabData.experiments.length : 0;
    fetchTabFreshData('experiments', { append: true, offset: currentCount });
  }, [experimentsHasMore, experimentsLoadingMore, fetchTabFreshData, tabData.experiments]);

  const handleExperimentPageSizeChange = useCallback((nextSize) => {
    const parsed = Number(nextSize);
    if (!Number.isFinite(parsed) || parsed <= 0) return;
    setExperimentsPageSize(parsed);
    setExperimentsHasMore(true);
  }, []);

  // Compute per-tab delta indicators from latest vs previous experiment
  const tabDeltas = useMemo(() => {
    const d = data?.deltas;
    if (!d) return {};
    const deltas = {};
    if (d.programs !== 0) deltas.experiments = { text: d.programs > 0 ? `+${d.programs}` : `${d.programs}`, positive: d.programs > 0 };
    if (d.stage1 !== 0) {
      const entry = { text: d.stage1 > 0 ? `+${d.stage1} S1` : `${d.stage1} S1`, positive: d.stage1 > 0 };
      deltas.discoveries = entry;
    }
    if (d.best_loss != null && d.best_loss !== 0) {
      const sign = d.best_loss < 0 ? '' : '+';
      deltas.trends = { text: `${sign}${d.best_loss.toFixed(3)} loss`, positive: d.best_loss < 0 };
    }
    if (d.best_novelty != null && d.best_novelty !== 0) {
      const sign = d.best_novelty > 0 ? '+' : '';
      deltas.reports = { text: `${sign}${d.best_novelty.toFixed(3)} nov`, positive: d.best_novelty > 0 };
    }
    return deltas;
  }, [data?.deltas]);

  const upsertAutoRepairTask = useCallback((detail, fallbackSource = 'start') => {
    const task = detail?.task;
    if (!task || !task.task_id) {
      return false;
    }

    const nextTask = {
      ...task,
      source: detail?.source || fallbackSource,
      error: detail?.error || '',
      status: task.status || 'queued',
      updated_at: task.updated_at || Date.now() / 1000,
    };

    setAutoRepairTasks((prev) => {
      const idx = prev.findIndex((item) => item.task_id === nextTask.task_id);
      if (idx < 0) {
        return [nextTask, ...prev].slice(0, 8);
      }
      const merged = mergeAutoRepairTask(prev[idx], nextTask, fallbackSource);
      if (!merged) return prev;
      const updated = [...prev];
      updated[idx] = merged;
      return updated;
    });
    return true;
  }, []);

  useEffect(() => {
    const onAutoRepairStarted = (event) => {
      const detail = event?.detail || {};
      upsertAutoRepairTask(detail, detail?.source || 'event');
    };

    window.addEventListener('aria-auto-repair-started', onAutoRepairStarted);
    return () => {
      window.removeEventListener('aria-auto-repair-started', onAutoRepairStarted);
    };
  }, [upsertAutoRepairTask]);

  useEffect(() => {
    const activeTaskIds = autoRepairTasks
      .filter((task) => !isTerminalAgentStatus(task?.status))
      .map((task) => task.task_id)
      .filter(Boolean);

    if (!activeTaskIds.length) return undefined;

    const interval = setInterval(async () => {
      await Promise.all(activeTaskIds.map(async (taskId) => {
        try {
          const res = await apiCall(`/api/aria/agent/status/${encodeURIComponent(taskId)}`);
          const payload = await res.json();
          if (!res.ok || !payload?.task) return;

          const task = payload.task;
          setAutoRepairTasks((prev) => {
            const idx = prev.findIndex((item) => item.task_id === taskId);
            if (idx < 0) return prev;
            const merged = mergeAutoRepairTask(prev[idx], task, prev[idx]?.source || 'status_poll');
            if (!merged) return prev;
            const updated = [...prev];
            updated[idx] = merged;
            return updated;
          });
        } catch {
          // Ignore transient polling failures.
        }
      }));
    }, 2500);

    return () => clearInterval(interval);
  }, [autoRepairTasks]);

  const activeAutoRepairTasks = useMemo(
    () => autoRepairTasks.filter((task) => !isTerminalAgentStatus(task?.status)),
    [autoRepairTasks],
  );

  const completedAutoRepairCount = useMemo(
    () => autoRepairTasks.filter((task) => isTerminalAgentStatus(task?.status)).length,
    [autoRepairTasks],
  );

  const visibleAutoRepairTasks = useMemo(() => {
    if (showCompletedAutoRepairTasks) {
      return autoRepairTasks;
    }
    return activeAutoRepairTasks;
  }, [autoRepairTasks, activeAutoRepairTasks, showCompletedAutoRepairTasks]);


  const handleResetAutoRepairStripPreferences = useCallback(() => {
    setShowCompletedAutoRepairTasks(false);
    try {
      window.localStorage.removeItem(AUTO_REPAIR_SHOW_COMPLETED_KEY);
    } catch {
      // Ignore localStorage failures.
    }
  }, []);

  const emitAutoRepairStarted = useCallback((payload, source = 'start') => {
    const task = payload?.auto_repair_task;
    if (!payload?.auto_repair_started || !task || !task.task_id) {
      return false;
    }
    upsertAutoRepairTask({
      source,
      task,
      error: payload?.error || '',
    }, source);
    try {
      window.dispatchEvent(new CustomEvent('aria-auto-repair-started', {
        detail: {
          source,
          task,
          error: payload?.error || '',
        },
      }));
    } catch {
      // ignore UI event dispatch issues
    }
    return true;
  }, [upsertAutoRepairTask]);

  const summarizePreflightBlock = useCallback((err, fallbackMessage) => {
    const preflight = err?.preflight || {};
    const verdict = String(preflight?.verdict || '').toUpperCase();
    const checks = Array.isArray(preflight?.checks) ? preflight.checks : [];
    const failingCheck = checks.find((c) => c?.status === 'fail') || checks.find((c) => c?.status === 'warn');
    const detail = failingCheck?.message || failingCheck?.name || '';
    return [err?.error || fallbackMessage, verdict ? `(${verdict})` : '', detail].filter(Boolean).join(' ');
  }, []);

  const handleStartExperiment = async (config) => {
    try {
      const res = await apiCall(`/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (!res.ok) {
        const err = await res.json();
        const startedRepair = emitAutoRepairStarted(err, 'start_experiment');
        if (startedRepair) {
          const taskId = String(err?.auto_repair_task?.task_id || '').slice(0, 12);
          setActionError(`${err.error || 'Failed to start experiment'} — auto-repair started (${taskId}).`);
        } else if (err?.preflight_blocked) {
          setActionError(summarizePreflightBlock(err, 'Preflight gate blocked launch.'));
          setBlockedConfig(config);
        } else {
          setActionError(err.error || 'Failed to start experiment');
        }
        return { ok: false, ...err };
      }
      setActionError(null);
      setBlockedConfig(null);
      fetchDashboard();
      if (refreshSharedData) refreshSharedData();
      return { ok: true };
    } catch (err) {
      setActionError('Failed to start experiment: ' + err.message);
      return { ok: false, error: err.message };
    }
  };

  const handleForceStart = () => {
    if (blockedConfig) {
      handleStartExperiment({ ...blockedConfig, preflight_override: true });
    }
  };

  const handleStopExperiment = async () => {
    try {
      const res = await apiCall(`/api/experiments/stop`, {
        method: 'POST',
      });
      if (!res.ok) {
        const err = await res.json();
        setActionError(err.error || 'Failed to stop experiment');
        return;
      }
      setActionError(null);
      setAutonomousMode(false);
      fetchDashboard();
      if (refreshSharedData) refreshSharedData();
    } catch (err) {
      setActionError('Failed to stop: ' + err.message);
    }
  };

  const handleRerunExperiment = async (experimentId) => {
    if (!experimentId) {
      setActionError('No recent experiment available to restart');
      return;
    }
    try {
      const res = await apiCall(`/api/experiments/${experimentId}/rerun`, {
        method: 'POST',
      });
      if (!res.ok) {
        const err = await res.json();
        setActionError(err.error || 'Failed to restart experiment');
        return;
      }
      setActionError(null);
      fetchDashboard();
    } catch (err) {
      setActionError('Failed to restart experiment: ' + err.message);
    }
  };

  const handleFillGapsExperiment = async (experimentId) => {
    if (!experimentId) {
      setActionError('No experiment selected for gap fill');
      return;
    }
    try {
      const res = await apiCall(`/api/experiments/${experimentId}/fill-gaps`, {
        method: 'POST',
      });
      if (!res.ok) {
        const err = await res.json();
        setActionError(err.error || 'Failed to fill metric gaps');
        return;
      }
      setActionError(null);
      fetchDashboard();
      if (refreshSharedData) refreshSharedData();
    } catch (err) {
      setActionError('Failed to fill gaps: ' + err.message);
    }
  };

  const handleStartAutonomous = useCallback(async (config) => {
    const payload = {
      mode: 'continuous',
      model_source: 'mixed',
      source: 'action_queue',
      auto_harden: true,
      preflight_override: true,
      enforce_preflight: true,
      ...(config || {}),
    };
    try {
      const res = await apiCall(`/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json();
        const startedRepair = emitAutoRepairStarted(err, 'start_autonomous');
        if (startedRepair) {
          const taskId = String(err?.auto_repair_task?.task_id || '').slice(0, 12);
          setActionError(`${err.error || 'Failed to start autonomous mode'} — auto-repair started (${taskId}).`);
        } else if (err?.preflight_blocked) {
          setActionError(summarizePreflightBlock(err, 'Preflight gate blocked launch.'));
        } else {
          setActionError(err.error || 'Failed to start autonomous mode');
        }
        return;
      }
      setActionError(null);
      setAutonomousMode(true);
      fetchDashboard();
    } catch (err) {
      setActionError('Failed to start autonomous mode: ' + err.message);
    }
  }, [emitAutoRepairStarted, summarizePreflightBlock]);

  const handleStopAutonomous = async () => {
    try {
      setCycleControlBusy(true);
      const res = await apiCall(`/api/aria/cycle-control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'pause' }),
      });
      const payload = await res.json();
      if (!res.ok || payload?.error) {
        throw new Error(payload?.error || 'Failed to pause autonomous cycle');
      }
      if (payload?.cycle) {
        setAriaCycle(payload.cycle);
      }
      setAutonomousMode(false);
      setActionError(null);
      fetchDashboard();
    } catch (err) {
      setActionError(`Failed to stop autonomous loop: ${err.message}`);
    } finally {
      setCycleControlBusy(false);
    }
  };

  const handleCycleControl = async (action) => {
    if (!action || cycleControlBusy) return;
    setCycleControlBusy(true);
    try {
      const res = await apiCall(`/api/aria/cycle-control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
      const payload = await res.json();
      if (!res.ok || payload?.error) {
        throw new Error(payload?.error || `Failed to ${action} cycle`);
      }
      if (payload?.cycle) {
        setAriaCycle(payload.cycle);
      }
      if (action === 'start') {
        setAutonomousMode(true);
      }
      if (action === 'pause') {
        setAutonomousMode(false);
      }
      setActionError(null);
      fetchDashboard();
    } catch (err) {
      setActionError(`Cycle control failed: ${err.message}`);
    } finally {
      setCycleControlBusy(false);
    }
  };

  const handleSelectExperiment = (expId) => {
    setSelectedExperiment(expId);
    setActiveTab('experiment-detail');
  };

  const handleBackFromExperiment = () => {
    setSelectedExperiment(null);
    setActiveTab('experiments');
  };

  const handleSelectProgram = (resultId) => {
    if (resultId === '_CANDIDATES_TAB_') {
      setActiveTab('experiments');
      return;
    }
    if (resultId === '_QUALIFIED_TAB_') {
      setActiveTab('discoveries');
      return;
    }
    setSelectedProgram(resultId);
  };

  const filterEligibleResultIds = useCallback((mode, resultIds) => {
    const ids = Array.isArray(resultIds) ? resultIds.filter(Boolean) : [];
    if (!ids.length) {
      return {
        ok: false,
        eligibleIds: [],
        message: `No result ids provided for ${mode} action.`,
      };
    }
    const eligibilityKey = mode === 'validation' ? 'validationEligible' : 'investigationEligible';
    const eligibleIds = ids.filter(resultId => eligibilityByResultId[resultId]?.[eligibilityKey]);
    const ineligibleIds = ids.filter(resultId => !eligibilityByResultId[resultId]?.[eligibilityKey]);
    if (!eligibleIds.length) {
      const label = ineligibleIds.slice(0, 3).join(', ') || 'unknown';
      return {
        ok: false,
        eligibleIds: [],
        message: `No eligible ${mode} candidates found. Ineligible: ${label}.`,
      };
    }
    if (ineligibleIds.length > 0) {
      const label = ineligibleIds.slice(0, 3).join(', ');
      return {
        ok: true,
        eligibleIds,
        message: `Skipping ${ineligibleIds.length} ineligible ${mode} candidate${ineligibleIds.length === 1 ? '' : 's'} (${label}).`,
      };
    }
    return { ok: true, eligibleIds, message: null };
  }, [eligibilityByResultId]);

  const handleInvestigate = async (resultIds) => {
    const eligibility = filterEligibleResultIds('investigation', resultIds);
    const rawIds = Array.isArray(resultIds) ? resultIds.filter(Boolean) : [];
    const hasIneligible = rawIds.length > (eligibility.eligibleIds || []).length;
    const shouldForceAll = overrideIneligibleAlways && rawIds.length > 0;
    if (!eligibility.ok) {
      if (!rawIds.length) {
        setActionError(eligibility.message);
        return;
      }
      if (!shouldForceAll) {
        const confirmOverride = window.confirm(
          `${eligibility.message}\n\nForce override and start investigation anyway?`
        );
        if (!confirmOverride) {
          setActionError(eligibility.message);
          return;
        }
      }
      try {
        const res = await apiCall(`/api/experiments/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            mode: 'investigation',
            result_ids: rawIds,
            force: true,
            override_ineligible: true,
          }),
        });
        if (!res.ok) {
          const err = await res.json();
          setActionError(err.error || 'Failed to start forced investigation');
          return;
        }
        setActionError('Investigation started with override.');
        fetchDashboard();
      } catch (err) {
        setActionError('Failed to start forced investigation: ' + err.message);
      }
      return;
    }
    if (hasIneligible && rawIds.length) {
      let confirmOverride = false;
      if (shouldForceAll) {
        confirmOverride = true;
      } else {
        confirmOverride = window.confirm(
          `${eligibility.message}\n\nForce override and include the ineligible fingerprint(s) too?`
        );
      }
      if (confirmOverride) {
        try {
          const res = await apiCall(`/api/experiments/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              mode: 'investigation',
              result_ids: rawIds,
              force: true,
              override_ineligible: true,
            }),
          });
          if (!res.ok) {
            const err = await res.json();
            setActionError(err.error || 'Failed to start forced investigation');
            return;
          }
          setActionError('Investigation started with override.');
          fetchDashboard();
        } catch (err) {
          setActionError('Failed to start forced investigation: ' + err.message);
        }
        return;
      }
    }
    try {
      const res = await apiCall(`/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'investigation', result_ids: eligibility.eligibleIds }),
      });
      if (!res.ok) {
        const err = await res.json();
        const startedRepair = emitAutoRepairStarted(err, 'start_investigation');
        if (startedRepair) {
          const taskId = String(err?.auto_repair_task?.task_id || '').slice(0, 12);
          setActionError(`${err.error || 'Failed to start investigation'} — auto-repair started (${taskId}).`);
        } else {
          setActionError(err.error || 'Failed to start investigation');
        }
        return;
      }
      setActionError(eligibility.message || null);
      fetchDashboard();
    } catch (err) {
      setActionError('Failed to start investigation: ' + err.message);
    }
  };

  const handleValidate = async (resultIds) => {
    const eligibility = filterEligibleResultIds('validation', resultIds);
    const rawIds = Array.isArray(resultIds) ? resultIds.filter(Boolean) : [];
    const hasIneligible = rawIds.length > (eligibility.eligibleIds || []).length;
    const shouldForceAll = overrideIneligibleAlways && rawIds.length > 0;
    if (!eligibility.ok) {
      if (!rawIds.length) {
        setActionError(eligibility.message);
        return;
      }
      if (!shouldForceAll) {
        const confirmOverride = window.confirm(
          `${eligibility.message}\n\nForce override and start validation anyway?`
        );
        if (!confirmOverride) {
          setActionError(eligibility.message);
          return;
        }
      }
      try {
        const res = await apiCall(`/api/experiments/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            mode: 'validation',
            result_ids: rawIds,
            force: true,
            override_ineligible: true,
          }),
        });
        if (!res.ok) {
          const err = await res.json();
          setActionError(err.error || 'Failed to start forced validation');
          return;
        }
        setActionError('Validation started with override.');
        fetchDashboard();
      } catch (err) {
        setActionError('Failed to start forced validation: ' + err.message);
      }
      return;
    }
    if (hasIneligible && rawIds.length) {
      let confirmOverride = false;
      if (shouldForceAll) {
        confirmOverride = true;
      } else {
        confirmOverride = window.confirm(
          `${eligibility.message}\n\nForce override and include the ineligible fingerprint(s) too?`
        );
      }
      if (confirmOverride) {
        try {
          const res = await apiCall(`/api/experiments/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              mode: 'validation',
              result_ids: rawIds,
              force: true,
              override_ineligible: true,
            }),
          });
          if (!res.ok) {
            const err = await res.json();
            setActionError(err.error || 'Failed to start forced validation');
            return;
          }
          setActionError('Validation started with override.');
          fetchDashboard();
        } catch (err) {
          setActionError('Failed to start forced validation: ' + err.message);
        }
        return;
      }
    }
    try {
      const res = await apiCall(`/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'validation', result_ids: eligibility.eligibleIds }),
      });
      if (!res.ok) {
        const err = await res.json();
        const startedRepair = emitAutoRepairStarted(err, 'start_validation');
        if (startedRepair) {
          const taskId = String(err?.auto_repair_task?.task_id || '').slice(0, 12);
          setActionError(`${err.error || 'Failed to start validation'} — auto-repair started (${taskId}).`);
        } else {
          setActionError(err.error || 'Failed to start validation');
        }
        return;
      }
      setActionError(eligibility.message || null);
      fetchDashboard();
    } catch (err) {
      setActionError('Failed to start validation: ' + err.message);
    }
  };


  const handleRunProductionTemplate = async (template) => {
    const payload = template?.start_payload;
    if (!payload || typeof payload !== 'object') {
      setActionError('Invalid production template payload');
      return;
    }
    const templateMode = payload?.mode;
    let nextPayload = payload;
    let eligibilityMessage = null;
    if (templateMode === 'investigation' || templateMode === 'validation') {
      const rawResultIds = Array.isArray(payload.result_ids)
        ? payload.result_ids
        : payload.result_id
          ? [payload.result_id]
          : [];
      const eligibility = filterEligibleResultIds(templateMode, rawResultIds);
      if (!eligibility.ok) {
        setActionError(eligibility.message);
        return;
      }
      const { result_id, ...rest } = payload;
      nextPayload = { ...rest, result_ids: eligibility.eligibleIds };
      eligibilityMessage = eligibility.message || null;
    }
    try {
      const res = await apiCall(`/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(nextPayload),
      });
      if (!res.ok) {
        const err = await res.json();
        const startedRepair = emitAutoRepairStarted(err, 'run_production_template');
        if (startedRepair) {
          const taskId = String(err?.auto_repair_task?.task_id || '').slice(0, 12);
          setActionError(`${err.error || 'Failed to run production template'} — auto-repair started (${taskId}).`);
        } else {
          setActionError(err.error || 'Failed to run production template');
        }
        return;
      }
      setActionError(eligibilityMessage);
      setActiveTab('experiments');
      fetchDashboard();
    } catch (err) {
      setActionError('Failed to run production template: ' + err.message);
    }
  };

  const handleActionComplete = () => {
    fetchDashboard();
  };

  const handleQueueAdd = useCallback((candidate) => {
    const resultId = candidate?.resultId;
    if (!resultId) return;
    const eligibility = eligibilityByResultId[resultId] || null;
    if (candidate?.queueEligible === false && !eligibility?.queueEligible) {
      setActionError(queueReasonLabel(candidate?.queueReason));
      return;
    }
    const intent = resolveQueueIntent(candidate, eligibility);
    if (!intent) {
      setActionError(queueReasonLabel(candidate?.queueReason));
      return;
    }
    setInvestigationQueue(prev => normalizeQueue([
      ...prev.filter(item => item.resultId !== resultId),
      {
        resultId,
        fingerprint: candidate?.fingerprint || null,
        source: candidate?.source || 'unknown',
        architectureFamily: candidate?.architectureFamily || null,
        intent,
      },
    ]));
  }, [eligibilityByResultId]);

  const handleQueueRemove = useCallback((resultId) => {
    setInvestigationQueue(prev => prev.filter(item => item.resultId !== resultId));
  }, []);

  const handleQueueClear = useCallback(() => {
    setInvestigationQueue([]);
  }, []);

  useEffect(() => {
    setInvestigationQueue(prev => {
      let changed = false;
      const next = [];
      for (const item of prev) {
        const intent = item?.intent === 'validation' ? 'validation' : 'investigation';
        const eligibility = eligibilityByResultId[item.resultId];
        if (!eligibility) {
          if (item.intent !== intent) {
            changed = true;
            next.push({ ...item, intent });
          } else {
            next.push(item);
          }
          continue;
        }
        const stillEligibleForIntent = intent === 'validation'
          ? eligibility.validationEligible
          : eligibility.investigationEligible;
        if (!stillEligibleForIntent) {
          changed = true;
          continue;
        }
        if (item.intent !== intent) {
          changed = true;
          next.push({ ...item, intent });
        } else {
          next.push(item);
        }
      }
      return changed ? next : prev;
    });
  }, [eligibilityByResultId]);

  const handleQueueInvestigate = useCallback(() => {
    if (!investigationQueue.length) return;
    const queuedIds = investigationQueue
      .filter(item => item.intent === 'investigation')
      .map(item => item.resultId);
    const eligibleIds = queuedIds
      .filter(resultId => eligibilityByResultId[resultId]?.investigationEligible);
    if (!eligibleIds.length && !overrideIneligibleAlways) {
      setActionError('No queued investigation candidates are currently eligible.');
      return;
    }
    handleInvestigate(overrideIneligibleAlways ? queuedIds : eligibleIds);
  }, [investigationQueue, eligibilityByResultId, overrideIneligibleAlways, handleInvestigate]);

  const handleQueueValidate = useCallback(() => {
    if (!investigationQueue.length) return;
    const queuedIds = investigationQueue
      .filter(item => item.intent === 'validation')
      .map(item => item.resultId);
    const eligibleIds = queuedIds
      .filter(resultId => eligibilityByResultId[resultId]?.validationEligible);
    if (!eligibleIds.length && !overrideIneligibleAlways) {
      setActionError('No queued validation candidates are currently eligible.');
      return;
    }
    handleValidate(overrideIneligibleAlways ? queuedIds : eligibleIds);
  }, [investigationQueue, eligibilityByResultId, overrideIneligibleAlways, handleValidate]);

  const queueBreakdown = useMemo(() => {
    return investigationQueue.reduce((acc, item) => {
      if (item.intent === 'validation') {
        acc.validation += 1;
      } else {
        acc.investigation += 1;
      }
      return acc;
    }, { investigation: 0, validation: 0 });
  }, [investigationQueue]);

  const handleViewInLeaderboard = (resultId) => {
    setLeaderboardHighlight(resultId);
    setActiveTab('discoveries');
  };

  const handleSelectCampaign = (campaignId) => {
    setSelectedCampaignId(campaignId);
    setActiveTab('reports');
  };

  const handleHypothesisHandoff = useCallback((handoff) => {
    const suggestedMode = ['single', 'investigation', 'validation'].includes(handoff?.suggestedMode)
      ? handoff.suggestedMode
      : 'single';
    setControlPanelPrefill({
      source: handoff?.source || 'campaign',
      campaignId: handoff?.campaignId || null,
      campaignTitle: handoff?.campaignTitle || null,
      objective: handoff?.objective || null,
      hypothesis: handoff?.hypothesis || null,
      suggestedMode,
      requestedAt: Date.now(),
    });
    setActiveTab('command');
  }, []);

  const handleNavigateStrategy = useCallback(() => {
    setActiveTab('overview');
    setTimeout(() => {
      const el = document.getElementById('strategy-advisor');
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 50);
  }, []);

  const handleFingerprintLookup = useCallback(async () => {
    const value = String(fingerprintLookup || '').trim();
    if (!value) {
      setFingerprintLookupError('Enter a result ID or fingerprint prefix.');
      return;
    }
    setFingerprintLookupBusy(true);
    setFingerprintLookupError(null);
    try {
      const res = await apiCall(`/api/fingerprint/resolve?value=${encodeURIComponent(value)}`);
      const payload = await res.json();
      if (!res.ok) {
        setFingerprintLookupError(payload?.error || 'Fingerprint lookup failed.');
        return;
      }
      if (payload?.result_id) {
        setFingerprintLookup('');
        setSelectedProgram(payload.result_id);
      } else {
        setFingerprintLookupError('No matching fingerprint found.');
      }
    } catch (err) {
      setFingerprintLookupError(err.message || 'Fingerprint lookup failed.');
    } finally {
      setFingerprintLookupBusy(false);
    }
  }, [fingerprintLookup]);

  const ariaMood = data?.aria?.mood || 'curious';
  const autonomousActive = Boolean(autonomousMode || ariaCycle?.continuous_active);
  const compactInsights = useMemo(() => {
    const insights = Array.isArray(data?.insights) ? data.insights : [];
    const deduped = [];
    const seen = new Set();
    for (const insight of insights) {
      const key = `${(insight?.category || '').toLowerCase()}::${(insight?.content || '').trim().toLowerCase()}`;
      if (key === '::' || seen.has(key)) continue;
      seen.add(key);
      deduped.push(insight);
      if (deduped.length >= 5) break;
    }
    return deduped;
  }, [data?.insights]);
  const productionReadiness = data?.production_readiness || null;
  const epicRecommendation = productionReadiness?.epic_switch_recommendation || null;
  const topReadinessCandidates = Array.isArray(productionReadiness?.top_candidates)
    ? productionReadiness.top_candidates
    : [];
  const reproducibilityWorkflow = productionReadiness?.reproducibility_workflow || null;
  const topFingerprintSkipReasons = useMemo(() => {
    const byReason = fingerprintDiagnostics?.by_reason;
    if (!byReason || typeof byReason !== 'object') return [];
    return Object.entries(byReason)
      .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0))
      .slice(0, 2);
  }, [fingerprintDiagnostics]);

  const strategyBlocksAdvancedStart = useMemo(() => {
    if (data?.is_running) return false;
    return Boolean(activeOverviewStrategy?.action) && !allowAdvancedStartOverride;
  }, [data?.is_running, activeOverviewStrategy, allowAdvancedStartOverride]);

  const strategyLockReason = useMemo(() => {
    if (!strategyBlocksAdvancedStart || !activeOverviewStrategy) return '';
    return `Best next step is \"${activeOverviewStrategy.title}\" (Priority #${activeOverviewStrategy.id}). Use Strategy Advisor action or intentionally override advanced setup.`;
  }, [strategyBlocksAdvancedStart, activeOverviewStrategy]);

  const primaryTab = useMemo(() => {
    for (const cat in NAV_CATEGORIES) {
      if (NAV_CATEGORIES[cat].tabs.includes(activeTab)) return cat;
    }
    return 'workbench';
  }, [activeTab]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <AriaAvatar mood={ariaMood} size={40} />
          <div>
            <h1>Dr. Aria Nexus</h1>
            <p className="subtitle">AI Research Scientist — Computational Architecture Discovery</p>
          </div>
          {data?.is_running && (
            <span className="header-running-badge">
              <span className="pulse-dot"></span>
              Running
            </span>
          )}
        </div>
        <div className="header-right">
          <div className="header-meta" aria-hidden="true">
            <span className="kbd-chip">Keys 1-7 · ? · Esc</span>
            <span className="last-updated-chip">
              {dashboardUpdatedAt ? `Updated ${new Date(dashboardUpdatedAt).toLocaleTimeString()}` : 'Loading...'}
            </span>
          </div>
          <button
            className="refresh-btn"
            style={{ fontSize: 14, padding: '3px 8px', fontWeight: 700, lineHeight: 1, minWidth: 28 }}
            onClick={() => setShowChat(c => !c)}
            aria-label="Toggle chat"
            aria-pressed={showChat}
            title="Aria Chat"
          >
            &#x1F4AC;
          </button>
          <button
            className="refresh-btn"
            style={{ fontSize: 14, padding: '3px 8px', fontWeight: 700, lineHeight: 1, minWidth: 28 }}
            onClick={() => setShowSettings(s => !s)}
            aria-label="Toggle settings"
            aria-pressed={showSettings}
            title="Settings"
          >
            &#x2699;
          </button>
          <button
            className="refresh-btn"
            style={{ fontSize: 14, padding: '3px 8px', fontWeight: 700, lineHeight: 1, minWidth: 28 }}
            onClick={() => setShowHelp(h => !h)}
            aria-label="Toggle help"
            aria-pressed={showHelp}
            title="Help (press ? key)"
          >
            ?
          </button>
          <label className="auto-refresh">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh
          </label>
          <button className="refresh-btn" onClick={fetchDashboard}>Refresh</button>
          <button
            className="refresh-btn"
            onClick={openDesignerBlank}
            title="Open Aria Designer with a blank canvas"
          >
            Designer
          </button>
        </div>
      </header>

      <nav className="tab-nav-hierarchical">
        <div className="primary-nav">
          {Object.entries(NAV_CATEGORIES).map(([id, cat]) => (
            <button
              key={id}
              className={`primary-tab ${primaryTab === id ? 'active' : ''}`}
              onClick={() => {
                setActiveTab(cat.tabs[0]);
                setSelectedExperiment(null);
              }}
            >
              {cat.label}
            </button>
          ))}
        </div>
        <div className="secondary-nav">
          {NAV_CATEGORIES[primaryTab]?.tabs.map(tab => (
            <button
              key={tab}
              className={`tab ${activeTab === tab ? 'active' : ''}`}
              title={TAB_TIPS[tab]}
              onClick={() => {
                setActiveTab(tab);
                setSelectedExperiment(null);
              }}
            >
              {TAB_LABELS[tab]}
              {tabDeltas[tab] && (
                <span style={{
                  marginLeft: 4, fontSize: 9, fontWeight: 600, padding: '1px 4px',
                  borderRadius: 3,
                  background: tabDeltas[tab].positive ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                  color: tabDeltas[tab].positive ? 'var(--accent-green)' : 'var(--accent-red)',
                  whiteSpace: 'nowrap',
                }}>
                  {tabDeltas[tab].text}
                </span>
              )}
            </button>
          ))}
        </div>
      </nav>

      <main className="app-main">
        {initialLoading && !error && (
          <div className="ux-state ux-state-loading" style={{ justifyContent: 'center', marginBottom: 14 }}>
            <span className="ux-spinner" />
            <div className="ux-stack">
              <span className="ux-state-title">Loading dashboard</span>
              <span className="ux-state-subtle">Fetching latest research, insights, and run status.</span>
            </div>
          </div>
        )}
        {error && (
          <div className="error-banner">
            Unable to connect to API: {error}
            <br />
            <small>Start the server: python -m research --mode=dashboard</small>
          </div>
        )}

        {actionError && (
          <div className="error-banner" style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }} onClick={() => setActionError(null)}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span>{actionError}</span>
              {blockedConfig && (
                <button
                  className="refresh-btn"
                  style={{
                    fontSize: 11,
                    padding: '2px 8px',
                    borderColor: 'var(--accent-red)',
                    color: 'var(--accent-red)',
                    background: 'rgba(248, 81, 73, 0.1)',
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleForceStart();
                  }}
                >
                  Force Start
                </button>
              )}
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); setActionError(null); }}
              aria-label="Dismiss error"
              style={{
                background: 'none', border: 'none', color: 'inherit', cursor: 'pointer',
                fontSize: 16, padding: '0 4px', opacity: 0.8, flexShrink: 0,
              }}
            >&times;</button>
          </div>
        )}

        {data?.is_running && (
          <div className="card" style={{ marginBottom: 12, padding: '10px 12px', borderLeft: '3px solid var(--accent-green)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                <strong style={{ color: 'var(--accent-green)' }}>
                  {autonomousActive ? 'Autonomous active:' : 'Run active:'}
                </strong>{' '}
                {autonomousActive
                  ? `${ariaCycle?.phase_label || ariaCycle?.phase || data?.progress?.status || 'running'}`
                  : (data?.progress?.status || 'running')}
                {data?.progress?.experiment_id ? ` · ${String(data.progress.experiment_id).slice(0, 12)}` : ''}
                <span style={{ marginLeft: 8, color: 'var(--text-muted)' }}>
                  {autonomousActive
                    ? 'Aria is iterating across cycles while you browse other tabs.'
                    : 'Search continues in background while you browse other tabs.'}
                </span>
              </div>
              {activeTab !== 'command' && (
                <button className="refresh-btn" onClick={() => setActiveTab('command')}>
                  Open live view
                </button>
              )}
            </div>
          </div>
        )}

        {investigationQueue.length > 0 && (
          <div className="card" style={{ marginBottom: 12, padding: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                Progression Queue: {investigationQueue.length} candidate{investigationQueue.length === 1 ? '' : 's'} pinned
                {' '}({queueBreakdown.investigation} investigate, {queueBreakdown.validation} validate).
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <button
                  className="refresh-btn"
                  onClick={handleQueueInvestigate}
                  disabled={queueBreakdown.investigation === 0}
                >
                  Investigate Queue
                </button>
                <button
                  className="refresh-btn"
                  onClick={handleQueueValidate}
                  disabled={queueBreakdown.validation === 0}
                >
                  Validate Queue
                </button>
                <button className="refresh-btn" onClick={handleQueueClear} style={{ marginLeft: 8, color: 'var(--accent-red)', borderColor: 'var(--accent-red)' }}>Clear Queue</button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'command' && (
          <>
            {/* Zone 1: Status Bar */}
            <StatusBar
              aria={data?.aria}
              isRunning={data?.is_running}
              progress={data?.progress}
              ariaCycle={ariaCycle}
              onCycleControl={handleCycleControl}
              cycleControlBusy={cycleControlBusy}
              learningTrajectory={learningTrajectory}
              productionReadiness={productionReadiness}
            />

            {/* Zone 2: Action Queue */}
            <ActionQueue
              dashboardData={data}
              isRunning={data?.is_running}
              autonomousMode={autonomousActive}
              onStart={handleStartExperiment}
              onStop={handleStopExperiment}
              onStartAutonomous={handleStartAutonomous}
              onStopAutonomous={handleStopAutonomous}
              onStrategyChange={setActiveOverviewStrategy}
              onNavigateTab={(tab) => {
                const remap = { leaderboard: 'discoveries', learning: 'trends', report: 'reports' };
                const allowed = new Set(['command', 'experiments', 'discoveries', 'trends', 'reports']);
                const mapped = remap[tab] || tab;
                if (allowed.has(mapped)) {
                  setActiveTab(mapped);
                }
              }}
              onSelectProgram={handleSelectProgram}
            />


            {/* Zone 3: Summary + Activity */}
            <div className="overview-grid" style={{ marginTop: 24 }}>
              <div className="overview-left">
                <SummaryCards learningTrend={learningTrajectory} />
                <QuickAnalyticsPreview
                  deltas={data?.deltas}
                  learningTrajectory={learningTrajectory}
                  summary={data?.summary}
                  onOpenAnalytics={() => setActiveTab('trends')}
                />
              </div>
              <div className="overview-right card" style={{ padding: 24 }}>
                <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Discovery Frontier</div>
                <GlobalParetoChart programs={leaderboardEntries} onSelectProgram={handleSelectProgram} onNavigateTab={(tab) => setActiveTab(tab)} />
                <div style={{ marginTop: 20, borderTop: '1px solid var(--border)', paddingTop: 20 }}>
                  <LiveFeed
                    apiBase={API_BASE}
                    experimentId={data?.progress?.experiment_id || null}
                  />
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'experiments' && (
          <>
            {tabErrors.experiments && (
              <div className="error-banner" style={{ marginBottom: 12 }}>
                Fresh experiments fetch failed ({tabErrors.experiments}); showing dashboard snapshot.
              </div>
            )}
            <ExperimentList
              experiments={tabData.experiments || data?.recent_experiments}
              onSelectExperiment={handleSelectExperiment}
              onRefresh={fetchDashboard}
              onLoadMore={handleLoadMoreExperiments}
              hasMore={experimentsHasMore}
              loadingMore={experimentsLoadingMore}
              pageSize={experimentsPageSize}
              onPageSizeChange={handleExperimentPageSizeChange}
            />
          </>
        )}

        {activeTab === 'experiment-detail' && selectedExperiment && (
          <ExperimentDetail
            experimentId={selectedExperiment}
            onBack={handleBackFromExperiment}
            onSelectProgram={handleSelectProgram}
          />
        )}

        {activeTab === 'discoveries' && (
          <Discoveries
            onSelectProgram={handleSelectProgram}
            onInvestigate={handleInvestigate}
            onValidate={handleValidate}
            highlightResultId={leaderboardHighlight}
            onHighlightClear={() => setLeaderboardHighlight(null)}
            onQueueAdd={handleQueueAdd}
            onQueueRemove={handleQueueRemove}
            queuedResultIds={investigationQueue.map(item => item.resultId)}
            eligibilityByResultId={eligibilityByResultId}
            onOpenInDesigner={openDesignerForResult}
          />
        )}

        {activeTab === 'trends' && (
          <AnalyticsTab
            data={data}
            tabData={tabData}
            tabErrors={tabErrors}
            onSelectExperiment={handleSelectExperiment}
            onRerunExperiment={handleRerunExperiment}
            onFillGapsExperiment={handleFillGapsExperiment}
            onNavigateStrategy={handleNavigateStrategy}
            onStartExperiment={handleStartExperiment}
          />
        )}

        {activeTab === 'comparison' && (
          <CompareView
            comparisonList={comparisonList}
            onRemoveProgram={handleRemoveFromComparison}
            onSelectProgram={handleSelectProgram}
          />
        )}

        {activeTab === 'perf' && (
          <>
            <NativeProfilePanel />
            <PerfDashboard />
          </>
        )}

        {activeTab === 'reports' && (
          <>
            <ResearchReport
              onSelectProgram={handleSelectProgram}
              onSelectExperiment={handleSelectExperiment}
              onInvestigate={handleInvestigate}
              onValidate={handleValidate}
              onOpenInDesigner={openDesignerForResult}
              onQueueAdd={handleQueueAdd}
              onQueueRemove={handleQueueRemove}
              queuedResultIds={investigationQueue.map(item => item.resultId)}
              eligibilityByResultId={eligibilityByResultId}
              onHypothesisHandoff={handleHypothesisHandoff}
            />
            <div style={{ marginTop: 16 }}>
              <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>Campaigns</h3>
              <CampaignView
                onSelectExperiment={handleSelectExperiment}
                selectedCampaignId={selectedCampaignId}
                onCampaignIdClear={() => setSelectedCampaignId(null)}
                onHypothesisHandoff={handleHypothesisHandoff}
              />
            </div>
            <div style={{ marginTop: 16 }}>
              <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>Knowledge Base</h3>
              <KnowledgeBase onSelectExperiment={handleSelectExperiment} />
            </div>
          </>
        )}

        {activeTab === 'log' && (
          <LogTab
            entries={tabData.entries || data?.recent_entries}
            entriesError={tabErrors.entries}
            onSelectExperiment={handleSelectExperiment}
          />
        )}
      </main>

      {/* Chat drawer */}
      {showChat && (
        <div className="chat-drawer-backdrop" onClick={() => setShowChat(false)}>
          <div className="chat-drawer" onClick={e => e.stopPropagation()}>
            <div className="chat-drawer-header">
              <span>Aria Chat</span>
              <button onClick={() => setShowChat(false)} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', fontSize: 20, cursor: 'pointer', lineHeight: 1 }}>&times;</button>
            </div>
            <div style={{ flex: 1, overflow: 'auto' }}>
              <AriaChatPanel isRunning={Boolean(data?.is_running)} autonomousMode={autonomousActive} onAutonomousEnd={() => setAutonomousMode(false)} />
            </div>
          </div>
        </div>
      )}

      {/* Settings overlay */}
      {showSettings && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0,0,0,0.5)', zIndex: 1000,
          display: 'flex', justifyContent: 'center', alignItems: 'flex-start',
          paddingTop: 60, overflow: 'auto',
        }} onClick={() => setShowSettings(false)}>
          <div style={{
            background: 'var(--bg-primary)', borderRadius: 12, maxWidth: 800,
            width: '90%', maxHeight: 'calc(100vh - 120px)', overflow: 'auto',
            padding: 24, position: 'relative',
          }} onClick={e => e.stopPropagation()}>
            <button
              onClick={() => setShowSettings(false)}
              style={{
                position: 'absolute', top: 12, right: 12,
                background: 'none', border: 'none', color: 'var(--text-secondary)',
                fontSize: 20, cursor: 'pointer', lineHeight: 1,
              }}
              aria-label="Close settings"
            >&times;</button>
            <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>Experiment Settings</div>
            <label style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginBottom: 14,
              fontSize: 12,
              color: 'var(--text-secondary)',
            }}>
              <input
                type="checkbox"
                checked={overrideIneligibleAlways}
                onChange={(e) => setOverrideIneligibleAlways(Boolean(e.target.checked))}
              />
              Always allow override for ineligible fingerprints (Investigate/Validate)
            </label>
            {strategyBlocksAdvancedStart && (
              <div style={{
                marginBottom: 16,
                padding: '8px 10px',
                borderRadius: 6,
                border: '1px solid var(--accent-yellow)',
                background: 'rgba(210, 153, 34, 0.12)',
                fontSize: 12,
                color: 'var(--text-secondary)',
                lineHeight: 1.5,
              }}>
                <div style={{ marginBottom: 6 }}>
                  {strategyLockReason}
                </div>
                <button
                  className="refresh-btn"
                  style={{ fontSize: 11, padding: '3px 8px' }}
                  onClick={() => setAllowAdvancedStartOverride(true)}
                >
                  Use advanced setup anyway
                </button>
              </div>
            )}
            <ControlPanel
              isRunning={data?.is_running}
              progress={data?.progress}
              onStart={handleStartExperiment}
              onStop={handleStopExperiment}
              onRestart={() => handleRerunExperiment(data?.recent_experiments?.[0]?.experiment_id)}
              restartExperimentId={data?.recent_experiments?.[0]?.experiment_id}
              onRefresh={fetchDashboard}
              autoRecommendation={data?.last_recommendation}
              prefillRequest={controlPanelPrefill}
              onPrefillApplied={() => setControlPanelPrefill(null)}
              startLocked={strategyBlocksAdvancedStart}
              startLockReason={strategyLockReason}
            />
          </div>
        </div>
      )}

      {/* Help overlay */}
      {showHelp && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0,0,0,0.5)', zIndex: 1000,
          display: 'flex', justifyContent: 'center', alignItems: 'flex-start',
          paddingTop: 60, overflow: 'auto',
        }} onClick={() => setShowHelp(false)}>
          <div style={{
            background: 'var(--bg-primary)', borderRadius: 12, maxWidth: 720,
            width: '90%', maxHeight: 'calc(100vh - 120px)', overflow: 'auto',
            padding: 24, position: 'relative',
          }} onClick={e => e.stopPropagation()}>
            <button
              onClick={() => setShowHelp(false)}
              style={{
                position: 'absolute', top: 12, right: 12,
                background: 'none', border: 'none', color: 'var(--text-secondary)',
                fontSize: 20, cursor: 'pointer', lineHeight: 1,
              }}
              aria-label="Close help"
            >&times;</button>
            <HelpPanel />
          </div>
        </div>
      )}

      {/* Program Detail Modal (global) */}
      {selectedProgram && (
        <ProgramDetail
          resultId={selectedProgram}
          onClose={() => setSelectedProgram(null)}
          onActionComplete={handleActionComplete}
          onSelectExperiment={handleSelectExperiment}
          onViewInLeaderboard={handleViewInLeaderboard}
          onSelectCampaign={handleSelectCampaign}
          onOpenInDesigner={openDesignerForResult}
          onAddToComparison={handleAddToComparison}
          eligibilityByResultId={eligibilityByResultId}
          defaultOverrideIneligible={overrideIneligibleAlways}
        />
      )}

      {/* Architecture Designer Drawer */}
      {designerSession.open && (
        <ArchitectureDrawer
          resultId={designerSession.resultId}
          onClose={closeDesigner}
        />
      )}

      <footer className="app-footer">
        <span>HYDRA Architecture Explorer — Program Synthesis Engine</span>
        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
          Keys: 1-7 tabs · ? help · Esc close
        </span>
      </footer>
    </div>
  );
}

export default App;
