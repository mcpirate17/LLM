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
import TrendCharts from './components/TrendCharts';
import LearningPanel from './components/LearningPanel';
import CycleTimeline from './components/CycleTimeline';
import ResearchReport from './components/ResearchReport';
import HelpPanel from './components/HelpPanel';
import Leaderboard from './components/Leaderboard';
import CampaignView from './components/CampaignView';
import KnowledgeBase from './components/KnowledgeBase';
import StrategyAdvisor from './components/StrategyAdvisor';
import AriaChatPanel from './components/AriaChatPanel';
import { EventBusProvider } from './hooks/useEventBus';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || '';
const INVESTIGATION_QUEUE_KEY = 'aria_investigation_queue_v1';
const DISPLAY_MODE_KEY = 'aria_display_mode_v1';
const AUTO_REPAIR_SHOW_COMPLETED_KEY = 'aria_auto_repair_show_completed_v1';

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
  const hasInvestigationEvidence = entry.investigation_loss_ratio != null;
  const hasValidationEvidence = entry.validation_loss_ratio != null || Boolean(entry.validation_passed);

  const investigationEligible = tier === 'screening' && !hasInvestigationEvidence;
  const validationEligible = tier === 'investigation' && Boolean(entry.investigation_passed) && !hasValidationEvidence;

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

function App() {
  const TAB_LABELS = {
    overview: 'Overview',
    experiments: 'Runs',
    programs: 'Candidates (All)',
    leaderboard: 'Decisions (Curated)',
    campaigns: 'Campaigns',
    knowledge: 'Knowledge',
    trends: 'Trends',
    learning: 'Learning',
    cycles: 'Cycle Timeline',
    notebook: 'Notebook',
    insights: 'Insights',
    report: 'Report',
    help: 'Help',
  };
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [overviewActivityTab, setOverviewActivityTab] = useState('recent');
  const [displayMode, setDisplayMode] = useState(() => {
    try {
      const stored = window.localStorage.getItem(DISPLAY_MODE_KEY);
      return stored === 'expert' ? 'expert' : 'novice';
    } catch {
      return 'novice';
    }
  });
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

  // Drill-down state
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [selectedProgram, setSelectedProgram] = useState(null);

  // Action error state (replaces alert())
  const [actionError, setActionError] = useState(null);
  const [autoRepairTasks, setAutoRepairTasks] = useState([]);
  const [showCompletedAutoRepairTasks, setShowCompletedAutoRepairTasks] = useState(() => {
    try {
      return window.localStorage.getItem(AUTO_REPAIR_SHOW_COMPLETED_KEY) === '1';
    } catch {
      return false;
    }
  });

  // Cross-view navigation state
  const [leaderboardHighlight, setLeaderboardHighlight] = useState(null);
  const [selectedCampaignId, setSelectedCampaignId] = useState(null);
  const [controlPanelPrefill, setControlPanelPrefill] = useState(null);
  const [activeOverviewStrategy, setActiveOverviewStrategy] = useState(null);
  const [learningTrend, setLearningTrend] = useState(null);
  const [ariaCycle, setAriaCycle] = useState(null);
  const [cycleControlBusy, setCycleControlBusy] = useState(false);
  const [allowAdvancedStartOverride, setAllowAdvancedStartOverride] = useState(false);
  const [autonomousMode, setAutonomousMode] = useState(false);
  const advancedDetailsRef = useRef(null);
  const [eligibilityByResultId, setEligibilityByResultId] = useState({});
  const [investigationQueue, setInvestigationQueue] = useState(() => {
    try {
      const stored = window.localStorage.getItem(INVESTIGATION_QUEUE_KEY);
      return normalizeQueue(stored ? JSON.parse(stored) : []);
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(INVESTIGATION_QUEUE_KEY, JSON.stringify(investigationQueue));
    } catch {
      // Ignore localStorage failures.
    }
  }, [investigationQueue]);

  useEffect(() => {
    try {
      window.localStorage.setItem(DISPLAY_MODE_KEY, displayMode);
    } catch {}
  }, [displayMode]);


  useEffect(() => {
    try {
      window.localStorage.setItem(
        AUTO_REPAIR_SHOW_COMPLETED_KEY,
        showCompletedAutoRepairTasks ? '1' : '0',
      );
    } catch {}
  }, [showCompletedAutoRepairTasks]);

  const fetchDashboard = useCallback(async () => {
    try {
      const [dashRes, trendRes, eligibilityRes, cycleRes] = await Promise.all([
        fetch(`${API_BASE}/api/dashboard`),
        fetch(`${API_BASE}/api/analytics/learning-trajectory`),
        fetch(`${API_BASE}/api/leaderboard?sort=composite_score&limit=300`),
        fetch(`${API_BASE}/api/aria/cycle-status`),
      ]);
      if (!dashRes.ok) throw new Error(`HTTP ${dashRes.status}`);
      const json = await dashRes.json();
      setData(json);
      setError(null);
      if (trendRes.ok) {
        const trendJson = await trendRes.json();
        setLearningTrend(trendJson);
      }
      if (eligibilityRes.ok) {
        const eligibilityJson = await eligibilityRes.json();
        setEligibilityByResultId(buildEligibilityByResultId(eligibilityJson?.entries || []));
      }
      if (cycleRes.ok) {
        const cycleJson = await cycleRes.json();
        if (cycleJson && !cycleJson.error) {
          setAriaCycle(cycleJson);
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setInitialLoading(false);
    }
  }, []);

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

  const fetchTabFreshData = useCallback(async (tab) => {
    const endpoints = {
      experiments: '/api/experiments',
      programs: '/api/programs?n=50&sort=novelty_score',
      entries: '/api/entries?n=50',
      insights: '/api/insights',
    };
    const endpoint = endpoints[tab];
    if (!endpoint) return;

    try {
      const res = await fetch(`${API_BASE}${endpoint}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setTabData(prev => ({ ...prev, [tab]: Array.isArray(json) ? json : [] }));
      setTabErrors(prev => ({ ...prev, [tab]: null }));
    } catch (err) {
      setTabErrors(prev => ({ ...prev, [tab]: err.message }));
    }
  }, []);

  useEffect(() => {
    if (activeTab === 'experiments') {
      fetchTabFreshData('experiments');
    } else if (activeTab === 'programs') {
      fetchTabFreshData('programs');
    } else if (activeTab === 'notebook') {
      fetchTabFreshData('entries');
    } else if (activeTab === 'insights') {
      fetchTabFreshData('insights');
    }
  }, [activeTab, fetchTabFreshData]);

  // Compute per-tab delta indicators from latest vs previous experiment
  const tabDeltas = useMemo(() => {
    const d = data?.deltas;
    if (!d) return {};
    const deltas = {};
    if (d.programs !== 0) deltas.experiments = { text: d.programs > 0 ? `+${d.programs}` : `${d.programs}`, positive: d.programs > 0 };
    if (d.stage1 !== 0) {
      const entry = { text: d.stage1 > 0 ? `+${d.stage1} S1` : `${d.stage1} S1`, positive: d.stage1 > 0 };
      deltas.programs = entry;
      deltas.leaderboard = entry;
    }
    if (d.best_loss != null && d.best_loss !== 0) {
      const sign = d.best_loss < 0 ? '' : '+';
      deltas.trends = { text: `${sign}${d.best_loss.toFixed(3)} loss`, positive: d.best_loss < 0 };
    }
    if (d.best_novelty != null && d.best_novelty !== 0) {
      const sign = d.best_novelty > 0 ? '+' : '';
      deltas.report = { text: `${sign}${d.best_novelty.toFixed(3)} nov`, positive: d.best_novelty > 0 };
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
          const res = await fetch(`${API_BASE}/api/aria/agent/status/${encodeURIComponent(taskId)}`);
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

  const handleStartExperiment = async (config) => {
    try {
      const res = await fetch(`${API_BASE}/api/experiments/start`, {
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
        } else {
          setActionError(err.error || 'Failed to start experiment');
        }
        return;
      }
      setActionError(null);
      fetchDashboard();
    } catch (err) {
      setActionError('Failed to start experiment: ' + err.message);
    }
  };

  const handleStopExperiment = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/experiments/stop`, {
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
      const res = await fetch(`${API_BASE}/api/experiments/${experimentId}/rerun`, {
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

  const handleStartAutonomous = async (config) => {
    try {
      const res = await fetch(`${API_BASE}/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (!res.ok) {
        const err = await res.json();
        const startedRepair = emitAutoRepairStarted(err, 'start_autonomous');
        if (startedRepair) {
          const taskId = String(err?.auto_repair_task?.task_id || '').slice(0, 12);
          setActionError(`${err.error || 'Failed to start autonomous mode'} — auto-repair started (${taskId}).`);
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
  };

  const handleStopAutonomous = async () => {
    setAutonomousMode(false);
    await handleStopExperiment();
  };

  const handleCycleControl = async (action) => {
    if (!action || cycleControlBusy) return;
    setCycleControlBusy(true);
    try {
      const res = await fetch(`${API_BASE}/api/aria/cycle-control`, {
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
    setSelectedProgram(resultId);
  };

  const handleInvestigate = async (resultIds) => {
    try {
      const res = await fetch(`${API_BASE}/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'investigation', result_ids: resultIds }),
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
      setActionError(null);
      fetchDashboard();
    } catch (err) {
      setActionError('Failed to start investigation: ' + err.message);
    }
  };

  const handleValidate = async (resultIds) => {
    try {
      const res = await fetch(`${API_BASE}/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'validation', result_ids: resultIds }),
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
      setActionError(null);
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
    try {
      const res = await fetch(`${API_BASE}/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
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
      setActionError(null);
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
    const eligibleIds = investigationQueue
      .filter(item => item.intent === 'investigation')
      .map(item => item.resultId)
      .filter(resultId => eligibilityByResultId[resultId]?.investigationEligible);
    if (!eligibleIds.length) {
      setActionError('No queued investigation candidates are currently eligible.');
      return;
    }
    handleInvestigate(eligibleIds);
  }, [investigationQueue, eligibilityByResultId]);

  const handleQueueValidate = useCallback(() => {
    if (!investigationQueue.length) return;
    const eligibleIds = investigationQueue
      .filter(item => item.intent === 'validation')
      .map(item => item.resultId)
      .filter(resultId => eligibilityByResultId[resultId]?.validationEligible);
    if (!eligibleIds.length) {
      setActionError('No queued validation candidates are currently eligible.');
      return;
    }
    handleValidate(eligibleIds);
  }, [investigationQueue, eligibilityByResultId]);

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
    setActiveTab('leaderboard');
  };

  const handleSelectCampaign = (campaignId) => {
    setSelectedCampaignId(campaignId);
    setActiveTab('campaigns');
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
    setActiveTab('overview');
  }, []);

  const ariaMood = data?.aria?.mood || 'curious';
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

  const strategyBlocksAdvancedStart = useMemo(() => {
    if (data?.is_running) return false;
    return Boolean(activeOverviewStrategy?.action) && !allowAdvancedStartOverride;
  }, [data?.is_running, activeOverviewStrategy, allowAdvancedStartOverride]);

  const strategyLockReason = useMemo(() => {
    if (!strategyBlocksAdvancedStart || !activeOverviewStrategy) return '';
    return `Best next step is \"${activeOverviewStrategy.title}\" (Priority #${activeOverviewStrategy.id}). Use Strategy Advisor action or intentionally override advanced setup.`;
  }, [strategyBlocksAdvancedStart, activeOverviewStrategy]);

  return (
    <EventBusProvider apiBase={API_BASE}>
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
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginRight: 8 }}>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Mode</span>
            <button
              className="refresh-btn"
              style={{ fontSize: 11, padding: '3px 8px' }}
              onClick={() => setDisplayMode('novice')}
              aria-pressed={displayMode === 'novice'}
            >
              Novice
            </button>
            <button
              className="refresh-btn"
              style={{ fontSize: 11, padding: '3px 8px' }}
              onClick={() => setDisplayMode('expert')}
              aria-pressed={displayMode === 'expert'}
            >
              Expert
            </button>
          </div>
          <label className="auto-refresh">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh
          </label>
          <button className="refresh-btn" onClick={fetchDashboard}>Refresh</button>
        </div>
      </header>

      <nav className="tab-nav">
        {[
          { section: null, tabs: ['overview'] },
          { section: 'Research', tabs: ['experiments', 'programs', 'leaderboard'] },
          { section: 'Analysis', tabs: ['trends', 'learning', 'cycles', 'insights', 'report'] },
          { section: 'Meta', tabs: ['campaigns', 'knowledge', 'notebook', 'help'] },
        ].map(group => (
          <React.Fragment key={group.section || 'main'}>
            {group.section && (
              <span style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, padding: '0 4px', alignSelf: 'center', userSelect: 'none' }}>
                {group.section}
              </span>
            )}
            {group.tabs.map(tab => (
              <button
                key={tab}
                className={`tab ${activeTab === tab ? 'active' : ''} ${
                  tab === 'experiment-detail' ? 'hidden' : ''
                }`}
                onClick={() => {
                  setActiveTab(tab);
                  if (tab !== 'experiment-detail') setSelectedExperiment(null);
                }}
              >
                {TAB_LABELS[tab] || (tab.charAt(0).toUpperCase() + tab.slice(1))}
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
          </React.Fragment>
        ))}
      </nav>

      <main className="app-main">
        {initialLoading && !error && (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '80px 0', color: 'var(--text-secondary)', fontSize: 14 }}>
            Loading dashboard...
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
            <span>{actionError}</span>
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
                <strong style={{ color: 'var(--accent-green)' }}>Run active:</strong>{' '}
                {data?.progress?.status || 'running'}
                {data?.progress?.experiment_id ? ` · ${String(data.progress.experiment_id).slice(0, 12)}` : ''}
                <span style={{ marginLeft: 8, color: 'var(--text-muted)' }}>
                  Search continues in background while you browse other tabs.
                </span>
              </div>
              {activeTab !== 'overview' && (
                <button className="refresh-btn" onClick={() => setActiveTab('overview')}>
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

        {activeTab === 'overview' && (
          <div className="overview-grid">
            {autoRepairTasks.length > 0 && (
              <div className="card" style={{ gridColumn: '1 / -1', marginBottom: 0, padding: 10, borderLeft: '3px solid var(--accent-purple)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <div style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                    Auto-repair tasks
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                      {activeAutoRepairTasks.length} active
                    </div>
                    <button
                      className="refresh-btn"
                      style={{ fontSize: 11, padding: '2px 8px' }}
                      onClick={handleResetAutoRepairStripPreferences}
                    >
                      Reset preferences
                    </button>
                    {completedAutoRepairCount > 0 && (
                      <button
                        className="refresh-btn"
                        style={{ fontSize: 11, padding: '2px 8px' }}
                        onClick={() => setShowCompletedAutoRepairTasks((prev) => !prev)}
                      >
                        {showCompletedAutoRepairTasks ? 'Hide completed' : `Show completed (${completedAutoRepairCount})`}
                      </button>
                    )}
                  </div>
                </div>
                <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {visibleAutoRepairTasks.length === 0 && (
                    <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                      No active auto-repair tasks right now.
                    </div>
                  )}
                  {visibleAutoRepairTasks.map((task) => {
                    const status = String(task?.status || 'queued').toLowerCase();
                    const taskId = String(task?.task_id || '').slice(0, 12);
                    const source = String(task?.source || 'start').replaceAll('_', ' ');
                    const headline = task?.summary || task?.goal || task?.error || 'Repairing startup failure';
                    return (
                      <div key={task.task_id} style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                        <strong style={{ color: 'var(--text-primary)' }}>{taskId}</strong>
                        <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--accent-purple)' }}>{status}</span>
                        <span style={{ marginLeft: 8, color: 'var(--text-muted)' }}>from {source}</span>
                        <div style={{ marginTop: 2, color: 'var(--text-secondary)' }}>{headline}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            <StrategyAdvisor
              dashboardData={data}
              isRunning={data?.is_running}
              autonomousMode={autonomousMode}
              onStart={handleStartExperiment}
              onStop={handleStopExperiment}
              onStartAutonomous={handleStartAutonomous}
              onStopAutonomous={handleStopAutonomous}
              onStrategyChange={setActiveOverviewStrategy}
              onNavigateEvidence={(tab) => {
                const allowed = new Set(['experiments', 'leaderboard', 'trends', 'learning', 'report']);
                if (allowed.has(tab)) {
                  setActiveTab(tab);
                }
              }}
              onOpenAdvancedPanel={() => {
                const el = advancedDetailsRef.current;
                if (el) { el.open = true; el.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
              }}
              onApplyStrategy={(payload) => {
                const action = payload?.action;
                if (action === 'export_breakthrough') {
                  setActiveTab('report');
                  return;
                }
                if (action === 'monitor_validation') {
                  setActiveTab('leaderboard');
                  return;
                }
                if (payload?.strategy?.action === null) {
                  setActiveTab('leaderboard');
                }
              }}
            />

            {/* Left column: Aria + Control Panel */}
            <div className="overview-left">
              <AriaStatus aria={data?.aria} isRunning={data?.is_running} progress={data?.progress} />
              {ariaCycle && (
                <div className="card" style={{ marginTop: 12, marginBottom: 0, padding: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                      Aria Continuous Activity
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--accent-purple)', fontWeight: 600 }}>
                      {ariaCycle.phase_label || ariaCycle.phase || 'Idle'}
                    </div>
                  </div>
                  <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                    {ariaCycle.last_note || ariaCycle.aria_message || 'Awaiting run.'}
                  </div>
                  <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                    <span>Cycle {ariaCycle.cycle_index || 0}</span>
                    <span>{(ariaCycle.selected_mode || ariaCycle.last_completed_mode || 'idle').charAt(0).toUpperCase() + (ariaCycle.selected_mode || ariaCycle.last_completed_mode || 'idle').slice(1).replace('_', ' ')}</span>
                    <span>{ariaCycle.continuous_active ? 'Running continuously' : 'Paused'}</span>
                  </div>
                  <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {!ariaCycle.continuous_active && (
                      <button
                        className="refresh-btn"
                        onClick={() => handleCycleControl('start')}
                        disabled={cycleControlBusy || Boolean(data?.is_running)}
                      >
                        {cycleControlBusy ? 'Working...' : 'Start Cycle'}
                      </button>
                    )}
                    {ariaCycle.continuous_active && !ariaCycle.cycle_paused && (
                      <button
                        className="refresh-btn"
                        onClick={() => handleCycleControl('pause')}
                        disabled={cycleControlBusy}
                      >
                        {cycleControlBusy ? 'Working...' : 'Pause Cycle'}
                      </button>
                    )}
                    {ariaCycle.cycle_paused && (
                      <button
                        className="refresh-btn"
                        onClick={() => handleCycleControl('resume')}
                        disabled={cycleControlBusy}
                      >
                        {cycleControlBusy ? 'Working...' : 'Resume Cycle'}
                      </button>
                    )}
                  </div>
                  {Array.isArray(ariaCycle.cycle_history) && ariaCycle.cycle_history.length > 0 && (
                    <div style={{ marginTop: 8 }}>
                      <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 4 }}>
                        Recent Cycles
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                        {ariaCycle.cycle_history.slice(-4).reverse().map((row) => (
                          <div
                            key={`${row.cycle_index}-${row.timestamp}`}
                            style={{ fontSize: 11, color: 'var(--text-secondary)' }}
                          >
                            Cycle {row.cycle_index} — {(row.mode || 'synthesis').charAt(0).toUpperCase() + (row.mode || 'synthesis').slice(1)} — {(row.delta_stage1_survivors ?? 0) === 0 ? '0 new survivors' : `${row.delta_stage1_survivors > 0 ? '+' : ''}${row.delta_stage1_survivors} survivor${Math.abs(row.delta_stage1_survivors) !== 1 ? 's' : ''}`}{row.timestamp ? ` — ${new Date(row.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : ''}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              <AriaChatPanel isRunning={Boolean(data?.is_running)} autonomousMode={autonomousMode} onAutonomousEnd={() => setAutonomousMode(false)} />
              <details ref={advancedDetailsRef} className="card" style={{ marginTop: 12, marginBottom: 0 }} open={Boolean(data?.is_running) || displayMode === 'expert'}>
                <summary style={{ cursor: 'pointer', fontWeight: 600, color: 'var(--text-primary)', padding: 4 }}>
                  Experiment setup (advanced)
                </summary>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 6, marginBottom: 6 }}>
                  Fine-tune run mode, training, and validation settings.
                </div>
                {strategyBlocksAdvancedStart && (
                  <div style={{
                    marginTop: 6,
                    marginBottom: 8,
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
                <div style={{ marginTop: 10 }}>
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
                    displayMode={displayMode}
                    startLocked={strategyBlocksAdvancedStart}
                    startLockReason={strategyLockReason}
                  />
                </div>
              </details>
            </div>

            {/* Right column: Summary + Activity */}
            <div className="overview-right">
              <SummaryCards summary={data?.summary} learningTrend={learningTrend} />
              {productionReadiness && (
                <div className="card" style={{ marginBottom: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, marginBottom: 6, flexWrap: 'wrap' }}>
                    <div style={{ fontSize: 13, fontWeight: 600 }}>Production Readiness</div>
                    <button className="refresh-btn" style={{ fontSize: 11, padding: '2px 8px' }} onClick={() => setActiveTab('leaderboard')}>
                      Open Leaderboard
                    </button>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                    <strong style={{ color: 'var(--text-primary)' }}>Epic switch:</strong>{' '}
                    {epicRecommendation?.action === 'switch_to_scale_up_epic' ? 'Switch to scale-up epic' : 'Stay in current epic'}
                    {epicRecommendation?.reason ? ` — ${epicRecommendation.reason}` : ''}
                  </div>
                  <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 11 }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Breakthroughs: <strong style={{ color: 'var(--text-primary)' }}>{productionReadiness?.breakthrough_count ?? 0}</strong></span>
                    <span style={{ color: 'var(--text-secondary)' }}>Decision-ready: <strong style={{ color: 'var(--text-primary)' }}>{productionReadiness?.decision_ready_count ?? 0}</strong></span>
                    <span style={{ color: 'var(--text-secondary)' }}>High confidence: <strong style={{ color: 'var(--text-primary)' }}>{productionReadiness?.high_confidence_count ?? 0}</strong></span>
                    <span style={{ color: 'var(--text-secondary)' }}>Repro ready: <strong style={{ color: 'var(--text-primary)' }}>{productionReadiness?.full_repro_packet_count ?? 0}</strong></span>
                    <span style={{ color: 'var(--text-secondary)' }}>Artifact CKA: <strong style={{ color: 'var(--text-primary)' }}>{productionReadiness?.artifact_cka_count ?? 0}</strong></span>
                  </div>

                  {reproducibilityWorkflow && (
                    <div style={{ marginTop: 8, padding: '6px 8px', borderRadius: 6, border: '1px solid var(--border)' }}>
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                        <strong style={{ color: 'var(--text-primary)' }}>Complete repro packet:</strong>{' '}
                        {reproducibilityWorkflow?.progress_label || '0/6'}
                        {typeof reproducibilityWorkflow?.remaining === 'number' ? ` (${reproducibilityWorkflow.remaining} remaining)` : ''}
                      </div>
                      {Array.isArray(reproducibilityWorkflow?.next_actions) && reproducibilityWorkflow.next_actions.length > 0 && (
                        <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 6 }}>
                          {reproducibilityWorkflow.next_actions.slice(0, 2).map((action) => (
                            <div key={`repro-next-${action?.check_id || action?.label || 'step'}`} style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                              <span>{action?.label || action?.check_id}</span>
                              {action?.start_payload && (
                                <button
                                  className="refresh-btn"
                                  style={{ marginLeft: 6, fontSize: 10, padding: '2px 6px' }}
                                  onClick={() => handleRunProductionTemplate(action)}
                                >
                                  Run step
                                </button>
                              )}
                              {action?.guidance && (
                                <div style={{ color: 'var(--text-muted)' }}>{action.guidance}</div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                  {topReadinessCandidates.length > 0 && (
                    <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {topReadinessCandidates.map((candidate) => {
                        const repro = candidate?.repro_packet || {};
                        const reproWorkflow = candidate?.reproducibility_workflow || null;
                        const resultId = String(candidate?.result_id || '').slice(0, 12);
                        const blockers = [];
                        if (Array.isArray(candidate?.decision_missing) && candidate.decision_missing.length > 0) {
                          blockers.push(...candidate.decision_missing);
                        }
                        if (Array.isArray(repro?.missing) && repro.missing.length > 0) {
                          blockers.push(...repro.missing.map((item) => `repro:${item}`));
                        }
                        const blockerText = blockers.length > 0 ? blockers.slice(0, 3).join(', ') : 'none';
                        return (
                          <div key={candidate?.result_id || resultId} style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                            <strong style={{ color: 'var(--text-primary)' }}>{resultId || 'unknown'}</strong>
                            <span style={{ marginLeft: 6 }}>confidence {candidate?.promotion_confidence_score ?? 0}%</span>
                            <span style={{ marginLeft: 6 }}>repro {repro?.ready_count ?? 0}/{repro?.total_checks ?? 0}</span>
                            {reproWorkflow?.progress_label && (
                              <span style={{ marginLeft: 6 }}>workflow {reproWorkflow.progress_label}</span>
                            )}
                            <div style={{ color: 'var(--text-muted)' }}>Blockers: {blockerText}</div>
                            {Array.isArray(candidate?.scale_up_templates) && candidate.scale_up_templates.length > 0 && (
                              <div style={{ marginTop: 4, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                                {candidate.scale_up_templates.slice(0, 2).map((template) => (
                                  <button
                                    key={`${candidate?.result_id || resultId}-${template?.template_id || template?.title || 'tpl'}`}
                                    className="refresh-btn"
                                    style={{ fontSize: 10, padding: '2px 6px' }}
                                    onClick={() => handleRunProductionTemplate(template)}
                                  >
                                    Run Scale-Up Template
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}

              <div className="card" style={{ marginBottom: 0 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                  <div style={{ fontSize: 13, fontWeight: 600 }}>Activity</div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button className="refresh-btn" onClick={() => setOverviewActivityTab('live')} aria-pressed={overviewActivityTab === 'live'}>
                      Live updates
                    </button>
                    <button className="refresh-btn" onClick={() => setOverviewActivityTab('recent')} aria-pressed={overviewActivityTab === 'recent'}>
                      Recent trends
                    </button>
                  </div>
                </div>

                {overviewActivityTab === 'live' ? (
                  data?.is_running ? (
                    <LiveFeed
                      apiBase={API_BASE}
                      experimentId={data?.progress?.experiment_id || null}
                    />
                  ) : (
                    <div style={{ fontSize: 12, color: 'var(--text-secondary)', padding: '8px 4px 12px' }}>
                      No active run right now. Switch to Recent to review latest experiment trends.
                    </div>
                  )
                ) : (
                  <MetricsChart experiments={data?.recent_experiments} />
                )}
              </div>
            </div>

            {/* Bottom row */}
            <div className="overview-bottom">
              <TopPrograms
                programs={data?.top_programs?.slice(0, 3)}
                totalCount={data?.top_programs?.length || 0}
                compact
                onSelectProgram={handleSelectProgram}
              />
              <InsightsPanel insights={compactInsights} compact />
            </div>
          </div>
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

        {activeTab === 'programs' && (
          <>
            {tabErrors.programs && (
              <div className="error-banner" style={{ marginBottom: 12 }}>
                Fresh programs fetch failed ({tabErrors.programs}); showing dashboard snapshot.
              </div>
            )}
            <TopPrograms
              programs={tabData.programs || data?.top_programs}
              onSelectProgram={handleSelectProgram}
              onQueueAdd={handleQueueAdd}
              onQueueRemove={handleQueueRemove}
              queuedResultIds={investigationQueue.map(item => item.resultId)}
              eligibilityByResultId={eligibilityByResultId}
            />
          </>
        )}

        {activeTab === 'leaderboard' && (
          <Leaderboard
            onSelectProgram={handleSelectProgram}
            onInvestigate={handleInvestigate}
            onValidate={handleValidate}
            highlightResultId={leaderboardHighlight}
            onHighlightClear={() => setLeaderboardHighlight(null)}
            onQueueAdd={handleQueueAdd}
            onQueueRemove={handleQueueRemove}
            queuedResultIds={investigationQueue.map(item => item.resultId)}
            eligibilityByResultId={eligibilityByResultId}
          />
        )}

        {activeTab === 'campaigns' && (
          <CampaignView
            onSelectExperiment={handleSelectExperiment}
            selectedCampaignId={selectedCampaignId}
            onCampaignIdClear={() => setSelectedCampaignId(null)}
            onHypothesisHandoff={handleHypothesisHandoff}
          />
        )}

        {activeTab === 'knowledge' && (
          <KnowledgeBase onSelectExperiment={handleSelectExperiment} />
        )}

        {activeTab === 'trends' && (
          <TrendCharts onSelectExperiment={handleSelectExperiment} />
        )}

        {activeTab === 'learning' && (
          <LearningPanel />
        )}

        {activeTab === 'cycles' && (
          <CycleTimeline />
        )}

        {activeTab === 'notebook' && (
          <>
            {tabErrors.entries && (
              <div className="error-banner" style={{ marginBottom: 12 }}>
                Fresh notebook fetch failed ({tabErrors.entries}); showing dashboard snapshot.
              </div>
            )}
            <LabNotebook
              entries={tabData.entries || data?.recent_entries}
              onSelectExperiment={handleSelectExperiment}
            />
          </>
        )}

        {activeTab === 'insights' && (
          <>
            {tabErrors.insights && (
              <div className="error-banner" style={{ marginBottom: 12 }}>
                Fresh insights fetch failed ({tabErrors.insights}); showing dashboard snapshot.
              </div>
            )}
            <InsightsPanel insights={tabData.insights || data?.insights} />
          </>
        )}

        {activeTab === 'report' && (
          <ResearchReport
            onSelectProgram={handleSelectProgram}
            onSelectExperiment={handleSelectExperiment}
            onInvestigate={handleInvestigate}
            onValidate={handleValidate}
            onQueueAdd={handleQueueAdd}
            onQueueRemove={handleQueueRemove}
            queuedResultIds={investigationQueue.map(item => item.resultId)}
            eligibilityByResultId={eligibilityByResultId}
            onHypothesisHandoff={handleHypothesisHandoff}
          />
        )}

        {activeTab === 'help' && (
          <HelpPanel />
        )}
      </main>

      {/* Program Detail Modal (global) */}
      {selectedProgram && (
        <ProgramDetail
          resultId={selectedProgram}
          onClose={() => setSelectedProgram(null)}
          onActionComplete={handleActionComplete}
          onSelectExperiment={handleSelectExperiment}
          onViewInLeaderboard={handleViewInLeaderboard}
          onSelectCampaign={handleSelectCampaign}
          eligibilityByResultId={eligibilityByResultId}
        />
      )}

      <footer className="app-footer">
        <span>HYDRA Architecture Explorer — Program Synthesis Engine</span>
      </footer>
    </div>
    </EventBusProvider>
  );
}

export default App;
