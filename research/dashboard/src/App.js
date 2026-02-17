import React, { useState, useEffect, useCallback, useMemo } from 'react';
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
import ResearchReport from './components/ResearchReport';
import HelpPanel from './components/HelpPanel';
import Leaderboard from './components/Leaderboard';
import CampaignView from './components/CampaignView';
import KnowledgeBase from './components/KnowledgeBase';
import StrategyAdvisor from './components/StrategyAdvisor';
import AriaChatPanel from './components/AriaChatPanel';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || '';
const INVESTIGATION_QUEUE_KEY = 'aria_investigation_queue_v1';
const DISPLAY_MODE_KEY = 'aria_display_mode_v1';

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
    notebook: 'Notebook',
    insights: 'Insights',
    report: 'Report',
    help: 'Help',
  };
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
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

  // Cross-view navigation state
  const [leaderboardHighlight, setLeaderboardHighlight] = useState(null);
  const [selectedCampaignId, setSelectedCampaignId] = useState(null);
  const [controlPanelPrefill, setControlPanelPrefill] = useState(null);
  const [activeOverviewStrategy, setActiveOverviewStrategy] = useState(null);
  const [learningTrend, setLearningTrend] = useState(null);
  const [ariaCycle, setAriaCycle] = useState(null);
  const [allowAdvancedStartOverride, setAllowAdvancedStartOverride] = useState(false);
  const [autonomousMode, setAutonomousMode] = useState(false);
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
    if (d.programs !== 0) deltas.experiments = d.programs > 0 ? `+${d.programs}` : `${d.programs}`;
    if (d.stage1 !== 0) deltas.programs = d.stage1 > 0 ? `+${d.stage1} S1` : `${d.stage1} S1`;
    if (d.stage1 !== 0) deltas.leaderboard = deltas.programs;
    if (d.best_loss != null && d.best_loss !== 0) {
      const sign = d.best_loss < 0 ? '' : '+';
      deltas.trends = `${sign}${d.best_loss.toFixed(3)} loss`;
    }
    if (d.best_novelty != null && d.best_novelty !== 0) {
      const sign = d.best_novelty > 0 ? '+' : '';
      deltas.report = `${sign}${d.best_novelty.toFixed(3)} nov`;
    }
    return deltas;
  }, [data?.deltas]);

  const handleStartExperiment = async (config) => {
    try {
      const res = await fetch(`${API_BASE}/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (!res.ok) {
        const err = await res.json();
        setActionError(err.error || 'Failed to start experiment');
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

  const handleStartAutonomous = async (config) => {
    try {
      const res = await fetch(`${API_BASE}/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (!res.ok) {
        const err = await res.json();
        setActionError(err.error || 'Failed to start autonomous mode');
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
        setActionError(err.error || 'Failed to start investigation');
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
        setActionError(err.error || 'Failed to start validation');
        return;
      }
      setActionError(null);
      fetchDashboard();
    } catch (err) {
      setActionError('Failed to start validation: ' + err.message);
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

  const strategyBlocksAdvancedStart = useMemo(() => {
    if (data?.is_running) return false;
    return Boolean(activeOverviewStrategy?.action) && !allowAdvancedStartOverride;
  }, [data?.is_running, activeOverviewStrategy, allowAdvancedStartOverride]);

  const strategyLockReason = useMemo(() => {
    if (!strategyBlocksAdvancedStart || !activeOverviewStrategy) return '';
    return `Best next step is \"${activeOverviewStrategy.title}\" (Priority #${activeOverviewStrategy.id}). Use Strategy Advisor action or intentionally override advanced setup.`;
  }, [strategyBlocksAdvancedStart, activeOverviewStrategy]);

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
          { section: 'Analysis', tabs: ['trends', 'learning', 'insights', 'report'] },
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
                    borderRadius: 3, background: 'rgba(63, 185, 80, 0.15)',
                    color: 'var(--accent-green)', whiteSpace: 'nowrap',
                  }}>
                    {tabDeltas[tab]}
                  </span>
                )}
              </button>
            ))}
          </React.Fragment>
        ))}
      </nav>

      <main className="app-main">
        {error && (
          <div className="error-banner">
            Unable to connect to API: {error}
            <br />
            <small>Start the server: python -m research --mode=dashboard</small>
          </div>
        )}

        {actionError && (
          <div className="error-banner" style={{ cursor: 'pointer' }} onClick={() => setActionError(null)}>
            {actionError}
            <span style={{ marginLeft: 12, fontSize: 11, opacity: 0.7 }}>Click to dismiss</span>
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
                <button className="refresh-btn" onClick={handleQueueClear}>Clear Queue</button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'overview' && (
          <div className="overview-grid">
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
                    <span>Cycle: {ariaCycle.cycle_index || 0}</span>
                    <span>Mode: {ariaCycle.selected_mode || ariaCycle.last_completed_mode || 'n/a'}</span>
                    <span>{ariaCycle.continuous_active ? 'Continuous active' : 'Continuous idle'}</span>
                  </div>
                </div>
              )}
              <AriaChatPanel isRunning={Boolean(data?.is_running)} autonomousMode={autonomousMode} onAutonomousEnd={() => setAutonomousMode(false)} />
              <details className="card" style={{ marginTop: 12, marginBottom: 0 }} open={Boolean(data?.is_running) || displayMode === 'expert'}>
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
              <div className="card" style={{ marginBottom: 0 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                  <div style={{ fontSize: 13, fontWeight: 600 }}>Activity</div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button className="refresh-btn" onClick={() => setOverviewActivityTab('live')} style={{ opacity: overviewActivityTab === 'live' ? 1 : 0.75 }}>
                      Live updates
                    </button>
                    <button className="refresh-btn" onClick={() => setOverviewActivityTab('recent')} style={{ opacity: overviewActivityTab === 'recent' ? 1 : 0.75 }}>
                      Recent trends
                    </button>
                  </div>
                </div>

                {overviewActivityTab === 'live' ? (
                  data?.is_running ? (
                    <LiveFeed apiBase={API_BASE} />
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
  );
}

export default App;
