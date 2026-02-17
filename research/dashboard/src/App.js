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
    });
  }
  return normalized;
}

function App() {
  const TAB_LABELS = {
    overview: 'Overview',
    experiments: 'Experiments',
    programs: 'Programs (Raw)',
    leaderboard: 'Leaderboard (Curated)',
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
      const res = await fetch(`${API_BASE}/api/dashboard`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      setError(null);
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
      fetchDashboard();
    } catch (err) {
      setActionError('Failed to stop: ' + err.message);
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
    setInvestigationQueue(prev => normalizeQueue([
      ...prev,
      {
        resultId,
        fingerprint: candidate?.fingerprint || null,
        source: candidate?.source || 'unknown',
        architectureFamily: candidate?.architectureFamily || null,
      },
    ]));
  }, []);

  const handleQueueRemove = useCallback((resultId) => {
    setInvestigationQueue(prev => prev.filter(item => item.resultId !== resultId));
  }, []);

  const handleQueueClear = useCallback(() => {
    setInvestigationQueue([]);
  }, []);

  const handleQueueInvestigate = useCallback(() => {
    if (!investigationQueue.length) return;
    handleInvestigate(investigationQueue.map(item => item.resultId));
  }, [investigationQueue]);

  const handleQueueValidate = useCallback(() => {
    if (!investigationQueue.length) return;
    handleValidate(investigationQueue.map(item => item.resultId));
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

        {investigationQueue.length > 0 && (
          <div className="card" style={{ marginBottom: 12, padding: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                Investigation Queue: {investigationQueue.length} candidate{investigationQueue.length === 1 ? '' : 's'} pinned for batch review.
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <button className="refresh-btn" onClick={handleQueueInvestigate}>Investigate Queue</button>
                <button className="refresh-btn" onClick={handleQueueValidate}>Validate Queue</button>
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
              onApplyStrategy={(strategy) => {
                if (strategy.action === null) {
                  setActiveTab('leaderboard');
                } else {
                  setControlPanelPrefill({
                    source: 'strategy_advisor',
                    suggestedMode: strategy.action.suggestedMode,
                    configOverrides: strategy.action.configOverrides || {},
                    requestedAt: Date.now(),
                  });
                }
              }}
            />

            {/* Left column: Aria + Control Panel */}
            <div className="overview-left">
              <AriaStatus aria={data?.aria} isRunning={data?.is_running} progress={data?.progress} />
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
              />
            </div>

            {/* Right column: Summary + Live Feed or Chart */}
            <div className="overview-right">
              <SummaryCards summary={data?.summary} />
              {data?.is_running ? (
                <LiveFeed apiBase={API_BASE} />
              ) : (
                <MetricsChart experiments={data?.recent_experiments} />
              )}
            </div>

            {/* Bottom row */}
            <div className="overview-bottom">
              <TopPrograms
                programs={data?.top_programs?.slice(0, 5)}
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
        />
      )}

      <footer className="app-footer">
        <span>HYDRA Architecture Explorer — Program Synthesis Engine</span>
      </footer>
    </div>
  );
}

export default App;
