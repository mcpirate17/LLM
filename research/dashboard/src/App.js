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
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || '';

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

        {activeTab === 'overview' && (
          <div className="overview-grid">
            <div className="card" style={{ gridColumn: '1 / -1', marginBottom: 0 }}>
              <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                This AI scientist is focused on one job: searching for novel neural network layer designs that learn
                more efficiently than standard transformers. It does not tune training recipes or datasets — it
                generates and evaluates architecture candidates.
              </div>
            </div>

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
            />
          </>
        )}

        {activeTab === 'leaderboard' && (
          <Leaderboard
            onSelectProgram={handleSelectProgram}
            onInvestigate={handleInvestigate}
            onValidate={handleValidate}
          />
        )}

        {activeTab === 'campaigns' && (
          <CampaignView
            onSelectExperiment={handleSelectExperiment}
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
        />
      )}

      <footer className="app-footer">
        <span>HYDRA Architecture Explorer — Program Synthesis Engine</span>
      </footer>
    </div>
  );
}

export default App;
