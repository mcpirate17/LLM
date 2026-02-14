import React, { useState, useEffect, useCallback } from 'react';
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
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || '';

function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Drill-down state
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [selectedProgram, setSelectedProgram] = useState(null);

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

  const handleStartExperiment = async (config) => {
    try {
      const res = await fetch(`${API_BASE}/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(err.error || 'Failed to start experiment');
        return;
      }
      // Refresh immediately to get new state
      fetchDashboard();
    } catch (err) {
      alert('Failed to start experiment: ' + err.message);
    }
  };

  const handleStopExperiment = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/experiments/stop`, {
        method: 'POST',
      });
      if (!res.ok) {
        const err = await res.json();
        alert(err.error || 'Failed to stop experiment');
        return;
      }
      fetchDashboard();
    } catch (err) {
      alert('Failed to stop: ' + err.message);
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

  const ariaMood = data?.aria?.mood || 'curious';

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
        {['overview', 'experiments', 'programs', 'trends', 'notebook', 'insights'].map(tab => (
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
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
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

        {activeTab === 'overview' && (
          <div className="overview-grid">
            {/* Left column: Aria + Control Panel */}
            <div className="overview-left">
              <AriaStatus aria={data?.aria} isRunning={data?.is_running} progress={data?.progress} />
              <ControlPanel
                isRunning={data?.is_running}
                progress={data?.progress}
                onStart={handleStartExperiment}
                onStop={handleStopExperiment}
                onRefresh={fetchDashboard}
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
                compact
                onSelectProgram={handleSelectProgram}
              />
              <InsightsPanel insights={data?.insights?.slice(0, 5)} compact />
            </div>
          </div>
        )}

        {activeTab === 'experiments' && (
          <ExperimentList
            experiments={data?.recent_experiments}
            onSelectExperiment={handleSelectExperiment}
          />
        )}

        {activeTab === 'experiment-detail' && selectedExperiment && (
          <ExperimentDetail
            experimentId={selectedExperiment}
            onBack={handleBackFromExperiment}
            onSelectProgram={handleSelectProgram}
          />
        )}

        {activeTab === 'programs' && (
          <TopPrograms
            programs={data?.top_programs}
            onSelectProgram={handleSelectProgram}
          />
        )}

        {activeTab === 'trends' && (
          <TrendCharts />
        )}

        {activeTab === 'notebook' && (
          <LabNotebook entries={data?.recent_entries} />
        )}

        {activeTab === 'insights' && (
          <InsightsPanel insights={data?.insights} />
        )}
      </main>

      {/* Program Detail Modal (global) */}
      {selectedProgram && (
        <ProgramDetail
          resultId={selectedProgram}
          onClose={() => setSelectedProgram(null)}
        />
      )}

      <footer className="app-footer">
        <span>HYDRA Architecture Explorer — Program Synthesis Engine</span>
      </footer>
    </div>
  );
}

export default App;
