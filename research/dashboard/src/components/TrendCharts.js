import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { formatTime, formatDuration, scoreColor } from '../utils/format';
import { lossColor, noveltyColor } from '../utils/colors';
import { trendScore, trendScoreBreakdown } from '../utils/scoringEngine';
import useCopyToClipboard from '../hooks/useCopyToClipboard';
import apiService from '../services/apiService';
import { filterRowsByQuery } from '../utils/tableFiltering';
import { CHART_DEFAULTS, clampToScale, getFixedScale } from '../utils/chartScales';
import ChartActions from './ChartActions';

import MiniChart, { TREND_CHART_WINDOW } from './charts/MiniChart';
import RegressionBaselineChart from './charts/RegressionBaselineChart';
import ParetoEfficiencyChart from './charts/ParetoEfficiencyChart';
import ExperimentDataTab from './charts/ExperimentDataTab';
export { ExperimentDataTab };

/**
 * TrendCharts — Cross-experiment line charts using inline SVG
 * plus a sortable data table with per-experiment scores.
 */
function TrendCharts({ onSelectExperiment }) {
  const [trends, setTrends] = useState(null);
  const [weightEvents, setWeightEvents] = useState([]);
  const [frontier, setFrontier] = useState([]);
  const [topPrograms, setTopPrograms] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTier] = useState('trends'); // 'trends' or 'data' or 'efficiency'

  const fetchData = useCallback(async (isBackground = false) => {
    if (!isBackground) {
      setLoading(true);
      setError(null);
    }
    try {
      const [tData, frData, tpData] = await Promise.all([
        apiService.getTrends(),
        apiService.getEfficiencyFrontier(),
        apiService.getPrograms(50),
      ]);
      setTrends(Array.isArray(tData?.trends) ? tData.trends : []);
      setWeightEvents(Array.isArray(tData?.adaptation_events) ? tData.adaptation_events : []);
      setFrontier(Array.isArray(frData) ? frData : []);
      setTopPrograms(Array.isArray(tpData) ? tpData : []);
      setError(null);
    } catch (e) {
      if (!isBackground) setError('Failed to load trends: ' + e.message);
    } finally {
      if (!isBackground) setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData(false);
    const interval = setInterval(() => fetchData(true), 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (loading) {
    return (
      <div className="card">
        <div className="ux-state ux-state-loading">
          <span className="ux-spinner" />
          <div className="ux-stack">
            <span className="ux-state-title">Analyzing trends</span>
            <span className="ux-state-subtle">Aggregating cross-experiment KPIs and Pareto frontier.</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) return <div className="card"><p style={{ color: 'var(--accent-red)' }}>{error}</p></div>;

  if (!trends || trends.length < 2) {
    return (
      <div className="card">
        <div className="card-title">Research Trends</div>
        <div className="empty-state">
          <div className="empty-state-icon">&#x1F4C8;</div>
          <div className="empty-state-title">Not enough data yet</div>
          <p className="empty-state-hint">
            Need at least 2 completed experiments to visualize search trends.
            Run your first experiment from the Control tab.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Tab Switcher */}
      <div role="tablist" aria-label="Trend views" style={{ display: 'flex', gap: 4, marginBottom: 4 }}>
        {[
          { id: 'trends', label: 'Research Trends' },
          { id: 'efficiency', label: 'Efficiency Frontier' },
          { id: 'data', label: 'Full Experiment Log' },
        ].map(tab => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            onClick={() => setActiveTier(tab.id)}
            className={`step-btn ${activeTab === tab.id ? 'active' : ''}`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'trends' && (
        <div className="card">
          <div className="card-title" style={{ marginBottom: 12 }}>Research Trends</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 24 }}>
            <MiniChart
              data={trends}
              valueKey="s1_pass_rate"
              label="Stage 1 Pass Rate"
              color="var(--accent-green)"
              formatValue={v => `${(v * 100).toFixed(1)}%`}
              weightEvents={weightEvents}
              scaleKey="s1_rate"
              onSelectExperiment={onSelectExperiment}
            />
            <MiniChart
              data={trends}
              valueKey="best_loss_ratio"
              label="Best Loss Ratio"
              color="var(--accent-blue)"
              scaleKey="loss_ratio"
              onSelectExperiment={onSelectExperiment}
            />
            <MiniChart
              data={trends}
              valueKey="best_novelty_score"
              label="Max Novelty"
              color="var(--accent-purple)"
              scaleKey="novelty"
              onSelectExperiment={onSelectExperiment}
            />
            <MiniChart
              data={trends}
              valueKey="avg_throughput_tok_s"
              label="Avg Throughput"
              color="var(--accent-yellow)"
              formatValue={v => `${Math.round(v).toLocaleString()} tok/s`}
              scaleKey="throughput_tok_s"
              onSelectExperiment={onSelectExperiment}
            />
            <MiniChart
              data={trends}
              valueKey="avg_sample_efficiency"
              label="Avg Sample Efficiency"
              color="var(--accent-cyan, #22d3ee)"
              formatValue={v => v != null ? v.toFixed(3) : '--'}
              scaleKey="sample_efficiency"
              onSelectExperiment={onSelectExperiment}
            />
          </div>
        </div>
      )}

      {activeTab === 'efficiency' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div className="card">
            <div className="card-title">Regression vs Baseline</div>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>
              Targeting the <strong>Bottom-Right</strong>: beating the baseline loss ratio ({"<"} 1.0) 
              at high throughput. The dashed line represents the current Pareto frontier.
            </p>
            <RegressionBaselineChart points={topPrograms} frontier={frontier} />
          </div>
          <div className="card">
            <div className="card-title">Pareto Efficiency (Acc vs Eff)</div>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>
              Global efficiency view. X-axis is normalized throughput, Y-axis is accuracy (1 - LR).
              Bubble size represents parameter compression ratio.
            </p>
            <ParetoEfficiencyChart points={topPrograms} />
          </div>
        </div>
      )}

      {activeTab === 'data' && (
        <ExperimentDataTab onSelectExperiment={onSelectExperiment} />
      )}
    </div>
  );
}

export default TrendCharts;
