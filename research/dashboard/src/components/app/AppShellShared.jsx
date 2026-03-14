import React, { Suspense, useEffect, useState } from 'react';
import LabNotebook from '../LabNotebook';
import CycleTimeline from '../CycleTimeline';
import InsightsPanel from '../InsightsPanel';
import TrendCharts, { ExperimentDataTab } from '../TrendCharts';
import StabilityQualityQuadrant from '../charts/StabilityQualityQuadrant';
import { apiCall } from '../../services/apiService';

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="card" style={{ border: '1px solid var(--accent-red)', padding: 20 }}>
          <h3 style={{ color: 'var(--accent-red)' }}>Component crashed</h3>
          <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>{this.state.error?.message}</p>
          <button className="refresh-btn" onClick={() => this.setState({ hasError: false, error: null })}>
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export function LazyFallback() {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 40,
        color: 'var(--text-muted)',
        fontSize: 13,
      }}
    >
      Loading…
    </div>
  );
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

function TabSelector({ activeKey, options, onChange }) {
  return (
    <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
      {options.map((option) => (
        <button
          key={option.key}
          className="refresh-btn"
          style={{
            fontSize: 12,
            padding: '4px 12px',
            fontWeight: activeKey === option.key ? 700 : 400,
            background: activeKey === option.key ? 'var(--accent-blue)' : 'transparent',
            color: activeKey === option.key ? '#fff' : 'var(--text-secondary)',
            borderColor: activeKey === option.key ? 'var(--accent-blue)' : 'var(--border)',
          }}
          onClick={() => onChange(option.key)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function ReferenceBaselinesPanel() {
  const [refs, setRefs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function fetchRefs() {
      try {
        const res = await apiCall('/api/discoveries?sort=composite_score&limit=200&view=ranked');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        const entries = (json.entries || []).filter((entry) => entry.is_reference);
        if (!cancelled) setRefs(entries);
      } catch {
        // Supplementary panel: ignore fetch failures.
      }
      if (!cancelled) setLoading(false);
    }

    fetchRefs();
    const intervalId = setInterval(fetchRefs, 30000);
    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, []);

  if (loading || refs.length === 0) return null;

  const metricKeys = [
    { key: 'screening_loss_ratio', label: 'Loss Ratio', fmt: (value) => value?.toFixed(4) },
    { key: 'moe_routing_efficiency', label: 'MoE Eff', fmt: (value) => value?.toFixed(3) },
    { key: 'arch_quality_score', label: 'Arch Q', fmt: (value) => value?.toFixed(3) },
    { key: 'screening_novelty', label: 'Novelty', fmt: (value) => value?.toFixed(4) },
    { key: 'composite_score', label: 'Score', fmt: (value) => value?.toFixed(4) },
    { key: 'validation_baseline_ratio', label: 'vs Baseline', fmt: (value) => (value ? `${value.toFixed(2)}x` : '--') },
    { key: 'param_efficiency', label: 'Param Eff', fmt: (value) => (value ? value.toFixed(1) : '--') },
    { key: 'quant_int8_retention', label: 'Quant Ret', fmt: (value) => (value ? `${(value * 100).toFixed(1)}%` : '--') },
    { key: 'robustness_noise_score', label: 'Noise', fmt: (value) => value?.toFixed(2) },
    { key: 'init_sensitivity_std', label: 'Init Std', fmt: (value) => value?.toFixed(4) },
  ];

  return (
    <div
      className="card"
      style={{
        padding: 16,
        marginTop: 16,
        border: '1px solid var(--accent-purple)',
        background: 'rgba(188, 140, 255, 0.04)',
      }}
    >
      <div
        style={{
          fontSize: 13,
          fontWeight: 700,
          marginBottom: 12,
          color: 'var(--accent-purple)',
          textTransform: 'uppercase',
          letterSpacing: 0.5,
        }}
      >
        Reference Architecture Baselines
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table table-compact">
          <thead>
            <tr style={{ borderBottom: '2px solid var(--border)' }}>
              <th style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--text-muted)', fontWeight: 600 }}>
                Architecture
              </th>
              {metricKeys.map((metric) => (
                <th
                  key={metric.key}
                  style={{ textAlign: 'right', padding: '6px 8px', color: 'var(--text-muted)', fontWeight: 600 }}
                >
                  {metric.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {refs
              .slice()
              .sort((a, b) => (a.screening_loss_ratio || 99) - (b.screening_loss_ratio || 99))
              .map((ref) => (
                <tr key={ref.entry_id || ref.result_id} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '8px', fontWeight: 600, color: 'var(--accent-purple)' }}>
                    {ref.reference_name || ref.architecture_desc || 'Reference'}
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 400, marginTop: 2 }}>
                      {ref.tags?.split(',').filter((tag) => tag !== 'reference').join(', ')}
                    </div>
                  </td>
                  {metricKeys.map((metric) => {
                    const value = ref[metric.key];
                    return (
                      <td
                        key={metric.key}
                        style={{
                          textAlign: 'right',
                          padding: '8px',
                          color: 'var(--text-secondary)',
                          fontFamily: 'monospace',
                        }}
                      >
                        {value != null ? metric.fmt(Number(value)) : '--'}
                      </td>
                    );
                  })}
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 8 }}>
        Aria-generated architectures should aim to beat these baselines. Target: 3-5x lower loss ratio than best
        reference.
      </div>
    </div>
  );
}

export function AnalyticsTab({
  data,
  insights,
  leaderboardEntries = [],
  tabData = {},
  tabErrors = {},
  onSelectExperiment,
  onSelectProgram,
  onRerunExperiment,
  onFillGapsExperiment,
  onNavigateStrategy,
  onStartExperiment,
  LearningPanelComponent,
}) {
  const [analyticsView, setAnalyticsView] = useState('trends');
  const resolvedInsights = Array.isArray(tabData?.insights)
    ? tabData.insights
    : Array.isArray(insights)
      ? insights
      : Array.isArray(data?.insights)
        ? data.insights
        : [];

  return (
    <>
      <TabSelector activeKey={analyticsView} options={ANALYTICS_SUB_TABS} onChange={setAnalyticsView} />
      {analyticsView === 'trends' && (
        <ErrorBoundary>
          <Suspense fallback={<LazyFallback />}>
            <div style={{ display: 'grid', gap: 16 }}>
              <TrendCharts onSelectExperiment={onSelectExperiment} />
              <StabilityQualityQuadrant
                entries={leaderboardEntries}
                onSelectProgram={onSelectProgram}
              />
            </div>
          </Suspense>
        </ErrorBoundary>
      )}
      {analyticsView === 'data' && (
        <ErrorBoundary>
          <Suspense fallback={<LazyFallback />}>
            <ExperimentDataTab
              onSelectExperiment={onSelectExperiment}
              onRerunExperiment={onRerunExperiment}
              onFillGapsExperiment={onFillGapsExperiment}
              onStartExperiment={onStartExperiment}
            />
          </Suspense>
        </ErrorBoundary>
      )}
      {analyticsView === 'insights' && (
        <ErrorBoundary>
          <Suspense fallback={<LazyFallback />}>
            <div>
              {tabErrors?.insights && (
                <div className="error-banner" style={{ marginBottom: 12 }}>
                  Fresh insights fetch failed ({tabErrors.insights}); showing dashboard snapshot.
                </div>
              )}
              <InsightsPanel insights={resolvedInsights} />
            </div>
          </Suspense>
        </ErrorBoundary>
      )}
      {analyticsView === 'learning' && (
        <ErrorBoundary>
          <Suspense fallback={<LazyFallback />}>
            <LearningPanelComponent
              onNavigateStrategy={onNavigateStrategy}
              onStartExperiment={onStartExperiment}
            />
          </Suspense>
        </ErrorBoundary>
      )}
      <ErrorBoundary>
        <ReferenceBaselinesPanel />
      </ErrorBoundary>
    </>
  );
}

export function LogTab({ entries, entriesError, onSelectExperiment }) {
  const [logView, setLogView] = useState('notebook');

  return (
    <>
      <TabSelector activeKey={logView} options={LOG_SUB_TABS} onChange={setLogView} />
      {logView === 'notebook' && (
        <Suspense fallback={<LazyFallback />}>
          {entriesError && (
            <div className="error-banner" style={{ marginBottom: 12 }}>
              Fresh notebook fetch failed ({entriesError}); showing dashboard snapshot.
            </div>
          )}
          <LabNotebook entries={entries} onSelectExperiment={onSelectExperiment} />
        </Suspense>
      )}
      {logView === 'cycles' && (
        <Suspense fallback={<LazyFallback />}>
          <CycleTimeline />
        </Suspense>
      )}
    </>
  );
}

export function QuickAnalyticsPreview({ deltas, learningTrajectory, summary, onOpenAnalytics }) {
  const trend = learningTrajectory?.trend;
  const trendLabel =
    trend === 'improving'
      ? 'Improving'
      : trend === 'declining'
        ? 'Declining'
        : trend === 'plateaued'
          ? 'Plateaued'
          : 'Insufficient data';
  const trendColor =
    trend === 'improving'
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
      <div
        className="card-title"
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}
      >
        <span>Quick Analytics</span>
        <button className="refresh-btn" onClick={onOpenAnalytics} style={{ fontSize: 11 }}>
          Open Analytics
        </button>
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
        <div>
          Trend: <strong style={{ color: trendColor }}>{trendLabel}</strong>
          {learningTrajectory?.slope != null && (
            <span style={{ color: trendColor }}>
              {' '}
              · {learningTrajectory.slope > 0 ? '+' : ''}
              {(learningTrajectory.slope * 100).toFixed(2)}%/exp
            </span>
          )}
        </div>
        <div>
          Δ Loss: {deltaLoss != null ? `${deltaLoss > 0 ? '+' : ''}${deltaLoss.toFixed(4)}` : 'n/a'} · Δ Novelty:{' '}
          {deltaNovelty != null ? `${deltaNovelty > 0 ? '+' : ''}${deltaNovelty.toFixed(3)}` : 'n/a'}
        </div>
        <div>
          Δ S1 survivors: {deltaStage1 != null ? `${deltaStage1 > 0 ? '+' : ''}${deltaStage1}` : 'n/a'} · Δ Programs:{' '}
          {deltaPrograms != null ? `${deltaPrograms > 0 ? '+' : ''}${deltaPrograms}` : 'n/a'}
        </div>
        <div>
          Avg throughput: {avgThroughput != null ? `${Math.round(avgThroughput).toLocaleString()} tok/s` : 'n/a'} ·
          Routing entropy: {routingEntropy != null ? routingEntropy.toFixed(2) : 'n/a'} · Token retention:{' '}
          {avgRoutingRetention != null ? `${(avgRoutingRetention * 100).toFixed(1)}%` : 'n/a'}
        </div>
        <div>
          Depth savings: {depthSavings != null ? `${(depthSavings * 100).toFixed(1)}%` : 'n/a'} · Recursion savings:{' '}
          {avgRecursionSavings != null ? `${(avgRecursionSavings * 100).toFixed(1)}%` : 'n/a'}
        </div>
      </div>
    </div>
  );
}
