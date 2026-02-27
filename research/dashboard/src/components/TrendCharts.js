import React, { useState, useEffect, useMemo } from 'react';
import { formatTime, formatDuration, scoreColor } from '../utils/format';
import { lossColor, noveltyColor } from '../utils/colors';
import { trendScore, trendScoreBreakdown } from '../utils/scoringEngine';
import useCopyToClipboard from '../hooks/useCopyToClipboard';
import apiService from '../services/apiService';
import { filterRowsByQuery } from '../utils/tableFiltering';
import { CHART_DEFAULTS, clampToScale, getFixedScale } from '../utils/chartScales';


/**
 * TrendCharts — Cross-experiment line charts using inline SVG
 * plus a sortable data table with per-experiment scores.
 */

const TREND_CHART_WINDOW = 30;

function MiniChart({ data, valueKey, label, color, formatValue, weightEvents, bandLowerKey, bandUpperKey, scaleKey, windowSize = TREND_CHART_WINDOW }) {
  if (!data || data.length < 2) {
    return (
      <div style={{ textAlign: 'center', padding: 16, color: 'var(--text-muted)', fontSize: 13 }}>
        Need at least 2 experiments for {label} trend
      </div>
    );
  }

  const windowed = data.slice(-windowSize);
  const values = windowed.map(d => d[valueKey]).filter(v => v != null && Number.isFinite(v));
  if (values.length < 2) return null;

  const W = 400;
  const H = 120;
  const PAD = 24;

  const defaults = CHART_DEFAULTS[scaleKey] || { min: 0, max: 1 };
  const scale = getFixedScale(`trend.${scaleKey}`, values, {
    defaultMin: defaults.min,
    defaultMax: defaults.max,
  });
  const min = scale.min;
  const max = scale.max;
  const range = max - min || 1;

  const tMin = windowed[0]?.timestamp || 0;
  const tMax = windowed[windowed.length - 1]?.timestamp || 1;
  const tRange = tMax - tMin || 1;

  const denom = Math.max(1, windowSize - 1);
  const points = windowed
    .map((d, i) => {
      const v = d[valueKey];
      if (v == null) return null;
      const x = PAD + (i / denom) * (W - 2 * PAD);
      const clamped = clampToScale(v, scale);
      const y = H - PAD - ((clamped - min) / range) * (H - 2 * PAD);
      const lowerRaw = bandLowerKey ? d[bandLowerKey] : null;
      const upperRaw = bandUpperKey ? d[bandUpperKey] : null;
      const hasBand = lowerRaw != null && upperRaw != null;
      const lower = hasBand ? clampToScale(lowerRaw, scale) : null;
      const upper = hasBand ? clampToScale(upperRaw, scale) : null;
      return { x, y, v: clamped, idx: i, lower, upper, hasBand };
    })
    .filter(Boolean);

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
  const bandPoints = points.filter((p) => p.hasBand);
  const bandPathD = bandPoints.length >= 2
    ? `${bandPoints.map((p, i) => {
        const yUpper = H - PAD - ((p.upper - min) / range) * (H - 2 * PAD);
        return `${i === 0 ? 'M' : 'L'} ${p.x} ${yUpper}`;
      }).join(' ')} ${bandPoints.slice().reverse().map((p) => {
        const yLower = H - PAD - ((p.lower - min) / range) * (H - 2 * PAD);
        return `L ${p.x} ${yLower}`;
      }).join(' ')} Z`
    : null;

  const fmt = formatValue || (v => v.toFixed(3));

  // Linear regression for trend line
  const n = points.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (const p of points) {
    sumX += p.idx;
    sumY += p.v;
    sumXY += p.idx * p.v;
    sumXX += p.idx * p.idx;
  }
  const regDenom = n * sumXX - sumX * sumX;
  const slope = regDenom !== 0 ? (n * sumXY - sumX * sumY) / regDenom : 0;
  const intercept = (sumY - slope * sumX) / n;
  const regY0 = intercept;
  const regYN = intercept + slope * (windowed.length - 1);
  const drawReg = regDenom !== 0 && Number.isFinite(regY0) && Number.isFinite(regYN);
  const regPx0 = drawReg ? H - PAD - ((Math.min(Math.max(regY0, min), max) - min) / range) * (H - 2 * PAD) : 0;
  const regPxN = drawReg ? H - PAD - ((Math.min(Math.max(regYN, min), max) - min) / range) * (H - 2 * PAD) : 0;

  // Compute weight event marker positions with before/after comparison
  const markers = (weightEvents || [])
    .filter(e => e.timestamp >= tMin && e.timestamp <= tMax)
    .map(e => {
      const x = PAD + ((e.timestamp - tMin) / (tRange || 1)) * (W - 2 * PAD);
      // Find experiments before and after this weight event
      const eventIdx = windowed.findIndex(d => (d.timestamp || 0) >= e.timestamp);
      const before = windowed.slice(Math.max(0, eventIdx - 3), eventIdx)
        .map(d => d[valueKey]).filter(v => v != null);
      const after = windowed.slice(eventIdx, eventIdx + 3)
        .map(d => d[valueKey]).filter(v => v != null);
      const avgBefore = before.length > 0 ? before.reduce((a, b) => a + b, 0) / before.length : null;
      const avgAfter = after.length > 0 ? after.reduce((a, b) => a + b, 0) / after.length : null;
      let delta = null;
      if (avgBefore != null && avgAfter != null) {
        delta = avgAfter - avgBefore;
      }
      return {
        x,
        desc: e.description || 'Grammar weights adjusted',
        delta,
        avgBefore,
        avgAfter,
      };
    });

  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4, fontWeight: 600 }}>
        {label}
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}
        style={{ width: '100%', height: 'auto' }}>
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map(frac => {
          const y = H - PAD - frac * (H - 2 * PAD);
          return (
            <g key={frac}>
              <line x1={PAD} y1={y} x2={W - PAD} y2={y}
                stroke="var(--border, #30363d)" strokeWidth={0.5} />
              <text x={2} y={y + 3} fontSize={8} fill="var(--text-muted, #484f58)">
                {fmt(min + frac * range)}
              </text>
            </g>
          );
        })}

        {/* Regression trend line */}
        {drawReg && (
          <line x1={PAD} y1={regPx0} x2={PAD + (windowed.length - 1) / denom * (W - 2 * PAD)} y2={regPxN}
            stroke={color} strokeWidth={1.5} strokeDasharray="6 3" opacity={0.5} />
        )}

        {/* Optional confidence band */}
        {bandPathD && (
          <path d={bandPathD} fill={color} opacity={0.16} stroke="none" />
        )}

        {/* Weight adjustment markers with before/after regions */}
        {markers.map((m, i) => (
          <g key={`wm-${i}`}>
            <line x1={m.x} y1={PAD - 4} x2={m.x} y2={H - PAD}
              stroke="var(--accent-orange, #f0883e)" strokeWidth={1} strokeDasharray="3 2" opacity={0.7} />
            <text x={m.x} y={PAD - 6} textAnchor="middle" fontSize={7}
              fill="var(--accent-orange, #f0883e)">
              W
            </text>
            {m.delta != null && (
              <text x={m.x + 3} y={H - PAD + 10} textAnchor="start" fontSize={7}
                fill={m.delta > 0 ? 'var(--accent-green, #3fb950)' : m.delta < 0 ? 'var(--accent-red, #f85149)' : 'var(--text-muted)'}>
                {m.delta > 0 ? '+' : ''}{fmt(m.delta)}
              </text>
            )}
            <title>{m.desc}{m.avgBefore != null && m.avgAfter != null ?
              ` | Before: ${fmt(m.avgBefore)} → After: ${fmt(m.avgAfter)} (${m.delta > 0 ? '+' : ''}${fmt(m.delta)})` : ''}</title>
          </g>
        ))}

        {/* Line */}
        <path d={pathD} fill="none" stroke={color} strokeWidth={2} />

        {/* Dots */}
        {points.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={3}
            fill={color} stroke="var(--bg-secondary, #161b22)" strokeWidth={1.5}>
            <title>Exp #{p.idx + 1}: {fmt(p.v)}</title>
          </circle>
        ))}
      </svg>
    </div>
  );
}

function metricText(value, fallbackReason, formatter) {
  if (value == null) return fallbackReason;
  return formatter(value);
}

function RegressionBaselineChart({ points, frontier }) {
  if (!Array.isArray(points) || points.length === 0) return null;
  const W = 440;
  const H = 180;
  const PAD = 28;
  const xs = points.map((p) => Number(p.throughput_tok_s || 0));
  const ys = points.map((p) => Number(p.baseline_loss_ratio || 0));
  const xDefaults = CHART_DEFAULTS.throughput_tok_s;
  const yDefaults = CHART_DEFAULTS.baseline_ratio;
  const xScale = getFixedScale('trend.throughput_tok_s', xs, {
    defaultMin: xDefaults.min,
    defaultMax: xDefaults.max,
  });
  const yScale = getFixedScale('trend.baseline_ratio', ys, {
    defaultMin: yDefaults.min,
    defaultMax: yDefaults.max,
  });
  const xMin = xScale.min;
  const xMax = xScale.max;
  const yMin = yScale.min;
  const yMax = yScale.max;
  const xRange = (xMax - xMin) || 1;
  const yRange = (yMax - yMin) || 1;

  const project = (x, y) => ({
    x: PAD + ((clampToScale(x, xScale) - xMin) / xRange) * (W - PAD * 2),
    y: H - PAD - ((clampToScale(y, yScale) - yMin) / yRange) * (H - PAD * 2),
  });
  const frontierPath = (frontier || [])
    .map((p, i) => {
      const pt = project(Number(p.throughput_tok_s || 0), Number(p.baseline_loss_ratio || 0));
      return `${i === 0 ? 'M' : 'L'} ${pt.x} ${pt.y}`;
    })
    .join(' ');

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', maxWidth: W }}>
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--border)" strokeWidth={1} />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="var(--border)" strokeWidth={1} />
      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} fill="var(--text-muted)">Throughput (tok/s)</text>
      <text x={8} y={H / 2} transform={`rotate(-90 8 ${H / 2})`} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
        Baseline Ratio (lower is better)
      </text>
      {points.map((p, idx) => {
        const pt = project(Number(p.throughput_tok_s || 0), Number(p.baseline_loss_ratio || 0));
        const beats = Number(p.baseline_loss_ratio || 0) < 1.0;
        return (
          <circle
            key={`${p.result_id || idx}`}
            cx={pt.x}
            cy={pt.y}
            r={3}
            fill={beats ? 'var(--accent-green)' : 'var(--accent-yellow)'}
            opacity={0.85}
          >
            <title>
              {`${(p.result_id || '').slice(0, 12)} | baseline=${Number(p.baseline_loss_ratio || 0).toFixed(3)} | throughput=${Math.round(Number(p.throughput_tok_s || 0))} tok/s`}
            </title>
          </circle>
        );
      })}
      {frontierPath && <path d={frontierPath} fill="none" stroke="var(--accent-red)" strokeWidth={1.5} strokeDasharray="4 3" />}
    </svg>
  );
}


const COLUMNS = [
  { key: '_score', label: 'Score' },
  { key: 'experiment_id', label: 'ID' },
  { key: 's1_pass_rate', label: 'S1 Rate (per-exp)' },
  { key: 'trend_confidence', label: 'Confidence' },
  { key: 'best_loss_ratio', label: 'Best Loss' },
  { key: 'best_novelty_score', label: 'Best Novelty' },
  {
    key: 'avg_throughput_tok_s',
    label: 'Avg Throughput',
    tooltip: 'Average per-program throughput (tok/s). Falls back to perf report if available.'
  },
  {
    key: 'avg_routing_token_retention',
    label: 'Routing Retention',
    tooltip: 'Share of tokens processed by routing modules (higher is better).'
  },
  {
    key: 'avg_routing_utilization_entropy',
    label: 'Routing Entropy',
    tooltip: 'Load-balance entropy across experts (higher = more balanced).'
  },
  {
    key: 'avg_depth_savings_ratio',
    label: 'Depth Savings',
    tooltip: 'MoD savings vs full depth (higher = more compute saved).'
  },
  {
    key: 'avg_recursion_savings_ratio',
    label: 'Recursion Savings',
    tooltip: 'MoR savings vs max recursion (higher = more compute saved).'
  },
  { key: 'n_programs_generated', label: 'Programs' },
  { key: 'n_stage1_passed', label: 'S1 Pass' },
  { key: 'duration_seconds', label: 'Duration' },
  { key: 'timestamp', label: 'Time' },
];

function TrendCharts({ onSelectExperiment }) {
  const [trends, setTrends] = useState(null);
  const [weightEvents, setWeightEvents] = useState([]);
  const [adaptationEvents, setAdaptationEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [statusFilter, setStatusFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  const [outcomeFilter, setOutcomeFilter] = useState('all');
  const [chartWindowSize, setChartWindowSize] = useState('30');
  const [lastUpdated, setLastUpdated] = useState(null);
  const [regressionVsBaseline, setRegressionVsBaseline] = useState({
    points: [],
    pareto_frontier: [],
    summary: null,
  });

  useEffect(() => {
    let active = true;

    const fetchTrendContext = async () => {
      try {
        const [payload, regressionPayload] = await Promise.all([
          apiService.getTrends(),
          apiService.getRegressionVsBaseline().catch(() => null),
        ]);
        if (!active) return;

        const trendsData = Array.isArray(payload?.trends) ? payload.trends : [];
        const adaptationData = Array.isArray(payload?.adaptation_events) ? payload.adaptation_events : [];
        setTrends(trendsData);
        setAdaptationEvents(adaptationData);
        setWeightEvents(
          adaptationData
            .map(event => ({
              event_type: event?.event_type,
              timestamp: event?.timestamp,
              description: event?.description,
            }))
            .filter(event => event.timestamp != null)
        );
        setLastUpdated(payload?.generated_at ? new Date(payload.generated_at * 1000) : new Date());
        if (regressionPayload && Array.isArray(regressionPayload.points)) {
          setRegressionVsBaseline({
            points: regressionPayload.points,
            pareto_frontier: regressionPayload.pareto_frontier || [],
            summary: regressionPayload.summary || null,
          });
        }
        setError(null);
      } catch (e) {
        if (!active) return;
        setError('Failed to load trends: ' + e.message);
      } finally {
        if (active) setLoading(false);
      }
    };

    fetchTrendContext();
    const interval = setInterval(fetchTrendContext, 10000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const augmented = useMemo(() => {
    if (!trends) return [];
    return trends.map(d => ({ ...d, _score: trendScore(d) }));
  }, [trends]);

  const experimentTypes = useMemo(() => {
    const unique = Array.from(new Set(
      augmented
        .map((row) => row?.experiment_type)
        .filter((value) => typeof value === 'string' && value.trim().length > 0)
    ));
    unique.sort((a, b) => a.localeCompare(b));
    return unique;
  }, [augmented]);

  const statusTypeOutcomeFiltered = useMemo(() => (
    augmented.filter((row) => {
      if (statusFilter !== 'all' && row.status !== statusFilter) return false;
      if (typeFilter !== 'all' && row.experiment_type !== typeFilter) return false;
      if (outcomeFilter === 'has_s1' && (row.n_stage1_passed || 0) <= 0) return false;
      if (outcomeFilter === 'no_s1' && (row.n_stage1_passed || 0) > 0) return false;
      return true;
    })
  ), [augmented, statusFilter, typeFilter, outcomeFilter]);

  const hasActiveFilters = (
    statusFilter !== 'all' ||
    typeFilter !== 'all' ||
    outcomeFilter !== 'all'
  );

  const clearFilters = () => {
    setStatusFilter('all');
    setTypeFilter('all');
    setOutcomeFilter('all');
  };

  const filtered = statusTypeOutcomeFiltered;

  const adaptationTimeline = useMemo(() => {
    if (!adaptationEvents || adaptationEvents.length === 0) return [];
    return [...adaptationEvents]
      .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0))
      .slice(0, 6)
      .map((event) => {
        const deltaS1 = event?.delta?.adjusted_s1_rate;
        const deltaNovelty = event?.delta?.best_novelty;
        const deltaLoss = event?.delta?.best_loss_ratio;
        const improved = (deltaS1 != null && deltaS1 > 0) || (deltaLoss != null && deltaLoss < 0);
        const degraded = (deltaS1 != null && deltaS1 < 0) || (deltaLoss != null && deltaLoss > 0);
        const verdict = improved && !degraded ? 'improved' : degraded && !improved ? 'regressed' : 'mixed';
        const windowCount = event?.after_window?.n_experiments ?? 0;
        const summaryParts = [];
        if (deltaS1 != null) {
          summaryParts.push(`S1 ${deltaS1 >= 0 ? 'improved' : 'declined'} by ${Math.abs(deltaS1 * 100).toFixed(1)}%`);
        }
        if (deltaLoss != null) {
          summaryParts.push(`loss ${deltaLoss <= 0 ? 'improved' : 'worsened'} by ${Math.abs(deltaLoss).toFixed(4)}`);
        }
        if (deltaNovelty != null) {
          summaryParts.push(`novelty ${deltaNovelty >= 0 ? 'up' : 'down'} ${Math.abs(deltaNovelty).toFixed(3)}`);
        }
        const summary = summaryParts.length > 0
          ? `Grammar adapted → ${summaryParts.join(', ')} over next ${windowCount} experiment${windowCount === 1 ? '' : 's'}.`
          : 'Grammar adapted → insufficient delta data.';
        return {
          ...event,
          verdict,
          summary,
        };
      });
  }, [adaptationEvents]);

  if (loading) return <div className="card"><p style={{ color: 'var(--text-muted)' }}>Loading trends...</p></div>;
  if (error) return <div className="card"><p style={{ color: 'var(--accent-red)' }}>{error}</p></div>;
  if (!trends || trends.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Experiment Trends</div>
        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
          No completed experiments yet. Trends will appear after 2+ experiments.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Experiment Trends</div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
        Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'loading'} · Source: /api/trends
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        How the search is improving over time. Rising S1 pass rate means the grammar is learning
        to generate better architectures. Decreasing loss ratio means the survivors are learning
        faster. These trends show whether the system's self-improvement loop is working.
      </p>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
          aria-label="Trends filter by status"
        >
          <option value="all">All status</option>
          <option value="completed">Completed</option>
          <option value="running">Running</option>
          <option value="failed">Failed</option>
        </select>
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
          aria-label="Trends filter by type"
        >
          <option value="all">All types</option>
          {experimentTypes.map((type) => (
            <option key={type} value={type}>{type}</option>
          ))}
        </select>
        <select
          value={outcomeFilter}
          onChange={(e) => setOutcomeFilter(e.target.value)}
          style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
          aria-label="Trends filter by outcome"
        >
          <option value="all">All outcomes</option>
          <option value="has_s1">Has S1 pass</option>
          <option value="no_s1">No S1 pass</option>
        </select>
        <select
          value={chartWindowSize}
          onChange={(e) => setChartWindowSize(e.target.value)}
          style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
          aria-label="Trends chart window size"
        >
          <option value="10">10 points</option>
          <option value="20">20 points</option>
          <option value="30">30 points</option>
          <option value="50">50 points</option>
          <option value="100">100 points</option>
        </select>
        <button
          className="refresh-btn"
          style={{ fontSize: 11, padding: '3px 10px' }}
          onClick={clearFilters}
          disabled={!hasActiveFilters}
        >
          Clear filters
        </button>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
        Showing {filtered.length} filtered experiments.
      </div>
      {filtered.length === 0 && (
        <div className="card" style={{ marginBottom: 12, padding: 10 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: 'var(--text-primary)' }}>
            No experiments match current filters
          </div>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: 0 }}>
            Adjust status/type/outcome filters or use Clear filters to restore the trend views.
          </p>
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 16 }}>
        <MiniChart
          data={filtered}
          valueKey="adjusted_s1_pass_rate"
          label="S1 Pass Rate (Bayesian-adjusted)"
          color="var(--accent-green, #3fb950)"
          formatValue={v => `${(v * 100).toFixed(1)}%`}
          weightEvents={weightEvents}
          bandLowerKey="s1_confidence_lower"
          bandUpperKey="s1_confidence_upper"
          scaleKey="s1_rate"
          windowSize={Number(chartWindowSize)}
        />
        <MiniChart
          data={filtered}
          valueKey="best_loss_ratio"
          label="Best Loss Ratio"
          color="var(--accent-yellow, #d29922)"
          formatValue={v => v.toFixed(4)}
          weightEvents={weightEvents}
          scaleKey="loss_ratio"
          windowSize={Number(chartWindowSize)}
        />
        <MiniChart
          data={filtered}
          valueKey="best_novelty_score"
          label="Best Novelty Score"
          color="var(--accent-purple, #bc8cff)"
          weightEvents={weightEvents}
          scaleKey="novelty"
          windowSize={Number(chartWindowSize)}
        />
        <MiniChart
          data={filtered}
          valueKey="avg_throughput_tok_s"
          label="Average Throughput (tok/s)"
          color="var(--accent-blue, #58a6ff)"
          formatValue={v => Math.round(v).toLocaleString()}
          weightEvents={weightEvents}
          scaleKey="throughput_tok_s"
          windowSize={Number(chartWindowSize)}
        />
        <MiniChart
          data={filtered}
          valueKey="avg_routing_token_retention"
          label="Routing Token Retention (MoE)"
          color="var(--accent-green, #3fb950)"
          formatValue={v => `${(v * 100).toFixed(1)}%`}
          weightEvents={weightEvents}
          scaleKey="routing_token_retention"
          windowSize={Number(chartWindowSize)}
        />
        <MiniChart
          data={filtered}
          valueKey="avg_routing_utilization_entropy"
          label="Routing Utilization Entropy (MoE)"
          color="var(--accent-green, #2ea043)"
          formatValue={v => v.toFixed(3)}
          weightEvents={weightEvents}
          scaleKey="routing_entropy"
          windowSize={Number(chartWindowSize)}
        />
        <MiniChart
          data={filtered}
          valueKey="avg_depth_savings_ratio"
          label="Depth Savings Ratio (MoD)"
          color="#c77dff"
          formatValue={v => `${(v * 100).toFixed(1)}%`}
          weightEvents={weightEvents}
          scaleKey="depth_savings_ratio"
          windowSize={Number(chartWindowSize)}
        />
        <MiniChart
          data={filtered}
          valueKey="avg_recursion_savings_ratio"
          label="Recursion Savings Ratio (MoR)"
          color="#9f7aea"
          formatValue={v => `${(v * 100).toFixed(1)}%`}
          weightEvents={weightEvents}
          scaleKey="recursion_savings_ratio"
          windowSize={Number(chartWindowSize)}
        />
      </div>
      {weightEvents.length > 0 && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ color: 'var(--accent-orange, #f0883e)', fontWeight: 600 }}>W</span>
          <span>= grammar weight adjustment ({weightEvents.length} total). Dashed orange lines mark when the system adapted its search strategy.</span>
        </div>
      )}
      {adaptationTimeline.length > 0 && (
        <div className="card" style={{ marginBottom: 12, padding: 10 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--text-primary)' }}>
            Adaptation outcomes (recent)
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {adaptationTimeline.map((event, index) => (
              <div
                key={`${event.timestamp || 0}-${index}`}
                style={{
                  fontSize: 11,
                  color: 'var(--text-secondary)',
                  padding: '6px 8px',
                  borderRadius: 6,
                  background: 'var(--bg-tertiary)',
                  border: '1px solid var(--border)',
                }}
              >
                <div style={{ marginBottom: 4 }}>
                  <strong>{formatTime(event.timestamp)}</strong> · {event.description || 'Grammar weights adjusted'} ·
                  {' '}<span style={{
                    color: event.verdict === 'improved'
                      ? 'var(--accent-green)'
                      : event.verdict === 'regressed'
                        ? 'var(--accent-red)'
                        : 'var(--accent-yellow)',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    fontSize: 10,
                  }}>{event.verdict}</span>
                </div>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', color: 'var(--text-muted)' }}>
                  <span>
                    Δ S1 adj: {event?.delta?.adjusted_s1_rate != null
                      ? `${event.delta.adjusted_s1_rate > 0 ? '+' : ''}${(event.delta.adjusted_s1_rate * 100).toFixed(2)}%`
                      : 'n/a'}
                  </span>
                  <span>
                    Δ novelty: {event?.delta?.best_novelty != null
                      ? `${event.delta.best_novelty > 0 ? '+' : ''}${event.delta.best_novelty.toFixed(3)}`
                      : 'n/a'}
                  </span>
                  <span>
                    Δ loss: {event?.delta?.best_loss_ratio != null
                      ? `${event.delta.best_loss_ratio > 0 ? '+' : ''}${event.delta.best_loss_ratio.toFixed(4)}`
                      : 'n/a'}
                  </span>
                  <span>
                    windows: {event?.before_window?.n_experiments ?? 0} → {event?.after_window?.n_experiments ?? 0}
                  </span>
                </div>
                {event.summary && (
                  <div style={{ marginTop: 4, fontSize: 11, color: 'var(--text-muted)' }}>
                    {event.summary}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12 }}>
        Stage-1 trend uses stabilized pass-rate estimates weighted by experiment size and mode; shaded region shows 95% confidence band.
      </div>
      {Array.isArray(regressionVsBaseline.points) && regressionVsBaseline.points.length > 0 && (
        <div className="card" style={{ marginBottom: 14, padding: 10 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--text-primary)' }}>
            Regression vs Baseline (Accuracy/Speed Tradeoff)
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
            Scatter of Stage-1 survivors by throughput vs baseline ratio. Dashed red line is Pareto frontier.
          </div>
          <RegressionBaselineChart
            points={regressionVsBaseline.points.slice(0, 120)}
            frontier={regressionVsBaseline.pareto_frontier || []}
          />
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
            {(regressionVsBaseline.summary?.n_points || 0)} points ·
            {' '}{(regressionVsBaseline.summary?.n_beating_baseline || 0)} beating baseline ·
            {' '}best ratio {Number(regressionVsBaseline.summary?.best_baseline_ratio || 0).toFixed(3)} ·
            {' '}best throughput {Math.round(Number(regressionVsBaseline.summary?.best_throughput_tok_s || 0))} tok/s
          </div>
        </div>
      )}

    </div>
  );
}

const DATA_COLUMNS = [
  ...COLUMNS,
  { key: '_actions', label: 'Actions', sortable: false },
];

const DATA_SORT_PREFS_KEY = 'dashboard.data_tab.sort.v1';

function hasGaps(d) {
  return d.best_loss_ratio == null || d.best_novelty_score == null || d.avg_throughput_tok_s == null;
}

function ExperimentDataTab({ onSelectExperiment, onRerunExperiment, onFillGapsExperiment, onStartExperiment }) {
  const [trends, setTrends] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortKey, setSortKey] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(DATA_SORT_PREFS_KEY) || '{}');
      const validKeys = new Set(DATA_COLUMNS.map((c) => c.key));
      if (typeof stored.sortKey === 'string' && validKeys.has(stored.sortKey)) return stored.sortKey;
    } catch {}
    return '_score';
  });
  const [sortDesc, setSortDesc] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(DATA_SORT_PREFS_KEY) || '{}');
      if (typeof stored.sortDesc === 'boolean') return stored.sortDesc;
    } catch {}
    return true;
  });
  const [filterQuery, setFilterQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  const [outcomeFilter, setOutcomeFilter] = useState('all');
  const [rerunningIds, setRerunningIds] = useState(new Set());
  const [copiedValue, copyText] = useCopyToClipboard();

  useEffect(() => {
    try {
      localStorage.setItem(DATA_SORT_PREFS_KEY, JSON.stringify({ sortKey, sortDesc }));
    } catch {}
  }, [sortKey, sortDesc]);

  useEffect(() => {
    let active = true;
    const fetchData = async () => {
      try {
        const payload = await apiService.getTrends();
        if (!active) return;
        setTrends(Array.isArray(payload?.trends) ? payload.trends : []);
        setError(null);
      } catch (e) {
        if (active) setError('Failed to load experiment data: ' + e.message);
      } finally {
        if (active) setLoading(false);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => { active = false; clearInterval(interval); };
  }, []);

  const handleSort = (key) => {
    if (key === '_actions') return;
    if (sortKey === key) {
      setSortDesc(!sortDesc);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  const handleRerun = async (experimentId) => {
    if (!experimentId || !onRerunExperiment) return; 
    setRerunningIds(prev => new Set(prev).add(experimentId));
    try {
      await onRerunExperiment(experimentId);
    } finally {
      setRerunningIds(prev => {
          const handleFillGaps = async (experimentId) => {
            if (!experimentId || !onFillGapsExperiment) return;
            setRerunningIds(prev => new Set(prev).add(experimentId));
            try {
              await onFillGapsExperiment(experimentId);
            } finally {
              setRerunningIds(prev => {
                const next = new Set(prev);
                next.delete(experimentId);
                return next;
              });
            }
          };
        const next = new Set(prev);
        next.delete(experimentId);
        return next;
      });
    }
  };

  const augmented = useMemo(() => {
    if (!trends) return [];
    return trends.map(d => ({ ...d, _score: trendScore(d) }));
  }, [trends]);

  const experimentTypes = useMemo(() => {
    const unique = Array.from(new Set(
      augmented.map((r) => r?.experiment_type).filter((v) => typeof v === 'string' && v.trim().length > 0)
    ));
    unique.sort((a, b) => a.localeCompare(b));
    return unique;
  }, [augmented]);

  const statusTypeOutcomeFiltered = useMemo(() => (
    augmented.filter((row) => {
      if (statusFilter !== 'all' && row.status !== statusFilter) return false;
      if (typeFilter !== 'all' && row.experiment_type !== typeFilter) return false;
      if (outcomeFilter === 'has_s1' && (row.n_stage1_passed || 0) <= 0) return false;
      if (outcomeFilter === 'no_s1' && (row.n_stage1_passed || 0) > 0) return false;
      return true;
    })
  ), [augmented, statusFilter, typeFilter, outcomeFilter]);

  const filtered = useMemo(() => (
    filterRowsByQuery(statusTypeOutcomeFiltered, filterQuery, [
      'experiment_id', 'hypothesis', 'experiment_type', 'status',
    ])
  ), [statusTypeOutcomeFiltered, filterQuery]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      let va = a[sortKey], vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [filtered, sortKey, sortDesc]);

  const hasActiveFilters = (
    filterQuery.trim().length > 0 ||
    statusFilter !== 'all' ||
    typeFilter !== 'all' ||
    outcomeFilter !== 'all'
  );

  const clearFilters = () => {
    setFilterQuery('');
    setStatusFilter('all');
    setTypeFilter('all');
    setOutcomeFilter('all');
  };

  if (loading) return <div className="card"><p style={{ color: 'var(--text-muted)' }}>Loading experiment data...</p></div>;
  if (error) return <div className="card"><p style={{ color: 'var(--accent-red)' }}>{error}</p></div>;
  if (!trends || trends.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Experiment Data</div>
        <p style={{ color: 'var(--text-muted)', fontSize: 13, marginBottom: 10 }}>
          No experiments yet. Run some experiments to populate this table with results.
        </p>
        {onStartExperiment && (
          <button
            className="refresh-btn"
            style={{ fontSize: 12, padding: '5px 14px' }}
            onClick={() => onStartExperiment({
              mode: 'continuous', n_cycles: 5, source: 'data_tab',
              auto_harden: true, preflight_override: true, enforce_preflight: true,
            })}
          >
            Run 5 Continuous Experiments
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Experiment Data</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Full experiment table with all metrics. Rows with missing key metrics (loss, novelty, throughput) are highlighted — use "Fill gaps" to re-evaluate them.
      </p>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
          aria-label="Data filter by status"
        >
          <option value="all">All status</option>
          <option value="completed">Completed</option>
          <option value="running">Running</option>
          <option value="failed">Failed</option>
        </select>
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
          aria-label="Data filter by type"
        >
          <option value="all">All types</option>
          {experimentTypes.map((type) => (
            <option key={type} value={type}>{type}</option>
          ))}
        </select>
        <select
          value={outcomeFilter}
          onChange={(e) => setOutcomeFilter(e.target.value)}
          style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
          aria-label="Data filter by outcome"
        >
          <option value="all">All outcomes</option>
          <option value="has_s1">Has S1 pass</option>
          <option value="no_s1">No S1 pass</option>
        </select>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Search experiments..."
          style={{
            fontSize: 11, padding: '4px 8px', borderRadius: 4,
            border: '1px solid var(--border)', background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)', minWidth: 160,
          }}
        />
        <button
          className="refresh-btn"
          style={{ fontSize: 11, padding: '3px 10px' }}
          onClick={clearFilters}
          disabled={!hasActiveFilters}
        >
          Clear filters
        </button>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
        Showing {sorted.length} of {augmented.length} experiments.
      </div>
      <div style={{ maxHeight: 600, overflowY: 'auto', border: '1px solid var(--border)', borderRadius: 6 }}>
        <table className="data-table" style={{ marginBottom: 0 }}>
          <thead>
            <tr>
              {DATA_COLUMNS.map(col => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  title={col.tooltip}
                  style={{
                    cursor: col.sortable === false ? 'default' : 'pointer',
                    userSelect: 'none',
                    whiteSpace: 'nowrap',
                    position: 'sticky',
                    top: 0,
                    background: 'var(--bg-secondary)',
                    zIndex: 1,
                  }}
                >
                  {col.label}
                  {sortKey === col.key && col.sortable !== false && (
                    <span style={{ marginLeft: 4, fontSize: 10 }}>
                      {sortDesc ? '\u25BC' : '\u25B2'}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((d, i) => {
              const gaps = hasGaps(d);
              const isRunning = d.status === 'running';
              const isRerunning = rerunningIds.has(d.experiment_id);
              return (
                <tr
                  key={d.experiment_id || i}
                  style={gaps ? { borderLeft: '2px solid var(--accent-yellow)' } : undefined}
                >
                  <td style={{ fontWeight: 600, color: scoreColor(d._score) }}>
                    <span title={`S1 rate ${(trendScoreBreakdown(d).passRate || 0).toFixed(1)}/35 | Loss ${(trendScoreBreakdown(d).loss || 0).toFixed(1)}/30 | Novelty ${(trendScoreBreakdown(d).novelty || 0).toFixed(1)}/25 | Efficiency ${(trendScoreBreakdown(d).efficiency || 0).toFixed(1)}/10`}>
                      {d._score}
                    </span>
                  </td>
                  <td style={{ fontFamily: 'monospace', fontSize: 12 }}>
                    <button
                      className="refresh-btn"
                      style={{ fontSize: 11, padding: '2px 6px', marginRight: 6 }}
                      onClick={() => onSelectExperiment && d.experiment_id && onSelectExperiment(d.experiment_id)}
                      disabled={!onSelectExperiment || !d.experiment_id}
                      aria-label={`Open experiment ${(d.experiment_id || '').slice(0, 12)}`}
                    >
                      {(d.experiment_id || '').slice(0, 12)}
                    </button>
                    {d.experiment_id && (
                      <button
                        className="refresh-btn"
                        style={{ fontSize: 10, padding: '1px 5px' }}
                        onClick={() => copyText(d.experiment_id)}
                        aria-label={`Copy experiment id ${d.experiment_id}`}
                      >
                        {copiedValue === d.experiment_id ? 'Copied' : 'Copy'}
                      </button>
                    )}
                  </td>
                  <td style={{
                    color: (d.s1_pass_rate || 0) > 0.05 ? 'var(--accent-green)' : 'var(--text-muted)',
                  }}>
                    {d.adjusted_s1_pass_rate != null
                      ? `${(d.adjusted_s1_pass_rate * 100).toFixed(1)}% adj`
                      : d.s1_pass_rate != null
                        ? `${(d.s1_pass_rate * 100).toFixed(1)}%`
                        : 'insufficient data'}
                    {d.s1_pass_rate != null && (
                      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                        raw {(d.s1_pass_rate * 100).toFixed(1)}% ({d.n_stage1_passed || 0}/{d.n_programs_generated || 0})
                      </div>
                    )}
                  </td>
                  <td>
                    <span style={{
                      color: d.trend_confidence === 'high'
                        ? 'var(--accent-green)'
                        : d.trend_confidence === 'medium'
                          ? 'var(--accent-yellow)'
                          : 'var(--accent-red)',
                      fontWeight: 600,
                      textTransform: 'uppercase',
                      fontSize: 10,
                    }}>
                      {d.trend_confidence || 'low'}
                    </span>
                    {d.s1_confidence_halfwidth != null && (
                      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                        ±{(d.s1_confidence_halfwidth * 100).toFixed(1)}%
                      </div>
                    )}
                    {d.trend_weight != null && (
                      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                        weight {(d.trend_weight * 100).toFixed(0)}%
                      </div>
                    )}
                  </td>
                  <td style={{ color: lossColor(d.best_loss_ratio) }}>
                    {metricText(d.best_loss_ratio, 'not computed', (v) => v.toFixed(4))}
                  </td>
                  <td style={{ color: noveltyColor(d.best_novelty_score) }}>
                    {metricText(d.best_novelty_score, 'not computed', (v) => v.toFixed(3))}
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>
                    {d.avg_throughput_tok_s != null
                      ? `${Math.round(d.avg_throughput_tok_s).toLocaleString()} tok/s`
                      : '—'}
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>
                    {d.avg_routing_token_retention != null
                      ? `${(d.avg_routing_token_retention * 100).toFixed(1)}%`
                      : '—'}
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>
                    {d.avg_routing_utilization_entropy != null
                      ? d.avg_routing_utilization_entropy.toFixed(3)
                      : '—'}
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>
                    {d.avg_depth_savings_ratio != null
                      ? `${(d.avg_depth_savings_ratio * 100).toFixed(1)}%`
                      : '—'}
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>
                    {d.avg_recursion_savings_ratio != null
                      ? `${(d.avg_recursion_savings_ratio * 100).toFixed(1)}%`
                      : '—'}
                  </td>
                  <td>{d.n_programs_generated || 0}</td>
                  <td style={{ color: (d.n_stage1_passed || 0) > 0 ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                    {d.n_stage1_passed || 0}
                  </td>
                  <td>{formatDuration(d.duration_seconds)}</td>
                  <td style={{ fontSize: 12, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                    {formatTime(d.timestamp)}
                  </td>
                  <td>
                    <button
                      className="refresh-btn"
                      style={{
                        fontSize: 10,
                        padding: '2px 8px',
                        whiteSpace: 'nowrap',
                        color: gaps ? 'var(--accent-yellow)' : undefined,
                        borderColor: gaps ? 'var(--accent-yellow)' : undefined,
                      }}
                      onClick={() => (gaps ? handleFillGaps(d.experiment_id) : handleRerun(d.experiment_id))}
                      disabled={isRunning || isRerunning || !d.experiment_id}
                    >
                      {isRerunning ? (gaps ? 'Filling...' : 'Starting...') : gaps ? 'Fill gaps' : 'Rerun'}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export { ExperimentDataTab };
export default TrendCharts;
