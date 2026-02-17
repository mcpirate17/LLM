import React, { useState, useEffect, useMemo } from 'react';
import { formatTime, formatDuration, scoreColor } from '../utils/format';
import { lossColor, noveltyColor } from '../utils/colors';
import useCopyToClipboard from '../hooks/useCopyToClipboard';

const API_BASE = process.env.REACT_APP_API_URL || '';

/**
 * TrendCharts — Cross-experiment line charts using inline SVG
 * plus a sortable data table with per-experiment scores.
 */

function MiniChart({ data, valueKey, label, color, formatValue, weightEvents, bandLowerKey, bandUpperKey }) {
  if (!data || data.length < 2) {
    return (
      <div style={{ textAlign: 'center', padding: 16, color: 'var(--text-muted)', fontSize: 13 }}>
        Need at least 2 experiments for {label} trend
      </div>
    );
  }

  const values = data.map(d => d[valueKey]).filter(v => v != null);
  if (values.length < 2) return null;

  const W = 400;
  const H = 120;
  const PAD = 24;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const tMin = data[0]?.timestamp || 0;
  const tMax = data[data.length - 1]?.timestamp || 1;
  const tRange = tMax - tMin || 1;

  const points = data
    .map((d, i) => {
      const v = d[valueKey];
      if (v == null) return null;
      const x = PAD + (i / (data.length - 1)) * (W - 2 * PAD);
      const y = H - PAD - ((v - min) / range) * (H - 2 * PAD);
      const lowerRaw = bandLowerKey ? d[bandLowerKey] : null;
      const upperRaw = bandUpperKey ? d[bandUpperKey] : null;
      const hasBand = lowerRaw != null && upperRaw != null;
      const lower = hasBand ? Math.min(Math.max(lowerRaw, min), max) : null;
      const upper = hasBand ? Math.min(Math.max(upperRaw, min), max) : null;
      return { x, y, v, idx: i, lower, upper, hasBand };
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
  const denom = n * sumXX - sumX * sumX;
  const slope = denom !== 0 ? (n * sumXY - sumX * sumY) / denom : 0;
  const intercept = (sumY - slope * sumX) / n;
  const regY0 = intercept;
  const regYN = intercept + slope * (data.length - 1);
  const regPx0 = H - PAD - ((Math.min(Math.max(regY0, min), max) - min) / range) * (H - 2 * PAD);
  const regPxN = H - PAD - ((Math.min(Math.max(regYN, min), max) - min) / range) * (H - 2 * PAD);

  // Compute weight event marker positions with before/after comparison
  const markers = (weightEvents || [])
    .filter(e => e.timestamp >= tMin && e.timestamp <= tMax)
    .map(e => {
      const x = PAD + ((e.timestamp - tMin) / tRange) * (W - 2 * PAD);
      // Find experiments before and after this weight event
      const eventIdx = data.findIndex(d => (d.timestamp || 0) >= e.timestamp);
      const before = data.slice(Math.max(0, eventIdx - 3), eventIdx)
        .map(d => d[valueKey]).filter(v => v != null);
      const after = data.slice(eventIdx, eventIdx + 3)
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
        style={{ width: '100%', height: 'auto', maxWidth: W }}>
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
        <line x1={PAD} y1={regPx0} x2={PAD + (data.length - 1) / (data.length - 1) * (W - 2 * PAD)} y2={regPxN}
          stroke={color} strokeWidth={1.5} strokeDasharray="6 3" opacity={0.5} />

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

/**
 * Score a trend data point (experiment) 0-100.
 * Weights: S1 pass rate (35%), best loss ratio (30%), best novelty (25%), efficiency (10%)
 */
function trendScore(d) {
  // S1 pass rate: scaled so 10% = max
  const stabilizedS1Rate = d.adjusted_s1_pass_rate != null
    ? d.adjusted_s1_pass_rate
    : (d.s1_pass_rate || 0);
  const passRate = Math.min(stabilizedS1Rate / 0.10, 1.0) * 35;

  // Loss ratio: lower is better
  const lossScore = d.best_loss_ratio != null
    ? Math.max(0, 1 - (d.best_loss_ratio - 0.2) / 0.8) * 30
    : 0;

  // Novelty
  const noveltyScore = d.best_novelty_score != null
    ? Math.min(d.best_novelty_score, 1.0) * 25
    : 0;

  // Efficiency: more programs per second = better, normalize to ~2 prog/s
  const efficiency = (d.duration_seconds && d.n_programs_generated)
    ? Math.min((d.n_programs_generated / d.duration_seconds) / 2, 1.0) * 10
    : 0;

  const reliabilityMultiplier = 0.5 + 0.5 * (d.trend_weight != null ? d.trend_weight : 1.0);
  return Math.round(Math.max(0, Math.min(100, (passRate + lossScore + noveltyScore + efficiency) * reliabilityMultiplier)));
}

function trendScoreBreakdown(d) {
  const stabilizedS1Rate = d.adjusted_s1_pass_rate != null
    ? d.adjusted_s1_pass_rate
    : (d.s1_pass_rate || 0);
  const passRate = Math.min(stabilizedS1Rate / 0.10, 1.0) * 35;
  const lossScore = d.best_loss_ratio != null
    ? Math.max(0, 1 - (d.best_loss_ratio - 0.2) / 0.8) * 30
    : 0;
  const noveltyScore = d.best_novelty_score != null
    ? Math.min(d.best_novelty_score, 1.0) * 25
    : 0;
  const efficiency = (d.duration_seconds && d.n_programs_generated)
    ? Math.min((d.n_programs_generated / d.duration_seconds) / 2, 1.0) * 10
    : 0;
  const reliabilityMultiplier = 0.5 + 0.5 * (d.trend_weight != null ? d.trend_weight : 1.0);
  return {
    passRate,
    loss: lossScore,
    novelty: noveltyScore,
    efficiency,
    reliabilityMultiplier,
  };
}

function metricText(value, fallbackReason, formatter) {
  if (value == null) return fallbackReason;
  return formatter(value);
}


const COLUMNS = [
  { key: '_score', label: 'Score' },
  { key: 'experiment_id', label: 'ID' },
  { key: 's1_pass_rate', label: 'S1 Rate' },
  { key: 'trend_confidence', label: 'Confidence' },
  { key: 'best_loss_ratio', label: 'Best Loss' },
  { key: 'best_novelty_score', label: 'Best Novelty' },
  { key: 'n_programs_generated', label: 'Programs' },
  { key: 'n_stage1_passed', label: 'S1 Pass' },
  { key: 'duration_seconds', label: 'Duration' },
  { key: 'timestamp', label: 'Time' },
];

const TRENDS_SORT_PREFS_KEY = 'dashboard.trends.sort.v1';

function TrendCharts({ onSelectExperiment }) {
  const [trends, setTrends] = useState(null);
  const [weightEvents, setWeightEvents] = useState([]);
  const [adaptationEvents, setAdaptationEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortKey, setSortKey] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(TRENDS_SORT_PREFS_KEY) || '{}');
      const validKeys = new Set(COLUMNS.map((column) => column.key));
      if (typeof stored.sortKey === 'string' && validKeys.has(stored.sortKey)) {
        return stored.sortKey;
      }
    } catch {}
    return '_score';
  });
  const [sortDesc, setSortDesc] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(TRENDS_SORT_PREFS_KEY) || '{}');
      if (typeof stored.sortDesc === 'boolean') {
        return stored.sortDesc;
      }
    } catch {}
    return true;
  });
  const [lastUpdated, setLastUpdated] = useState(null);
  const [copiedValue, copyText] = useCopyToClipboard();

  useEffect(() => {
    try {
      localStorage.setItem(TRENDS_SORT_PREFS_KEY, JSON.stringify({ sortKey, sortDesc }));
    } catch {}
  }, [sortKey, sortDesc]);

  useEffect(() => {
    let active = true;

    const fetchTrendContext = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/trends/context`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
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

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDesc(!sortDesc);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  const augmented = useMemo(() => {
    if (!trends) return [];
    return trends.map(d => ({ ...d, _score: trendScore(d) }));
  }, [trends]);

  const sorted = useMemo(() => {
    const arr = [...augmented];
    arr.sort((a, b) => {
      let va = a[sortKey], vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') {
        return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      }
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [augmented, sortKey, sortDesc]);

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
        return {
          ...event,
          verdict,
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
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20, marginBottom: 24 }}>
        <MiniChart
          data={trends}
          valueKey="adjusted_s1_pass_rate"
          label="Stage 1 Pass Rate"
          color="var(--accent-green, #3fb950)"
          formatValue={v => `${(v * 100).toFixed(1)}%`}
          weightEvents={weightEvents}
          bandLowerKey="s1_confidence_lower"
          bandUpperKey="s1_confidence_upper"
        />
        <MiniChart
          data={trends}
          valueKey="best_novelty_score"
          label="Best Novelty Score"
          color="var(--accent-purple, #bc8cff)"
          weightEvents={weightEvents}
        />
        <MiniChart
          data={trends}
          valueKey="best_loss_ratio"
          label="Best Loss Ratio"
          color="var(--accent-yellow, #d29922)"
          formatValue={v => v.toFixed(4)}
          weightEvents={weightEvents}
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
              </div>
            ))}
          </div>
        </div>
      )}
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12 }}>
        Stage-1 trend uses stabilized pass-rate estimates weighted by experiment size and mode; shaded region shows 95% confidence band.
      </div>

      {/* Data table */}
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
        Experiment Data
      </div>
      <table className="data-table">
        <thead>
          <tr>
            {COLUMNS.map(col => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}
              >
                {col.label}
                {sortKey === col.key && (
                  <span style={{ marginLeft: 4, fontSize: 10 }}>
                    {sortDesc ? '\u25BC' : '\u25B2'}
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((d, i) => (
            <tr key={d.experiment_id || i}>
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
              <td>{d.n_programs_generated || 0}</td>
              <td style={{ color: (d.n_stage1_passed || 0) > 0 ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                {d.n_stage1_passed || 0}
              </td>
              <td>{formatDuration(d.duration_seconds)}</td>
              <td style={{ fontSize: 12, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                {formatTime(d.timestamp)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default TrendCharts;
