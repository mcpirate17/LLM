import React, { useState, useEffect, useMemo } from 'react';

const API_BASE = process.env.REACT_APP_API_URL || '';

/**
 * TrendCharts — Cross-experiment line charts using inline SVG
 * plus a sortable data table with per-experiment scores.
 */

function MiniChart({ data, valueKey, label, color, formatValue }) {
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

  const points = data
    .map((d, i) => {
      const v = d[valueKey];
      if (v == null) return null;
      const x = PAD + (i / (data.length - 1)) * (W - 2 * PAD);
      const y = H - PAD - ((v - min) / range) * (H - 2 * PAD);
      return { x, y, v };
    })
    .filter(Boolean);

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

  const fmt = formatValue || (v => v.toFixed(3));

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

        {/* Line */}
        <path d={pathD} fill="none" stroke={color} strokeWidth={2} />

        {/* Dots */}
        {points.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={3}
            fill={color} stroke="var(--bg-secondary, #161b22)" strokeWidth={1.5}>
            <title>{fmt(p.v)}</title>
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
  const passRate = Math.min((d.s1_pass_rate || 0) / 0.10, 1.0) * 35;

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

  return Math.round(Math.max(0, Math.min(100, passRate + lossScore + noveltyScore + efficiency)));
}

function scoreColor(score) {
  if (score >= 70) return 'var(--accent-green)';
  if (score >= 40) return 'var(--accent-yellow)';
  if (score >= 20) return 'var(--accent-orange, #f0883e)';
  return 'var(--accent-red)';
}

function formatTime(timestamp) {
  if (!timestamp) return '--';
  return new Date(timestamp * 1000).toLocaleString();
}

function formatDuration(seconds) {
  if (!seconds) return '--';
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

const COLUMNS = [
  { key: '_score', label: 'Score' },
  { key: 'experiment_id', label: 'ID' },
  { key: 's1_pass_rate', label: 'S1 Rate' },
  { key: 'best_loss_ratio', label: 'Best Loss' },
  { key: 'best_novelty_score', label: 'Best Novelty' },
  { key: 'n_programs_generated', label: 'Programs' },
  { key: 'n_stage1_passed', label: 'S1 Pass' },
  { key: 'duration_seconds', label: 'Duration' },
  { key: 'timestamp', label: 'Time' },
];

function TrendCharts() {
  const [trends, setTrends] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortKey, setSortKey] = useState('_score');
  const [sortDesc, setSortDesc] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/api/trends`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(d => { setTrends(d); setLoading(false); })
      .catch(e => { setError('Failed to load trends: ' + e.message); setLoading(false); });
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
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        How the search is improving over time. Rising S1 pass rate means the grammar is learning
        to generate better architectures. Decreasing loss ratio means the survivors are learning
        faster. These trends show whether the system's self-improvement loop is working.
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20, marginBottom: 24 }}>
        <MiniChart
          data={trends}
          valueKey="s1_pass_rate"
          label="Stage 1 Pass Rate"
          color="var(--accent-green, #3fb950)"
          formatValue={v => `${(v * 100).toFixed(1)}%`}
        />
        <MiniChart
          data={trends}
          valueKey="best_novelty_score"
          label="Best Novelty Score"
          color="var(--accent-purple, #bc8cff)"
        />
        <MiniChart
          data={trends}
          valueKey="best_loss_ratio"
          label="Best Loss Ratio"
          color="var(--accent-yellow, #d29922)"
          formatValue={v => v.toFixed(4)}
        />
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
                {d._score}
              </td>
              <td style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--accent-blue)' }}>
                {(d.experiment_id || '').slice(0, 12)}
              </td>
              <td style={{
                color: (d.s1_pass_rate || 0) > 0.05 ? 'var(--accent-green)' : 'var(--text-muted)',
              }}>
                {d.s1_pass_rate != null ? `${(d.s1_pass_rate * 100).toFixed(1)}%` : '--'}
              </td>
              <td style={{
                color: d.best_loss_ratio != null
                  ? (d.best_loss_ratio < 0.5 ? 'var(--accent-green)' : d.best_loss_ratio < 0.8 ? 'var(--accent-yellow)' : 'var(--text-muted)')
                  : 'var(--text-muted)',
              }}>
                {d.best_loss_ratio?.toFixed(4) || '--'}
              </td>
              <td style={{
                color: d.best_novelty_score != null
                  ? (d.best_novelty_score > 0.8 ? 'var(--accent-green)' : d.best_novelty_score > 0.5 ? 'var(--accent-yellow)' : 'var(--text-muted)')
                  : 'var(--text-muted)',
              }}>
                {d.best_novelty_score?.toFixed(3) || '--'}
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
