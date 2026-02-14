import React, { useState, useEffect } from 'react';

const API_BASE = process.env.REACT_APP_API_URL || '';

/**
 * TrendCharts — Cross-experiment line charts using inline SVG.
 * Shows S1 pass rate, best novelty, best loss ratio over time.
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

function TrendCharts() {
  const [trends, setTrends] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/api/trends`)
      .then(r => r.json())
      .then(d => { setTrends(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <div className="card"><p style={{ color: 'var(--text-muted)' }}>Loading trends...</p></div>;
  if (!trends || trends.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Experiment Trends</div>
        <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
          No completed experiments yet. Trends will appear after 2+ experiments.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Experiment Trends</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
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
    </div>
  );
}

export default TrendCharts;
