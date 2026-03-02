import React from 'react';
import { CHART_DEFAULTS, clampToScale, getFixedScale } from '../../utils/chartScales';

export function LearningTrajectory({ trajectory, onNavigateStrategy, onStartExperiment }) {
  const minimumExperiments = Math.max(2, Number(trajectory?.min_experiments_required) || 5);
  const windowSize = 30;

  if (!trajectory || trajectory.trend === 'insufficient_data') {
    const current = trajectory?.n_experiments || 0;
    const pct = minimumExperiments > 0 ? Math.min(100, Math.round((current / minimumExperiments) * 100)) : 0;
    return (
      <div className="card">
        <div className="card-title">Learning Trajectory</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
          Tracks the stage-1 survival rate across experiments. Need at least {minimumExperiments} experiments to compute a learning trajectory.
        </p>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8 }}>
          Progress: {current} of {minimumExperiments} experiments
        </div>
        <div style={{
          height: 6, borderRadius: 3,
          background: 'var(--bg-tertiary)',
          overflow: 'hidden',
          marginBottom: 8,
        }}>
          <div style={{
            height: '100%', borderRadius: 3,
            width: `${pct}%`,
            background: 'var(--accent-purple)',
            opacity: 0.6,
            transition: 'width 0.4s ease',
          }} />
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, textAlign: 'right' }}>
          {pct}% ({current}/{minimumExperiments})
        </div>
        {onStartExperiment && current < minimumExperiments && (
          <button
            className="refresh-btn"
            style={{ fontSize: 11, padding: '4px 10px' }}
            onClick={() => onStartExperiment({
              mode: 'continuous', n_cycles: Math.max(5, minimumExperiments - current),
              source: 'learning_trajectory', auto_harden: true,
              preflight_override: true, enforce_preflight: true,
            })}
          >
            Run {Math.max(5, minimumExperiments - current)} Experiments
          </button>
        )}
      </div>
    );
  }

  const trendColor = trajectory.trend === 'improving'
    ? 'var(--accent-green)'
    : trajectory.trend === 'declining'
      ? 'var(--accent-red)'
      : 'var(--accent-yellow)';

  const trendLabel = trajectory.trend === 'improving'
    ? 'Improving'
    : trajectory.trend === 'declining'
      ? 'Declining'
      : 'Plateaued';

  const points = (trajectory.points || []).slice(-windowSize);
  const W = 600, H = 200, pad = 40, padRight = 12, padTop = 12;

  let sparkline = null;
  if (points.length >= 2) {
    const rates = points.map(p => p.s1_rate);
    const rateDefaults = CHART_DEFAULTS.s1_rate;
    const rateScale = getFixedScale('learning.s1_rate', rates, {
      defaultMin: rateDefaults.min,
      defaultMax: rateDefaults.max,
    });
    const maxR = Math.max(rateScale.max, 0.01);
    const denom = Math.max(1, windowSize - 1);
    const step = (W - pad - padRight) / denom;
    const pts = rates.map((r, i) => {
      const x = pad + i * step;
      const clamped = clampToScale(r, rateScale);
      const y = H - pad - (clamped / maxR) * (H - pad - padTop);
      return `${x},${y}`;
    });

    // Grid lines (4 horizontal)
    const gridLines = [];
    const nGrid = 4;
    for (let g = 0; g <= nGrid; g++) {
      const val = (maxR * g) / nGrid;
      const gy = H - pad - (val / maxR) * (H - pad - padTop);
      gridLines.push(
        <g key={`grid-${g}`}>
          <line x1={pad} y1={gy} x2={W - padRight} y2={gy}
            stroke="var(--border)" strokeWidth={0.5} strokeDasharray={g === 0 ? 'none' : '4 2'} />
          <text x={pad - 4} y={gy + 3} textAnchor="end"
            fill="var(--text-muted)" fontSize={9}>
            {(val * 100).toFixed(1)}%
          </text>
        </g>
      );
    }

    // X-axis labels (every ~5th experiment)
    const xLabels = [];
    const labelEvery = Math.max(1, Math.floor(windowSize / 8));
    for (let i = 0; i < points.length; i += labelEvery) {
      const x = pad + i * step;
      xLabels.push(
        <text key={`x-${i}`} x={x} y={H - pad + 14} textAnchor="middle"
          fill="var(--text-muted)" fontSize={9}>
          #{i + 1}
        </text>
      );
    }

    // Regression line
    const slope = trajectory.slope || 0;
    const meanY = trajectory.overall_s1_rate || 0;
    const midIdx = (points.length - 1) / 2;
    const regStart = Math.max(0, meanY - slope * midIdx);
    const regEnd = meanY + slope * (points.length - 1 - midIdx);
    const regY1 = H - pad - (Math.min(Math.max(regStart, 0), maxR) / maxR) * (H - pad - padTop);
    const regY2 = H - pad - (Math.min(Math.max(regEnd, 0), maxR) / maxR) * (H - pad - padTop);

    sparkline = (
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', maxWidth: 700 }}>
        {gridLines}
        {xLabels}
        <line x1={pad} y1={regY1} x2={pad + (points.length - 1) * step} y2={regY2}
          stroke={trendColor} strokeWidth={1.5} strokeDasharray="6 3" opacity={0.6} />
        <polyline points={pts.join(' ')} fill="none" stroke={trendColor} strokeWidth={2} />
        {pts.map((pt, i) => {
          const [x, y] = pt.split(',');
          return (
            <circle key={i} cx={x} cy={y} r={3} fill={trendColor}
              style={{ cursor: 'default' }}>
              <title>Exp #{i + 1}: {(rates[i] * 100).toFixed(1)}% S1 rate</title>
            </circle>
          );
        })}
        <text x={W / 2} y={H - 2} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>
          Experiment #
        </text>
        <text x={8} y={(H - pad) / 2 + padTop} textAnchor="middle"
          fill="var(--text-muted)" fontSize={10}
          transform={`rotate(-90, 8, ${(H - pad) / 2 + padTop})`}>
          S1 Rate
        </text>
      </svg>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Learning Trajectory</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Tracks the stage-1 survival rate across recent experiments to show whether the
        AI scientist's search strategy is getting better at finding architectures that learn.
      </p>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10, flexWrap: 'wrap' }}>
        <span style={{
          fontSize: 14, fontWeight: 700, color: trendColor,
          padding: '2px 10px', borderRadius: 12,
          background: trajectory.trend === 'improving'
            ? 'rgba(63,185,80,0.15)'
            : trajectory.trend === 'declining'
              ? 'rgba(248,81,73,0.15)'
              : 'rgba(210,153,34,0.15)',
        }}>
          {trendLabel}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          Recent S1 rate: {((trajectory.recent_s1_rate || 0) * 100).toFixed(1)}%
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Slope: {(trajectory.slope || 0) > 0 ? '+' : ''}{((trajectory.slope || 0) * 100).toFixed(2)}%/exp
        </span>
      </div>
      {trajectory.trend === 'plateaued' && (
        <div style={{
          marginBottom: 10,
          padding: '10px 12px',
          borderRadius: 6,
          border: '1px solid var(--border)',
          borderLeft: '3px solid var(--accent-purple)',
          background: 'var(--bg-tertiary)',
          fontSize: 12,
          color: 'var(--text-secondary)',
          lineHeight: 1.5,
        }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent-purple)', marginBottom: 4, textTransform: 'uppercase' }}>
            Aria's Analysis
          </div>
          <div>
            Search productivity has <strong>plateaued</strong>. Aria will likely recommend <strong>Novelty Search</strong> to escape this local minimum.
          </div>
          <div style={{ marginTop: 6 }}>
            <button
              className="refresh-btn"
              style={{ fontSize: 11, padding: '2px 8px' }}
              onClick={onNavigateStrategy}
            >
              See Strategy Advisor &rarr;
            </button>
          </div>
        </div>
      )}
      {sparkline}
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
        <span>{points.length} experiments</span>
        <span>Overall S1: {((trajectory.overall_s1_rate || 0) * 100).toFixed(1)}%</span>
        {trajectory.weight_adjustments != null && (
          <span>{trajectory.weight_adjustments} weight adjustments</span>
        )}
      </div>
    </div>
  );
}

export default LearningTrajectory;
