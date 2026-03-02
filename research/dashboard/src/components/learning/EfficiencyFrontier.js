import React, { useState } from 'react';
import { CHART_DEFAULTS, clampToScale, getFixedScale } from '../../utils/chartScales';

export function EfficiencyFrontier({ frontier }) {
  const [hover, setHover] = useState(null);

  if (!frontier || frontier.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Efficiency Frontier</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          Need Stage 1 survivors with FLOP data to compute frontier.
        </p>
      </div>
    );
  }

  // Simple scatter plot using SVG
  const W = 400, H = 200;
  const pad = 40;

  const losses = frontier.map(p => p.final_loss);
  const flops = frontier.map(p => Math.log10(Math.max(p.flops_forward, 1)));
  const lossDefaults = CHART_DEFAULTS.loss_ratio;
  const flopsDefaults = CHART_DEFAULTS.efficiency_log_flops;
  const lossScale = getFixedScale('learning.loss_ratio', losses, {
    defaultMin: lossDefaults.min,
    defaultMax: lossDefaults.max,
  });
  const flopsScale = getFixedScale('learning.efficiency_log_flops', flops, {
    defaultMin: flopsDefaults.min,
    defaultMax: flopsDefaults.max,
  });
  const minLoss = lossScale.min;
  const maxLoss = lossScale.max;
  const minFlops = flopsScale.min;
  const maxFlops = flopsScale.max;
  const rangeL = maxLoss - minLoss || 1;
  const rangeF = maxFlops - minFlops || 1;

  const points = frontier.map((p, i) => ({
    x: pad + ((clampToScale(flops[i], flopsScale) - minFlops) / rangeF) * (W - 2 * pad),
    y: H - pad - ((clampToScale(losses[i], lossScale) - minLoss) / rangeL) * (H - 2 * pad),
    label: p.graph_fingerprint?.slice(0, 8),
    novelty: p.novelty_score || 0,
    data: p,
    idx: i,
  }));

  return (
    <div className="card" style={{ position: 'relative' }}>
      <div className="card-title">Efficiency Frontier ({frontier.length} Pareto-optimal)</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
        Architectures that are the best trade-off between compute cost (FLOPs) and learning
        quality (loss). Points on the frontier can't be beaten on both axes simultaneously —
        these are the most promising candidates for scaling up.
      </p>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}
        onMouseLeave={() => setHover(null)}>
        {/* Axes */}
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="var(--border)" />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="var(--border)" />
        <text x={W / 2} y={H - 5} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>log10(FLOPs)</text>
        <text x={10} y={H / 2} textAnchor="middle" fill="var(--text-muted)" fontSize={10}
          transform={`rotate(-90, 10, ${H / 2})`}>Loss</text>

        {/* Frontier line */}
        {points.length > 1 && (
          <polyline
            points={[...points].sort((a, b) => a.x - b.x).map(p => `${p.x},${p.y}`).join(' ')}
            fill="none" stroke="var(--accent-purple)" strokeWidth={1.5} strokeDasharray="4 2"
          />
        )}

        {/* Points */}
        {points.map((p, i) => (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r={hover?.idx === i ? 7 : 5}
              fill={`rgba(188, 140, 255, ${0.3 + p.novelty * 0.7})`}
              stroke={hover?.idx === i ? 'var(--accent-blue)' : 'var(--accent-purple)'}
              strokeWidth={hover?.idx === i ? 2.5 : 1.5}
              style={{ cursor: 'pointer' }}
              onMouseEnter={() => setHover(p)}
              onMouseLeave={() => setHover(null)} />
          </g>
        ))}
      </svg>

      {/* Hover card */}
      {hover && (
        <div style={{
          position: 'absolute',
          top: 60,
          right: 12,
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          padding: '10px 14px',
          fontSize: 12,
          lineHeight: 1.6,
          zIndex: 10,
          minWidth: 200,
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
        }}>
          <div style={{ fontWeight: 600, color: 'var(--accent-purple)', marginBottom: 4 }}>
            {hover.label || 'Unknown'}
          </div>
          <div><span style={{ color: 'var(--text-muted)' }}>Loss:</span> {hover.data.final_loss?.toFixed(4)}</div>
          <div><span style={{ color: 'var(--text-muted)' }}>FLOPs:</span> {hover.data.flops_forward?.toLocaleString()}</div>
          <div><span style={{ color: 'var(--text-muted)' }}>Params:</span> {hover.data.param_count?.toLocaleString()}</div>
          <div><span style={{ color: 'var(--text-muted)' }}>Novelty:</span> {(hover.data.novelty_score || 0).toFixed(3)}</div>
          {hover.data.ops && hover.data.ops.length > 0 && (
            <div style={{ marginTop: 4 }}>
              <span style={{ color: 'var(--text-muted)' }}>Ops:</span>{' '}
              <span style={{ fontFamily: 'monospace', color: 'var(--accent-blue)', fontSize: 11 }}>
                {hover.data.ops.join(', ')}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default EfficiencyFrontier;
