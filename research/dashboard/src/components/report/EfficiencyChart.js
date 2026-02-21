import React from 'react';

export default function EfficiencyChart({ frontier }) {
  if (!frontier || frontier.length === 0) return <p style={{ color: 'var(--text-muted)' }}>No Pareto-optimal programs yet.</p>;

  const W = 500, H = 200;
  const pad = { l: 60, r: 20, t: 20, b: 35 };

  const losses = frontier.map(p => p.final_loss || p.loss_ratio || 0).filter(l => isFinite(l));
  const flops = frontier.map(p => p.flops_forward || p.param_count || 0).filter(f => f > 0);
  if (losses.length < 2 || flops.length < 2) return null;

  const minL = Math.min(...losses), maxL = Math.max(...losses);
  const minF = Math.min(...flops), maxF = Math.max(...flops);
  const rangeL = maxL - minL || 1, rangeF = maxF - minF || 1;

  const xScale = v => pad.l + ((v - minF) / rangeF) * (W - pad.l - pad.r);
  const yScale = v => H - pad.b - ((v - minL) / rangeL) * (H - pad.t - pad.b);

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
      <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="var(--border)" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="var(--border)" />
      <text x={W / 2} y={H - 5} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>FLOPs / Params</text>
      <text x={12} y={H / 2} textAnchor="middle" fill="var(--text-muted)" fontSize={10} transform={`rotate(-90, 12, ${H / 2})`}>Loss</text>
      {frontier.map((p, i) => {
        const x = xScale(p.flops_forward || p.param_count || 0);
        const y = yScale(p.final_loss || p.loss_ratio || 0);
        if (!isFinite(x) || !isFinite(y)) return null;
        return (
          <circle key={i} cx={x} cy={y} r={5}
            fill="var(--accent-purple)" opacity={0.7}
            stroke="var(--bg-secondary)" strokeWidth={1.5}>
            <title>{p.graph_fingerprint?.slice(0, 10)}: loss={p.final_loss || p.loss_ratio}</title>
          </circle>
        );
      })}
    </svg>
  );
}
