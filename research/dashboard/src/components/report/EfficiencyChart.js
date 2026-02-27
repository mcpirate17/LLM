import React, { useMemo } from 'react';
import { CHART_DEFAULTS, clampToScale, getFixedScale } from '../../utils/chartScales';

export default function EfficiencyChart({ frontier, showLabels = false, labelCount = 5 }) {
  const frontierRows = Array.isArray(frontier) ? frontier : [];

  const W = 700, H = 260;
  const pad = { l: 60, r: 20, t: 20, b: 35 };

  const losses = frontierRows.map(p => p.final_loss || p.loss_ratio || 0).filter(l => isFinite(l));
  const flops = frontierRows.map(p => Math.log10(Math.max(p.flops_forward || p.param_count || 1, 1)));

  const lossDefaults = CHART_DEFAULTS.loss_ratio;
  const flopsDefaults = CHART_DEFAULTS.efficiency_log_flops;
  const lossScale = getFixedScale('efficiency.loss_ratio', losses, {
    defaultMin: lossDefaults.min,
    defaultMax: lossDefaults.max,
  });
  const flopsScale = getFixedScale('efficiency.log_flops', flops, {
    defaultMin: flopsDefaults.min,
    defaultMax: flopsDefaults.max,
  });
  const minL = lossScale.min;
  const maxL = lossScale.max;
  const minF = flopsScale.min;
  const maxF = flopsScale.max;
  const rangeL = maxL - minL || 1, rangeF = maxF - minF || 1;

  const xScale = v => pad.l + ((clampToScale(Math.log10(Math.max(v, 1)), flopsScale) - minF) / rangeF) * (W - pad.l - pad.r);
  const yScale = v => H - pad.b - ((clampToScale(v, lossScale) - minL) / rangeL) * (H - pad.t - pad.b);

  const labelCandidates = useMemo(() => {
    if (!showLabels) return [];
    return [...frontierRows]
      .filter(p => p.graph_fingerprint)
      .sort((a, b) => (a.final_loss || a.loss_ratio || 0) - (b.final_loss || b.loss_ratio || 0))
      .slice(0, labelCount);
  }, [frontierRows, showLabels, labelCount]);

  if (frontierRows.length === 0) return <p style={{ color: 'var(--text-muted)' }}>No Pareto-optimal programs yet.</p>;
  if (losses.length < 2 || flops.length < 2) return null;

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
      <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="var(--border)" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="var(--border)" />
      <text x={W / 2} y={H - 5} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>log10(FLOPs / Params)</text>
      <text x={12} y={H / 2} textAnchor="middle" fill="var(--text-muted)" fontSize={10} transform={`rotate(-90, 12, ${H / 2})`}>Loss</text>
      {frontierRows.map((p, i) => {
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
      {labelCandidates.map((p, i) => {
        const x = xScale(p.flops_forward || p.param_count || 0);
        const y = yScale(p.final_loss || p.loss_ratio || 0);
        if (!isFinite(x) || !isFinite(y)) return null;
        return (
          <text key={`label-${i}`} x={x + 8} y={y - 6} fontSize={9} fill="var(--text-secondary)">
            {(p.graph_fingerprint || '').slice(0, 8)}
          </text>
        );
      })}
    </svg>
  );
}
