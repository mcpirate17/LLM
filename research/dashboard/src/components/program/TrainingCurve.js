import React, { useState, useEffect } from 'react';
import apiService from '../../services/apiService';
import { CHART_DEFAULTS, clampToScale, getFixedScale } from '../../utils/chartScales';

export function TrainingCurve({ resultId }) {
  const [curve, setCurve] = useState([]);
  const MAX_POINTS = 500; // Cap points to prevent memory leaks

  useEffect(() => {
    apiService.getTrainingCurve(resultId)
      .then(d => {
        if (Array.isArray(d)) {
          // If the data is massive, downsample or take recent window
          const recentData = d.slice(-MAX_POINTS);
          setCurve(recentData);
        }
      })
      .catch(() => {});
  }, [resultId]);

  if (!curve || curve.length === 0) return null;

  const W = 350, H = 120;
  const pad = { l: 45, r: 10, t: 10, b: 25 };

  const losses = curve.map(c => c.loss).filter(l => l != null && isFinite(l));
  if (losses.length < 2) return null;

  const lossDefaults = CHART_DEFAULTS.training_loss;
  const lossScale = getFixedScale('training.loss', losses, {
    defaultMin: lossDefaults.min,
    defaultMax: lossDefaults.max,
  });
  const minL = lossScale.min;
  const maxL = lossScale.max;
  const rangeL = maxL - minL || 1;

  const denom = Math.max(1, MAX_POINTS - 1);
  const xScale = i => pad.l + (i / denom) * (W - pad.l - pad.r);
  const yScale = v => H - pad.b - ((clampToScale(v, lossScale) - minL) / rangeL) * (H - pad.t - pad.b);

  const pathD = losses.map((l, i) => `${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(l)}`).join(' ');

  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
        Training Curve
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="var(--border)" />
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="var(--border)" />
        <text x={pad.l - 5} y={yScale(maxL)} textAnchor="end" fill="var(--text-muted)" fontSize={9}>{maxL.toFixed(2)}</text>
        <text x={pad.l - 5} y={yScale(minL)} textAnchor="end" fill="var(--text-muted)" fontSize={9}>{minL.toFixed(2)}</text>
        <text x={W / 2} y={H - 3} textAnchor="middle" fill="var(--text-muted)" fontSize={9}>Step</text>
        <path d={pathD} fill="none" stroke="var(--accent-green)" strokeWidth={1.5} />
      </svg>
    </div>
  );
}

export default TrainingCurve;
