import React from 'react';
import { CHART_DEFAULTS, clampToScale, getFixedScale } from '../../utils/chartScales';
import ChartActions from '../ChartActions';

export const TREND_CHART_WINDOW = 30;

export function MiniChart({ data, valueKey, label, color, formatValue, weightEvents, bandLowerKey, bandUpperKey, scaleKey, windowSize = TREND_CHART_WINDOW, onSelectExperiment }) {
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
              ` | Before: ${fmt(m.avgBefore)} &rarr; After: ${fmt(m.avgAfter)} (${m.delta > 0 ? '+' : ''}${fmt(m.delta)})` : ''}</title>
          </g>
        ))}

        {/* Line */}
        <path d={pathD} fill="none" stroke={color} strokeWidth={2} />

        {/* Dots */}
        {points.map((p, i) => {
          const expId = windowed[p.idx]?.experiment_id;
          return (
            <circle key={i} cx={p.x} cy={p.y} r={3}
              fill={color} stroke="var(--bg-secondary, #161b22)" strokeWidth={1.5}
              style={{ cursor: onSelectExperiment && expId ? 'pointer' : 'default' }}
              onClick={() => onSelectExperiment && expId && onSelectExperiment(expId)}>
              <title>Exp #${p.idx + 1}: ${fmt(p.v)}{onSelectExperiment && expId ? ' &mdash; click to view' : ''}</title>
            </circle>
          );
        })}
      </svg>
      {(() => {
        let insightText = null;
        let insightColor = 'var(--text-muted)';
        
        if (drawReg) {
          const upIsGood = valueKey.includes('pass_rate') || valueKey.includes('novelty') || valueKey.includes('throughput') || valueKey.includes('retention') || valueKey.includes('savings');
          const recentValues = points.slice(-3).map(p => p.v);
          const recentAvg = recentValues.length > 0 ? recentValues.reduce((a, b) => a + b, 0) / recentValues.length : null;
          const trendAvgAtEnd = regYN;
          const isFlat = Math.abs(slope) <= 0.001;
          const isImproving = upIsGood ? slope > 0.001 : slope < -0.001;
          
          let isDiverging = false;
          if (recentAvg !== null) {
              const diff = upIsGood ? trendAvgAtEnd - recentAvg : recentAvg - trendAvgAtEnd;
              if (diff > range * 0.15) isDiverging = true;
          }

          if (isDiverging) {
             insightText = 'Search quality dropping \u2014 consider grammar reset';
             insightColor = 'var(--accent-red, #f85149)';
          } else if (isFlat) {
             if (valueKey.includes('pass_rate')) {
                 insightText = 'Search space exhausted \u2014 increase mutation rate';
             } else if (valueKey.includes('loss')) {
                 insightText = 'Performance floor reached \u2014 increase training steps';
             } else {
                 insightText = `Plateau reached for ${windowed.length} experiments`;
             }
             insightColor = 'var(--accent-yellow, #d29922)';
          } else if (isImproving) {
             insightText = 'Improving \u2014 current strategy working';
             insightColor = 'var(--accent-green, #3fb950)';
          } else {
             insightText = 'Degrading \u2014 consider strategy change';
             insightColor = 'var(--accent-red, #f85149)';
          }
        }
        
        return (
          <ChartActions
            insight={insightText}
            insightColor={insightColor}
          />
        );
      })()}
    </div>
  );
}

export default MiniChart;
