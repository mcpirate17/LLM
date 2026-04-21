import React from 'react';
import { splitCurveIntoSegments } from './utils';

export function MiniNoveltyChart({ points, label = '', width = 600 }) {
  if (!Array.isArray(points) || points.length === 0) return null;
  const W = width;
  const H = 113;
  const pad = { l: 8, r: 8, t: 8, b: 14 };
  const maxGeneration = Math.max(...points.map((point) => Number(point.generation) || 0), 1);
  const metricValues = points.flatMap((point) => [Number(point.best_fitness) || 0, Number(point.best_novelty) || 0]);
  const minValue = Math.min(...metricValues);
  const maxValue = Math.max(...metricValues);
  const rangeValue = maxValue - minValue || 1;
  const xScale = (generation) => pad.l + ((Number(generation) || 0) / Math.max(maxGeneration, 1)) * (W - pad.l - pad.r);
  const yScale = (value) => H - pad.b - (((Number(value) || 0) - minValue) / rangeValue) * (H - pad.t - pad.b);
  const buildPath = (key) => points.map((point, idx) => `${idx === 0 ? 'M' : 'L'} ${xScale(point.generation).toFixed(1)} ${yScale(point[key]).toFixed(1)}`).join(' ');
  const latest = points[points.length - 1];

  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 8,
      padding: '4px 8px', borderRadius: 6,
      background: 'rgba(63,185,80,0.08)', border: '1px solid rgba(63,185,80,0.2)',
      marginBottom: 4,
    }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
        <rect width={W} height={H} rx={4} fill="rgba(0,0,0,0.3)" />
        <path d={buildPath('best_fitness')} fill="none" stroke="var(--accent-green)" strokeWidth={1.5} />
        <path d={buildPath('best_novelty')} fill="none" stroke="var(--accent-yellow)" strokeWidth={1.5} />
        {points.map((point) => (
          <line
            key={point.generation}
            x1={xScale(point.generation)}
            y1={H - pad.b}
            x2={xScale(point.generation)}
            y2={H - pad.b + 4}
            stroke="rgba(255,255,255,0.25)"
          />
        ))}
        <text x={pad.l} y={pad.t + 10} fill="var(--text-muted)" fontSize="9" fontFamily="monospace">best_fit</text>
        <text x={pad.l + 46} y={pad.t + 10} fill="var(--accent-yellow)" fontSize="9" fontFamily="monospace">best_novelty</text>
      </svg>
      <div style={{ fontSize: 10, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
        {label && <div style={{ color: 'var(--text-muted)', marginBottom: 4, maxWidth: 220 }}>{label}</div>}
        <div style={{ color: 'var(--accent-green)', fontWeight: 700, fontSize: 12, fontFamily: 'monospace' }}>
          fit {Number(latest.best_fitness || 0).toFixed(3)}
        </div>
        <div style={{ color: 'var(--accent-yellow)', fontFamily: 'monospace' }}>
          nov {Number(latest.best_novelty || 0).toFixed(3)}
        </div>
        <div>gen {latest.generation}/{latest.total_generations}</div>
        <div>archive {latest.archive_size}</div>
      </div>
    </div>
  );
}

export function MiniLossChart({
  curve,
  statusText = '',
  statusTone = 'info',
  label = '',
  segmentLabelPrefix = 'run',
  width = 600,
}) {
  if (!curve || curve.length < 2) return null;
  const W = width;
  const H = 113;
  const pad = { l: 4, r: 4, t: 4, b: 4 };
  const segments = splitCurveIntoSegments(curve);

  const losses = curve.map((p) => p.loss);
  const minL = Math.min(...losses);
  const maxL = Math.max(...losses);
  const rangeL = maxL - minL || 1;

  const xScale = (i) => pad.l + (i / Math.max(curve.length - 1, 1)) * (W - pad.l - pad.r);
  const yScale = (v) => H - pad.b - ((v - minL) / rangeL) * (H - pad.t - pad.b);

  let pointOffset = 0;
  const segmentPaths = segments.map((segment) => {
    const startOffset = pointOffset;
    pointOffset += segment.length;
    const pathD = segment
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xScale(startOffset + i).toFixed(1)} ${yScale(p.loss).toFixed(1)}`)
      .join(' ');
    return { pathD, startOffset };
  });
  const currentLoss = losses[losses.length - 1];
  const currentStep = curve[curve.length - 1].step;
  const totalSteps = curve[curve.length - 1].total_steps;
  const phase = curve[curve.length - 1].phase || '';
  const statusColor = statusTone === 'warn'
    ? 'var(--accent-yellow)'
    : statusTone === 'success'
      ? 'var(--accent-green)'
      : 'var(--text-secondary)';

  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 8,
      padding: '4px 8px', borderRadius: 6,
      background: 'rgba(63,185,80,0.08)', border: '1px solid rgba(63,185,80,0.2)',
      marginBottom: 4,
    }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
        <rect width={W} height={H} rx={4} fill="rgba(0,0,0,0.3)" />
        {segmentPaths.map((segment, idx) => (
          <g key={`${segment.startOffset}-${idx}`}>
            <path d={segment.pathD} fill="none" stroke="var(--accent-green)" strokeWidth={1.5} />
            {idx < segmentPaths.length - 1 && (
              <line
                x1={xScale(segment.startOffset + Math.max(0, segments[idx].length - 1))}
                y1={pad.t}
                x2={xScale(segment.startOffset + Math.max(0, segments[idx].length - 1))}
                y2={H - pad.b}
                stroke="rgba(255,255,255,0.14)"
                strokeDasharray="3 3"
              />
            )}
            <text
              x={xScale(segment.startOffset) + 6}
              y={pad.t + 12}
              fill="var(--text-muted)"
              fontSize="9"
              fontFamily="monospace"
            >
              {`${segmentLabelPrefix} ${idx + 1}`}
            </text>
          </g>
        ))}
      </svg>
      <div style={{ fontSize: 10, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
        {label && <div style={{ color: 'var(--text-muted)', marginBottom: 4, maxWidth: 220 }}>{label}</div>}
        <div style={{ color: 'var(--accent-green)', fontWeight: 700, fontSize: 12, fontFamily: 'monospace' }}>
          {currentLoss < 0.0001 && currentLoss !== 0 ? currentLoss.toExponential(2) : currentLoss.toFixed(4)}
        </div>
        <div>step {currentStep}/{totalSteps}</div>
        {phase && <div style={{ textTransform: 'capitalize', color: 'var(--text-muted)' }}>{phase}</div>}
        {statusText && <div style={{ color: statusColor, maxWidth: 220 }}>{statusText}</div>}
      </div>
    </div>
  );
}
