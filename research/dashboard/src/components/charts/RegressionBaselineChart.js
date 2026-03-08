import React from 'react';
import { CHART_DEFAULTS, clampToScale, getFixedScale } from '../../utils/chartScales';

export function RegressionBaselineChart({ points, frontier }) {
  if (!Array.isArray(points) || points.length === 0) return null;
  const W = 440;
  const H = 180;
  const PAD = 28;
  const validPoints = points.filter((p) => {
    const x = Number(p?.throughput_tok_s);
    const y = Number(p?.baseline_loss_ratio);
    return Number.isFinite(x) && Number.isFinite(y) && y > 0;
  });
  if (validPoints.length === 0) {
    return (
      <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
        No baseline-ratio points to plot yet. Baseline comparison is only available after baseline matching succeeds.
      </div>
    );
  }

  const xs = validPoints.map((p) => Number(p.throughput_tok_s));
  const ys = validPoints.map((p) => Number(p.baseline_loss_ratio));
  const xDefaults = CHART_DEFAULTS.throughput_tok_s;
  const yDefaults = CHART_DEFAULTS.baseline_ratio;
  const xScale = getFixedScale('trend.throughput_tok_s', xs, {
    defaultMin: xDefaults.min,
    defaultMax: xDefaults.max,
  });
  const yScale = getFixedScale('trend.baseline_ratio', ys, {
    defaultMin: yDefaults.min,
    defaultMax: yDefaults.max,
  });
  const xMin = xScale.min;
  const xMax = xScale.max;
  const yMin = yScale.min;
  const yMax = yScale.max;
  const xRange = (xMax - xMin) || 1;
  const yRange = (yMax - yMin) || 1;
  const ySpread = Math.max(...ys) - Math.min(...ys);
  const applyJitter = ySpread < 0.01;
  const frontierColor = 'var(--accent-purple, #00d4ff)';

  const project = (x, y) => ({
    x: PAD + ((clampToScale(x, xScale) - xMin) / xRange) * (W - PAD * 2),
    y: H - PAD - ((clampToScale(y, yScale) - yMin) / yRange) * (H - PAD * 2),
  });
  const frontierPath = (frontier || [])
    .map((p, i) => {
      const pt = project(Number(p.throughput_tok_s || 0), Number(p.baseline_loss_ratio || 0));
      return `${i === 0 ? 'M' : 'L'} ${pt.x} ${pt.y}`;
    })
    .join(' ');

  const stableJitter = (seed, amplitude = 0.003) => {
    let h = 0;
    const s = String(seed || '');
    for (let i = 0; i < s.length; i += 1) {
      h = ((h << 5) - h) + s.charCodeAt(i);
      h |= 0;
    }
    return ((Math.abs(h) % 1000) / 1000 - 0.5) * 2 * amplitude;
  };

  const histBins = 12;
  const histogram = Array.from({ length: histBins }, () => 0);
  for (const y of ys) {
    const norm = (clampToScale(y, yScale) - yMin) / yRange;
    const idx = Math.max(0, Math.min(histBins - 1, Math.floor(norm * histBins)));
    histogram[idx] += 1;
  }
  const histMax = Math.max(...histogram, 1);
  const histX = W - PAD - 44;
  const histW = 40;

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', maxWidth: W }}>
      <defs>
        <filter id="paretoGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx="0" dy="0" stdDeviation="1.4" floodColor={frontierColor} floodOpacity="0.85" />
        </filter>
      </defs>
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--border)" strokeWidth={1} />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="var(--border)" strokeWidth={1} />
      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} fill="var(--text-muted)">Throughput (tok/s)</text>
      <text x={8} y={H / 2} transform={`rotate(-90 8 ${H / 2})`} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
        Baseline Ratio (lower is better)
      </text>
      {validPoints.map((p, idx) => {
        const baseY = Number(p.baseline_loss_ratio);
        const yValue = applyJitter ? baseY + stableJitter(p.result_id || idx) : baseY;
        const pt = project(Number(p.throughput_tok_s), yValue);
        const beats = baseY < 1.0;
        return (
          <circle
            key={`${p.result_id || idx}`}
            cx={pt.x}
            cy={pt.y}
            r={3}
            fill={beats ? 'var(--accent-green, #3fb950)' : 'var(--accent-yellow, #d29922)'}
            opacity={0.85}
          >
            <title>
              {`${(p.result_id || '').slice(0, 12)} | baseline=${Number(p.baseline_loss_ratio || 0).toFixed(3)} | throughput=${Math.round(Number(p.throughput_tok_s || 0))} tok/s`}
            </title>
          </circle>
        );
      })}
      {frontierPath && (
        <path
          d={frontierPath}
          fill="none"
          stroke={frontierColor}
          strokeWidth={2.6}
          strokeDasharray="6 4"
          filter="url(#paretoGlow)"
        />
      )}
      <g>
        <rect
          x={histX - 2}
          y={PAD - 2}
          width={histW + 4}
          height={H - (PAD * 2) + 4}
          fill="rgba(255,255,255,0.02)"
          stroke="var(--border)"
          strokeWidth={0.7}
          rx={2}
        />
        {histogram.map((count, i) => {
          if (count <= 0) return null;
          const binH = (H - PAD * 2) / histBins;
          const y = H - PAD - (i + 1) * binH;
          const w = (count / histMax) * (histW - 4);
          return (
            <rect
              key={`hist-${i}`}
              x={histX}
              y={y + 1}
              width={w}
              height={Math.max(1, binH - 2)}
              fill="var(--accent-green, #3fb950)"
              opacity={0.7}
            />
          );
        })}
        <text x={histX + histW / 2} y={H - 8} textAnchor="middle" fontSize={8.5} fill="var(--text-muted)">
          Y dist
        </text>
      </g>
    </svg>
  );
}

export default RegressionBaselineChart;
