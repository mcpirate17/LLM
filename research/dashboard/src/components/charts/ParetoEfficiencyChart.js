import React, { useMemo } from 'react';
import { CHART_DEFAULTS, clampToScale, getFixedScale } from '../../utils/chartScales';

export function ParetoEfficiencyChart({ points, frontier }) {
  if (!Array.isArray(points) || points.length === 0) return null;
  const W = 440;
  const H = 220;
  const PAD = 36;
  
  const validPoints = points.filter(p => {
    const x = Number(p.throughput_tok_s);
    const y = 1.0 - (p.validation_loss_ratio || p.loss_ratio || 1.0);
    return Number.isFinite(x) && Number.isFinite(y);
  });

  if (validPoints.length === 0) return null;

  const xs = validPoints.map(p => Number(p.throughput_tok_s));
  const ys = validPoints.map(p => 1.0 - (p.validation_loss_ratio || p.loss_ratio || 1.0));
  
  const xDefaults = CHART_DEFAULTS.throughput_tok_s;
  const xScale = getFixedScale('trend.throughput_tok_s', xs, {
    defaultMin: xDefaults.min,
    defaultMax: xDefaults.max,
  });
  const yScale = { min: 0, max: 1 }; // Accuracy is always 0-1

  const project = (x, y) => ({
    x: PAD + ((clampToScale(x, xScale) - xScale.min) / (xScale.max - xScale.min || 1)) * (W - PAD * 2),
    y: H - PAD - ((clampToScale(y, yScale) - yScale.min) / (yScale.max - yScale.min || 1)) * (H - PAD * 2),
  });

  const sizes = validPoints.map(p => {
      const ratio = p.compression_ratio || 1.0;
      return Math.sqrt(Math.min(10, 1 / Math.max(0.1, ratio))) * 4 + 2; 
  });

  // Calculate local Pareto frontier if not provided
  const frontierPoints = useMemo(() => {
    if (frontier && frontier.length > 0) {
        // If provided frontier uses baseline_loss_ratio, we must convert to accuracy
        return frontier.map(p => ({
            x: Number(p.throughput_tok_s),
            y: 1.0 - (p.baseline_loss_ratio || 1.0)
        })).sort((a, b) => a.x - b.x);
    }
    
    // Fallback: calculate from visible points
    const sorted = [...validPoints]
        .map((p, i) => ({ x: Number(p.throughput_tok_s), y: ys[i], id: p.result_id }))
        .sort((a, b) => a.x - b.x);
    
    const front = [];
    let maxSoFar = -Infinity;
    for (const p of sorted) {
        if (p.y > maxSoFar) {
            front.push(p);
            maxSoFar = p.y;
        }
    }
    return front;
  }, [validPoints, ys, frontier]);

  const frontierPath = frontierPoints
    .map((p, i) => {
      const pt = project(p.x, p.y);
      return `${i === 0 ? 'M' : 'L'} ${pt.x} ${pt.y}`;
    })
    .join(' ');

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', maxWidth: W }}>
      <defs>
        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>
      
      {/* Grid lines */}
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--border)" strokeWidth={1} />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="var(--border)" strokeWidth={1} />
      
      <text x={W/2} y={H-8} textAnchor="middle" fontSize={10} fill="var(--text-muted)">Compute Efficiency (Throughput)</text>
      <text x={10} y={H/2} transform={`rotate(-90 10 ${H/2})`} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
        Accuracy (1 - LR)
      </text>

      {/* Points */}
      {validPoints.map((p, i) => {
        const pt = project(xs[i], ys[i]);
        return (
          <circle 
            key={i} cx={pt.x} cy={pt.y} r={sizes[i]} 
            fill="var(--accent-blue)" opacity={0.4} stroke="var(--bg-primary)" strokeWidth={1}
          >
            <title>{`${p.result_id?.slice(0,8)} | Acc: ${(ys[i]*100).toFixed(1)}% | Thr: ${Math.round(xs[i])} | Comp: ${Number(p.compression_ratio || 1).toFixed(2)}x`}</title>
          </circle>
        );
      })}

      {/* Frontier Line */}
      {frontierPath && (
        <path 
          d={frontierPath} 
          fill="none" 
          stroke="var(--accent-cyan)" 
          strokeWidth={2} 
          strokeDasharray="4 2"
          filter="url(#glow)"
        />
      )}
    </svg>
  );
}

export default ParetoEfficiencyChart;
