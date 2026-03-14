import React from 'react';

/**
 * Sparkline - Renders a compact trajectory of evaluation metrics.
 * Designed for PPL (lower is better).
 */
export default function Sparkline({ data, width = 80, height = 24, color = 'var(--accent-blue)', referenceData }) {
  if (!data || data.length < 2) {
    return <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>--</span>;
  }

  // Filter out nulls/invalid numbers
  const points = data.filter(v => v != null && Number.isFinite(v));
  if (points.length < 2) return <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>--</span>;

  // For PPL, we want to scale relative to the range, but maybe use log scale or fixed floor
  // Find min/max for scaling. Include reference data if present.
  const allPoints = referenceData ? [...points, ...referenceData] : points;
  const min = Math.min(...allPoints);
  const max = Math.max(...allPoints);
  const range = max - min || 1;

  const getPathD = (dataPoints) => {
    return dataPoints.map((v, i) => {
      const x = (i / (dataPoints.length - 1)) * width;
      // y=0 is top, so (max - v) / range maps max to height and min to 0
      // We want min at BOTTOM for quality? Wait, PPL lower is better.
      // So min (best) should be at BOTTOM? No, usually UP is GOOD.
      // Let's make UP = BETTER (lower PPL).
      const y = height - ((max - v) / range) * height;
      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');
  };

  const pathD = getPathD(points);
  const refPathD = referenceData ? getPathD(referenceData) : null;

  return (
    <div title={`Trajectory: ${points.join(' \u2192 ')}`} style={{ display: 'inline-flex', alignItems: 'center' }}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ overflow: 'visible' }}>
        {/* Reference trajectory (dashed gray) */}
        {refPathD && (
          <path
            d={refPathD}
            fill="none"
            stroke="var(--text-muted)"
            strokeWidth={1}
            strokeDasharray="2 1"
            opacity={0.5}
          />
        )}
        
        {/* Main trajectory */}
        <path
          d={pathD}
          fill="none"
          stroke={color}
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        
        {/* End point dot */}
        <circle
          cx={width}
          cy={height - ((max - points[points.length - 1]) / range) * height}
          r={2.5}
          fill={color}
        />
      </svg>
    </div>
  );
}
