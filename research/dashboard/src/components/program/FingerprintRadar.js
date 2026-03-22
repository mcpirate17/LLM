import React from 'react';

export function FingerprintRadar({ program, size = 240 }) {
  // Extended radar with fingerprint metrics
  const axes = [
    { key: 'novelty_score', label: 'Novelty' },
    { key: 'structural_novelty', label: 'Structural' },
    { key: 'behavioral_novelty', label: 'Behavioral' },
    { key: 'fp_interaction_locality', label: 'Locality' },
    { key: 'fp_interaction_sparsity', label: 'Sparsity' },
    { key: 'fp_isotropy', label: 'Isotropy' },
    { key: 'fp_rank_ratio', label: 'Rank' },
    { key: 'fp_sensitivity_uniformity', label: 'Sensitivity' },
  ].filter(a => program[a.key] !== null && program[a.key] !== undefined);

  // Fall back to minimal if no fingerprint data
  if (axes.length < 3) {
    const fallback = [
      { key: 'novelty_score', label: 'Novelty' },
      { key: 'structural_novelty', label: 'Structural' },
      { key: 'behavioral_novelty', label: 'Behavioral' },
    ];
    axes.length = 0;
    axes.push(...fallback);
  }

  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 30;
  const n = axes.length;

  const getPoint = (i, val) => {
    const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
    const d = val * r;
    return { x: cx + d * Math.cos(angle), y: cy + d * Math.sin(angle) };
  };

  const rings = [0.25, 0.5, 0.75, 1.0];
  const values = axes.map(a => Math.min(program[a.key] || 0, 1));
  const points = values.map((v, i) => getPoint(i, v));
  const polygonPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z';

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {rings.map(ring => {
        const ringPoints = axes.map((_, i) => getPoint(i, ring));
        const d = ringPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z';
        return <path key={ring} d={d} fill="none" stroke="var(--border, #30363d)" strokeWidth={0.5} />;
      })}
      {axes.map((_, i) => {
        const end = getPoint(i, 1);
        return <line key={i} x1={cx} y1={cy} x2={end.x} y2={end.y}
          stroke="var(--border, #30363d)" strokeWidth={0.5} />;
      })}
      <path d={polygonPath} fill="rgba(0, 212, 255, 0.2)" stroke="var(--accent-purple, #00d4ff)" strokeWidth={2} />
      {points.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={3}
          fill="var(--accent-purple, #00d4ff)" stroke="var(--bg-secondary, #161b22)" strokeWidth={1.5} />
      ))}
      {axes.map((axis, i) => {
        const labelPt = getPoint(i, 1.25);
        return (
          <text key={i} x={labelPt.x} y={labelPt.y}
            textAnchor="middle" dominantBaseline="middle"
            fill="var(--text-secondary, #8b949e)" fontSize={9}>
            {axis.label}
          </text>
        );
      })}
    </svg>
  );
}

export default React.memo(FingerprintRadar);
