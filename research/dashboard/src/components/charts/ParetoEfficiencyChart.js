import React from 'react';

export function ParetoEfficiencyChart({ points }) {
  if (!Array.isArray(points) || points.length === 0) return null;
  const W = 440;
  const H = 220;
  const PAD = 32;
  
  const xs = points.map(p => {
      const throughput = Number(p.throughput_tok_s || 0);
      return Math.min(1.0, throughput / 50000); 
  });
  const ys = points.map(p => Math.max(0, 1.0 - (p.validation_loss_ratio || p.loss_ratio || 1.0)));
  const sizes = points.map(p => {
      const ratio = p.compression_ratio || 1.0;
      return Math.sqrt(Math.min(10, 1 / Math.max(0.1, ratio))) * 4 + 2; 
  });

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', maxWidth: W }}>
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--border)" strokeWidth={1} />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="var(--border)" strokeWidth={1} />
      
      <text x={W/2} y={H-8} textAnchor="middle" fontSize={10} fill="var(--text-muted)">Compute Efficiency (Throughput)</text>
      <text x={10} y={H/2} transform={`rotate(-90 10 ${H/2})`} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
        Accuracy (1 - LR)
      </text>

      {points.map((p, i) => {
        const x = PAD + xs[i] * (W - PAD*2);
        const y = H - PAD - ys[i] * (H - PAD*2);
        return (
          <circle 
            key={i} cx={x} cy={y} r={sizes[i]} 
            fill="var(--accent-blue)" opacity={0.6} stroke="var(--bg-primary)" strokeWidth={1}
          >
            <title>{`${p.result_id?.slice(0,8)} | Acc: ${(ys[i]*100).toFixed(1)}% | Eff: ${(xs[i]*100).toFixed(1)}% | Comp: ${Number(p.compression_ratio || 1).toFixed(2)}x`}</title>
          </circle>
        );
      })}
    </svg>
  );
}

export default ParetoEfficiencyChart;
