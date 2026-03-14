import React, { useMemo } from 'react';

const PALETTE = [
  '#58a6ff', '#3fb950', '#d29922', '#bc8cff', '#f47067',
  '#39d2c0', '#e3b341', '#db61a2', '#79c0ff', '#7ee787',
];

/**
 * StabilityQualityQuadrant - 2D scatter plot of WikiText PPL vs Robustness.
 * X-axis: Capability (1/PPL) - Right is better.
 * Y-axis: Robustness (0-1) - Top is better.
 */
export default function StabilityQualityQuadrant({ entries, onSelectProgram }) {
  const W = 500;
  const H = 340;
  const PAD = 50;

  const validPoints = useMemo(() => {
    return (entries || [])
      .filter(e => {
        const ppl = Number(e.wikitext_ppl ?? e.wikitext_perplexity);
        const robustness = e.robustness_grade === 'A' ? 1.0 : (e.robustness_grade === 'B' ? 0.6 : (e.robustness_grade === 'C' ? 0.3 : Number(e.investigation_robustness ?? 0)));
        return Number.isFinite(ppl) && ppl > 0 && Number.isFinite(robustness);
      })
      .map(e => ({
        ...e,
        wikitext_ppl: Number(e.wikitext_ppl ?? e.wikitext_perplexity),
        capability: 1 / Number(e.wikitext_ppl ?? e.wikitext_perplexity),
        robustness: e.robustness_grade === 'A' ? 1.0 : (e.robustness_grade === 'B' ? 0.6 : (e.robustness_grade === 'C' ? 0.3 : Number(e.investigation_robustness ?? 0))),
        family: e.architecture_family || 'Custom',
      }));
  }, [entries]);

  const families = useMemo(() => Array.from(new Set(validPoints.map(p => p.family))), [validPoints]);
  const familyColors = useMemo(() => {
    const map = {};
    families.forEach((f, i) => { map[f] = PALETTE[i % PALETTE.length]; });
    return map;
  }, [families]);

  const bounds = useMemo(() => {
    if (validPoints.length === 0) return null;
    const caps = validPoints.map(p => p.capability);
    const robs = validPoints.map(p => p.robustness);
    return {
      cMin: Math.min(...caps),
      cMax: Math.max(...caps),
      rMin: 0,
      rMax: 1,
    };
  }, [validPoints]);

  const project = (cap, rob) => {
    if (!bounds) return { x: 0, y: 0 };
    const x = PAD + ((cap - bounds.cMin) / (bounds.cMax - bounds.cMin || 1)) * (W - PAD * 2);
    const y = H - PAD - (rob / bounds.rMax) * (H - PAD * 2);
    return { x, y };
  };

  if (validPoints.length === 0) {
    return (
      <div className="card" style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Not enough candidates with WikiText PPL data.</p>
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Stability-Quality Quadrant</div>
      <div style={{ position: 'relative', width: '100%', height: H, background: 'rgba(0,0,0,0.1)', borderRadius: 8, overflow: 'hidden' }}>
        <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
          {/* Quadrant Lines */}
          <line x1={W/2} y1={PAD} x2={W/2} y2={H-PAD} stroke="var(--border)" strokeWidth={1} strokeDasharray="4 4" />
          <line x1={PAD} y1={H/2} x2={W-PAD} y2={H/2} stroke="var(--border)" strokeWidth={1} strokeDasharray="4 4" />

          {/* Labels */}
          <text x={W - PAD} y={H - 10} textAnchor="end" fontSize={10} fill="var(--text-muted)">High Capability (Low PPL) &rarr;</text>
          <text x={10} y={PAD} transform={`rotate(-90 10 ${PAD})`} textAnchor="end" fontSize={10} fill="var(--text-muted)">High Robustness &rarr;</text>

          {/* Points */}
          {validPoints.map((p, i) => {
            const { x, y } = project(p.capability, p.robustness);
            return (
              <circle
                key={p.result_id || i}
                cx={x}
                cy={y}
                r={5}
                fill={familyColors[p.family]}
                fillOpacity={0.7}
                stroke="var(--bg-primary)"
                strokeWidth={1}
                style={{ cursor: 'pointer' }}
                onClick={() => onSelectProgram?.(p.result_id)}
              >
                <title>{`${p.result_id?.slice(0,8)} | PPL: ${p.wikitext_ppl.toFixed(2)} | Rob: ${p.robustness.toFixed(2)} | ${p.family}`}</title>
              </circle>
            );
          })}
        </svg>

        {/* Legend Overlay */}
        <div style={{ position: 'absolute', top: 10, right: 10, fontSize: 10, background: 'rgba(0,0,0,0.4)', padding: 6, borderRadius: 4 }}>
          <div style={{ color: 'var(--accent-green)', fontWeight: 700 }}>Robust & Capable</div>
          <div style={{ color: 'var(--accent-yellow)', fontSize: 9 }}>"Glass Cannons" (Bottom-Right)</div>
        </div>
      </div>
      <div style={{ display: 'flex', gap: 10, marginTop: 10, flexWrap: 'wrap' }}>
        {families.map(f => (
          <div key={f} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--text-muted)' }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: familyColors[f] }} />
            {f}
          </div>
        ))}
      </div>
    </div>
  );
}
