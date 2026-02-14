import React, { useState, useEffect } from 'react';
import GraphViewer from './GraphViewer';

const API_BASE = process.env.REACT_APP_API_URL || '';

/**
 * ProgramDetail — Modal showing computation graph, stage pipeline,
 * fingerprint radar chart, training metrics, similar architectures.
 */

function StagePipeline({ program }) {
  const stages = [
    { key: 'stage0_passed', label: 'Stage 0', sublabel: 'Compilation' },
    { key: 'stage05_passed', label: 'Stage 0.5', sublabel: 'Stability' },
    { key: 'stage1_passed', label: 'Stage 1', sublabel: 'Learning' },
  ];

  return (
    <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
      {stages.map((stage, i) => {
        const passed = program[stage.key];
        const color = passed ? 'var(--accent-green)' : 'var(--accent-red)';
        const bg = passed ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)';
        return (
          <React.Fragment key={stage.key}>
            {i > 0 && <span style={{ color: 'var(--text-muted)', fontSize: 16 }}>&rarr;</span>}
            <div style={{
              padding: '6px 12px',
              background: bg,
              border: `1px solid ${color}`,
              borderRadius: 6,
              textAlign: 'center',
              minWidth: 80,
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color }}>{passed ? 'PASS' : 'FAIL'}</div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{stage.label}</div>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}

function RadarChart({ program, size = 200 }) {
  const axes = [
    { key: 'novelty_score', label: 'Novelty' },
    { key: 'structural_novelty', label: 'Structural' },
    { key: 'behavioral_novelty', label: 'Behavioral' },
  ];

  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 30;
  const n = axes.length;

  const getPoint = (i, val) => {
    const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
    const d = val * r;
    return { x: cx + d * Math.cos(angle), y: cy + d * Math.sin(angle) };
  };

  // Grid rings
  const rings = [0.25, 0.5, 0.75, 1.0];

  // Data polygon
  const values = axes.map(a => Math.min(program[a.key] || 0, 1));
  const points = values.map((v, i) => getPoint(i, v));
  const polygonPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z';

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {/* Grid */}
      {rings.map(ring => {
        const ringPoints = axes.map((_, i) => getPoint(i, ring));
        const d = ringPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z';
        return <path key={ring} d={d} fill="none" stroke="var(--border, #30363d)" strokeWidth={0.5} />;
      })}

      {/* Axes */}
      {axes.map((_, i) => {
        const end = getPoint(i, 1);
        return <line key={i} x1={cx} y1={cy} x2={end.x} y2={end.y}
          stroke="var(--border, #30363d)" strokeWidth={0.5} />;
      })}

      {/* Data */}
      <path d={polygonPath} fill="rgba(188, 140, 255, 0.2)" stroke="var(--accent-purple, #bc8cff)" strokeWidth={2} />
      {points.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={4}
          fill="var(--accent-purple, #bc8cff)" stroke="var(--bg-secondary, #161b22)" strokeWidth={2} />
      ))}

      {/* Labels */}
      {axes.map((axis, i) => {
        const labelPt = getPoint(i, 1.2);
        return (
          <text key={i} x={labelPt.x} y={labelPt.y}
            textAnchor="middle" dominantBaseline="middle"
            fill="var(--text-secondary, #8b949e)" fontSize={10}>
            {axis.label}
          </text>
        );
      })}
    </svg>
  );
}

function ProgramDetail({ resultId, onClose }) {
  const [program, setProgram] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!resultId) return;
    setLoading(true);
    fetch(`${API_BASE}/api/programs/${resultId}`)
      .then(r => r.json())
      .then(d => { setProgram(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [resultId]);

  if (!resultId) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ fontSize: 16, margin: 0 }}>Program Detail</h3>
          <button className="refresh-btn" onClick={onClose} style={{ fontSize: 18, lineHeight: 1, padding: '4px 8px' }}>&times;</button>
        </div>

        {loading ? (
          <p style={{ color: 'var(--text-muted)' }}>Loading...</p>
        ) : !program ? (
          <p style={{ color: 'var(--accent-red)' }}>Program not found</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Header info */}
            <div>
              <div style={{ fontFamily: 'monospace', fontSize: 13, color: 'var(--accent-blue)', marginBottom: 4 }}>
                {program.graph_fingerprint}
              </div>
              <StagePipeline program={program} />
            </div>

            {/* Error if failed */}
            {program.stage0_error && (
              <div style={{
                padding: 8,
                background: 'rgba(248, 81, 73, 0.1)',
                border: '1px solid var(--accent-red)',
                borderRadius: 4,
                fontSize: 12,
                fontFamily: 'monospace',
                color: 'var(--accent-red)',
              }}>
                {program.stage0_error}
              </div>
            )}

            {/* Metrics + Radar side by side */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
                  Metrics
                </div>
                <div style={{ fontSize: 13 }}>
                  {[
                    ['Parameters', program.param_count ? `${(program.param_count / 1e6).toFixed(2)}M` : '--'],
                    ['Loss Ratio', program.loss_ratio?.toFixed(4) || '--'],
                    ['Final Loss', program.final_loss?.toFixed(4) || '--'],
                    ['Throughput', program.throughput_tok_s ? `${program.throughput_tok_s.toFixed(0)} tok/s` : '--'],
                    ['Novelty', program.novelty_score?.toFixed(3) || '--'],
                    ['Similar To', program.most_similar_to || '--'],
                  ].map(([label, value]) => (
                    <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: '1px solid var(--border)' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
                      <span>{value}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <RadarChart program={program} size={180} />
              </div>
            </div>

            {/* LLM Explanation */}
            {program.llm_explanation && (
              <div style={{
                padding: 12,
                background: 'var(--bg-tertiary)',
                borderRadius: 4,
                borderLeft: '2px solid var(--accent-purple)',
                fontSize: 13,
                color: 'var(--text-secondary)',
                fontStyle: 'italic',
              }}>
                <div style={{ fontSize: 11, color: 'var(--accent-purple)', marginBottom: 4, fontWeight: 600, fontStyle: 'normal' }}>
                  ARIA'S ANALYSIS
                </div>
                {program.llm_explanation}
              </div>
            )}

            {/* Graph Viewer */}
            {program.graph_json_parsed && (
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
                  Computation Graph
                </div>
                <GraphViewer graph={program.graph_json_parsed} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default ProgramDetail;
