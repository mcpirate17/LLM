import React from 'react';
import { fmtNumber, fmtPct } from '../../utils/format';

export function TargetBalanceCards({ summary }) {
  if (!summary) return null;
  const { efficiency, routing, adaptive } = summary;

  return (
    <div className="card">
      <div className="card-title">Balanced Targets (MoE · MoD · MoR · Mamba)</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        These KPIs track Aria’s balance across routing health (MoE), adaptive compute (MoD/MoR), and efficiency (Mamba-like throughput).
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
        <div style={{ padding: 12, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-secondary)' }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent-blue)', textTransform: 'uppercase', marginBottom: 6 }}>
            Efficiency
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Median throughput: <strong>{fmtNumber(efficiency.throughputMedian, 0)} tok/s</strong>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Median params: <strong>{efficiency.paramsMedian ? `${fmtNumber(efficiency.paramsMedian / 1e6, 2)}M` : '—'}</strong>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Median FLOPs: <strong>{efficiency.flopsMedian ? fmtNumber(efficiency.flopsMedian, 0) : '—'}</strong>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
            Samples: {efficiency.sampleCount}
          </div>
        </div>
        <div style={{ padding: 12, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-secondary)' }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent-green)', textTransform: 'uppercase', marginBottom: 6 }}>
            Routing (MoE)
          </div>
          {routing.sampleCount > 0 ? (<>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Token retention: <strong>{fmtPct(routing.retention, 1)}</strong>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Utilization entropy: <strong>{fmtNumber(routing.entropy, 3)}</strong>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Confidence: <strong>{fmtNumber(routing.confidence, 3)}</strong>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
              Best mode: {routing.bestMode || '—'} · Samples: {routing.sampleCount}
            </div>
          </>) : (
            <div style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic', marginTop: 4 }}>
              N/A — no routing architectures evaluated yet
            </div>
          )}
        </div>
        <div style={{ padding: 12, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-secondary)' }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: '#c77dff', textTransform: 'uppercase', marginBottom: 6 }}>
            Adaptive Compute
          </div>
          {adaptive.sampleCount > 0 ? (<>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Depth savings: <strong>{fmtPct(adaptive.depthSavings, 1)}</strong>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Effective depth: <strong>{fmtPct(adaptive.effectiveDepth, 1)}</strong>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Recursion savings: <strong>{fmtPct(adaptive.recursionSavings, 1)}</strong>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
              Samples with telemetry: {adaptive.sampleCount}
            </div>
          </>) : (
            <div style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic', marginTop: 4 }}>
              N/A — no adaptive compute architectures evaluated yet
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default TargetBalanceCards;
