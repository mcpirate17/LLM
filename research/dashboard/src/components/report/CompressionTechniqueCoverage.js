import React, { useMemo } from 'react';
import { parseArchSpec, COMPRESSION_FACTORS, WEIGHT_STORAGE_LABELS, TOKEN_REP_LABELS } from './reportUtils';

export default function CompressionTechniqueCoverage({ programs }) {
  const analysis = useMemo(() => {
    const byTechnique = {};
    let denseCount = 0;
    let compressedCount = 0;

    for (const p of programs) {
      const spec = parseArchSpec(p.arch_spec_json);
      const ws = spec?.choices?.weight_storage || 'dense_matrix';
      const tr = spec?.choices?.token_representation;
      const isDense = ws === 'dense_matrix' && (!tr || tr === 'standard_float');

      if (isDense) { denseCount++; } else { compressedCount++; }

      const key = ws !== 'dense_matrix' ? ws : (tr && tr !== 'standard_float' ? tr : 'dense_matrix');
      if (!byTechnique[key]) {
        byTechnique[key] = {
          count: 0, s1Pass: 0, totalLoss: 0, lossCount: 0,
          totalParams: 0, paramsCount: 0, bestLoss: Infinity, bestFingerprint: null,
        };
      }
      const m = byTechnique[key];
      m.count++;
      if (p.stage1_passed) m.s1Pass++;
      if (p.loss_ratio != null) { m.totalLoss += p.loss_ratio; m.lossCount++; }
      if (p.param_count != null) { m.totalParams += p.param_count; m.paramsCount++; }
      if (p.loss_ratio != null && p.loss_ratio < m.bestLoss) {
        m.bestLoss = p.loss_ratio;
        m.bestFingerprint = (p.graph_fingerprint || '').slice(0, 12);
      }
    }

    const sorted = Object.entries(byTechnique)
      .map(([technique, m]) => ({
        technique,
        label: WEIGHT_STORAGE_LABELS[technique] || TOKEN_REP_LABELS[technique] || technique,
        count: m.count,
        s1Rate: m.count > 0 ? m.s1Pass / m.count : 0,
        avgLoss: m.lossCount > 0 ? m.totalLoss / m.lossCount : null,
        avgParams: m.paramsCount > 0 ? m.totalParams / m.paramsCount : null,
        factor: COMPRESSION_FACTORS[technique] || 1.0,
        bestLoss: m.bestLoss < Infinity ? m.bestLoss : null,
        bestFingerprint: m.bestFingerprint,
      }))
      .sort((a, b) => b.count - a.count);

    return { sorted, denseCount, compressedCount, total: programs.length };
  }, [programs]);

  if (analysis.compressedCount === 0) return null;

  return (
    <div className="card">
      <div className="card-title">Compression Technique Coverage</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Weight storage and token representation techniques used across stage-1 survivors.
        Compressed architectures use fewer parameters for comparable or better performance.
      </p>

      <div style={{ display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' }}>
        <div style={{
          padding: '8px 14px', borderRadius: 6, background: 'var(--bg-tertiary)',
          borderLeft: '3px solid var(--accent-green)',
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-green)' }}>
            {analysis.compressedCount}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Compressed</div>
        </div>
        <div style={{
          padding: '8px 14px', borderRadius: 6, background: 'var(--bg-tertiary)',
          borderLeft: '3px solid var(--text-muted)',
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-muted)' }}>
            {analysis.denseCount}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Dense (baseline)</div>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Technique</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>N</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>S1 Rate</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Avg Loss</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Avg Params</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Est. Ratio</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Best (Loss)</th>
            </tr>
          </thead>
          <tbody>
            {analysis.sorted.map(row => (
              <tr key={row.technique} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '6px 8px', fontWeight: 600, color: row.factor < 1 ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
                  {row.label}
                </td>
                <td style={{ padding: '6px 8px' }}>{row.count}</td>
                <td style={{
                  padding: '6px 8px',
                  color: row.s1Rate > 0.5 ? 'var(--accent-green)' : row.s1Rate > 0.2 ? 'var(--accent-yellow)' : 'var(--text-secondary)',
                }}>
                  {(row.s1Rate * 100).toFixed(0)}%
                </td>
                <td style={{
                  padding: '6px 8px',
                  color: row.avgLoss != null && row.avgLoss < 0.6 ? 'var(--accent-green)' : 'var(--text-secondary)',
                }}>
                  {row.avgLoss != null ? row.avgLoss.toFixed(4) : '--'}
                </td>
                <td style={{ padding: '6px 8px', color: 'var(--text-secondary)' }}>
                  {row.avgParams != null ? `${(row.avgParams / 1e6).toFixed(2)}M` : '--'}
                </td>
                <td style={{
                  padding: '6px 8px',
                  color: row.factor < 1 ? 'var(--accent-green)' : 'var(--text-muted)',
                }}>
                  {row.factor < 1 ? `${(row.factor * 100).toFixed(0)}%` : '100%'}
                </td>
                <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontSize: 11 }}>
                  {row.bestLoss != null ? (
                    <span title={`Best: ${row.bestFingerprint}`}>
                      {row.bestLoss.toFixed(4)} ({row.bestFingerprint})
                    </span>
                  ) : '--'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Est. Ratio = estimated parameter retention after compression (lower = more compressed).
        Techniques from the morphological box weight_storage and token_representation dimensions.
      </div>
    </div>
  );
}
