import React, { useMemo } from 'react';

export default function RoutingModeComparison({ programs, comparison }) {
  const analysis = useMemo(() => {
    if (comparison && Array.isArray(comparison.by_mode)) {
      const sorted = [...comparison.by_mode]
        .map((row) => ({
          mode: row.routing_mode,
          count: row.n_programs || 0,
          sampleLabel: row.sample_size_label || 'unknown',
          confidenceLabel: row.confidence_label || 'unknown',
          stabilityLabel: row.stability_label || 'unknown',
          s1Rate: row.stage1_pass_rate || 0,
          avgLoss: row.avg_loss_ratio,
          avgDrop: row.avg_drop_rate,
          avgEntropy: row.avg_utilization_entropy,
          avgConf: row.avg_confidence_mean,
          tokenRetention: row.token_retention,
        }))
        .sort((a, b) => b.count - a.count);

      return {
        sorted,
        routedCount: comparison.routed_programs || 0,
        uniformCount: comparison.uniform_programs || 0,
        total: comparison.total_programs || 0,
      };
    }

    const byMode = {};
    let routedCount = 0;
    let uniformCount = 0;

    for (const p of programs) {
      const mode = p.routing_mode;
      if (!mode) { uniformCount++; continue; }
      routedCount++;
      if (!byMode[mode]) {
        byMode[mode] = {
          count: 0, s1Pass: 0, totalLoss: 0, lossCount: 0,
          totalDrop: 0, dropCount: 0, totalEntropy: 0, entropyCount: 0,
          totalConf: 0, confCount: 0, bestLoss: Infinity, bestFingerprint: null,
        };
      }
      const m = byMode[mode];
      m.count++;
      if (p.stage1_passed) m.s1Pass++;
      if (p.loss_ratio != null) { m.totalLoss += p.loss_ratio; m.lossCount++; }
      if (p.routing_drop_rate != null) { m.totalDrop += p.routing_drop_rate; m.dropCount++; }
      if (p.routing_utilization_entropy != null) { m.totalEntropy += p.routing_utilization_entropy; m.entropyCount++; }
      if (p.routing_confidence_mean != null) { m.totalConf += p.routing_confidence_mean; m.confCount++; }
      if (p.loss_ratio != null && p.loss_ratio < m.bestLoss) {
        m.bestLoss = p.loss_ratio;
        m.bestFingerprint = (p.graph_fingerprint || '').slice(0, 12);
      }
    }

    const sorted = Object.entries(byMode)
      .map(([mode, m]) => ({
        mode,
        count: m.count,
        sampleLabel: m.count >= 80 ? 'high' : m.count >= 30 ? 'medium' : 'low',
        confidenceLabel: 'unknown',
        stabilityLabel: 'unknown',
        s1Rate: m.count > 0 ? m.s1Pass / m.count : 0,
        avgLoss: m.lossCount > 0 ? m.totalLoss / m.lossCount : null,
        avgDrop: m.dropCount > 0 ? m.totalDrop / m.dropCount : null,
        avgEntropy: m.entropyCount > 0 ? m.totalEntropy / m.entropyCount : null,
        avgConf: m.confCount > 0 ? m.totalConf / m.confCount : null,
        bestLoss: m.bestLoss < Infinity ? m.bestLoss : null,
        bestFingerprint: m.bestFingerprint,
        tokenRetention: m.dropCount > 0 ? Math.max(0, 1 - (m.totalDrop / m.dropCount)) : null,
      }))
      .sort((a, b) => b.count - a.count);

    return { sorted, routedCount, uniformCount, total: programs.length };
  }, [programs, comparison]);

  if (analysis.sorted.length === 0) return null;

  return (
    <div className="card">
      <div className="card-title">Routing Mode Comparison</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Consolidated routing-mode evidence across uniform and routed candidates.
        Includes sample-size and confidence labels to avoid over-reading small-N differences.
      </p>

      <div style={{ display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' }}>
        <div style={{
          padding: '8px 14px', borderRadius: 6, background: 'var(--bg-tertiary)',
          borderLeft: '3px solid var(--accent-purple)',
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-purple)' }}>
            {analysis.routedCount}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Routed</div>
        </div>
        <div style={{
          padding: '8px 14px', borderRadius: 6, background: 'var(--bg-tertiary)',
          borderLeft: '3px solid var(--text-muted)',
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-muted)' }}>
            {analysis.uniformCount}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Uniform (no routing)</div>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Mode</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>N</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Sample</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>S1 Rate</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Avg Loss</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Drop %</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Entropy</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Confidence</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Conf Label</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Stability</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Token Retention</th>
            </tr>
          </thead>
          <tbody>
            {analysis.sorted.map(row => (
              <tr key={row.mode} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '6px 8px', color: 'var(--accent-blue)', fontWeight: 600 }}>{row.mode}</td>
                <td style={{ padding: '6px 8px' }}>{row.count}</td>
                <td style={{ padding: '6px 8px', textTransform: 'uppercase', fontSize: 11 }}>{row.sampleLabel}</td>
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
                <td style={{
                  padding: '6px 8px',
                  color: row.avgDrop != null
                    ? (row.avgDrop > 0.3 ? 'var(--accent-red)' : row.avgDrop > 0.1 ? 'var(--accent-yellow)' : 'var(--accent-green)')
                    : 'var(--text-muted)',
                }}>
                  {row.avgDrop != null ? `${(row.avgDrop * 100).toFixed(1)}%` : '--'}
                </td>
                <td style={{ padding: '6px 8px', color: 'var(--text-secondary)' }}>
                  {row.avgEntropy != null ? row.avgEntropy.toFixed(3) : '--'}
                </td>
                <td style={{
                  padding: '6px 8px',
                  color: row.avgConf != null
                    ? (row.avgConf > 0.8 ? 'var(--accent-green)' : row.avgConf > 0.5 ? 'var(--accent-yellow)' : 'var(--accent-red)')
                    : 'var(--text-muted)',
                }}>
                  {row.avgConf != null ? row.avgConf.toFixed(3) : '--'}
                </td>
                <td style={{ padding: '6px 8px', textTransform: 'uppercase', fontSize: 11 }}>{row.confidenceLabel}</td>
                <td style={{ padding: '6px 8px', textTransform: 'uppercase', fontSize: 11 }}>{row.stabilityLabel}</td>
                <td style={{ padding: '6px 8px' }}>{row.tokenRetention != null ? `${(row.tokenRetention * 100).toFixed(1)}%` : '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Sample labels reflect evidence depth by mode (`high`, `medium`, `low`).
        Confidence labels combine confidence mean and variance; stability reflects confidence variance.
      </div>
    </div>
  );
}
