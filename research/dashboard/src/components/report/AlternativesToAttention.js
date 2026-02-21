import React, { useMemo } from 'react';
import { classifyTokenMixing, FAMILY_LABELS, FAMILY_COLORS } from './reportUtils';

export default function AlternativesToAttention({ programs }) {
  const analysis = useMemo(() => {
    const familyStats = {};
    let qkvFreeCount = 0;
    let qkvCount = 0;
    let unknownCount = 0;
    const familyPrograms = {};

    for (const p of programs) {
      const { families, qkvFree } = classifyTokenMixing(p);
      if (qkvFree === null) { unknownCount++; continue; }
      if (qkvFree) qkvFreeCount++;
      else qkvCount++;

      for (const fam of families) {
        if (!familyStats[fam]) {
          familyStats[fam] = { count: 0, totalLoss: 0, totalNovelty: 0, bestLoss: Infinity, bestFingerprint: null };
          familyPrograms[fam] = [];
        }
        familyStats[fam].count++;
        if (p.loss_ratio != null) familyStats[fam].totalLoss += p.loss_ratio;
        if (p.novelty_score != null) familyStats[fam].totalNovelty += p.novelty_score;
        if (p.loss_ratio != null && p.loss_ratio < familyStats[fam].bestLoss) {
          familyStats[fam].bestLoss = p.loss_ratio;
          familyStats[fam].bestFingerprint = (p.graph_fingerprint || '').slice(0, 12);
        }
        familyPrograms[fam].push(p);
      }
    }

    const sorted = Object.entries(familyStats)
      .map(([fam, stats]) => ({
        family: fam,
        ...stats,
        avgLoss: stats.count > 0 ? stats.totalLoss / stats.count : null,
        avgNovelty: stats.count > 0 ? stats.totalNovelty / stats.count : null,
      }))
      .sort((a, b) => b.count - a.count);

    return { sorted, qkvFreeCount, qkvCount, unknownCount, total: programs.length };
  }, [programs]);

  if (analysis.sorted.length === 0) return null;

  return (
    <div className="card">
      <div className="card-title">Alternatives to Attention</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Token mixing mechanism breakdown across top programs. Shows which non-attention mechanisms
        appear in surviving architectures and their relative performance.
      </p>

      <div style={{ display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' }}>
        <div style={{
          padding: '8px 14px', borderRadius: 6, background: 'var(--bg-tertiary)',
          borderLeft: '3px solid var(--accent-green)',
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-green)' }}>
            {analysis.qkvFreeCount}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>QKV-free</div>
        </div>
        <div style={{
          padding: '8px 14px', borderRadius: 6, background: 'var(--bg-tertiary)',
          borderLeft: '3px solid var(--accent-blue)',
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-blue)' }}>
            {analysis.qkvCount}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Uses QKV</div>
        </div>
        {analysis.unknownCount > 0 && (
          <div style={{
            padding: '8px 14px', borderRadius: 6, background: 'var(--bg-tertiary)',
            borderLeft: '3px solid var(--text-muted)',
          }}>
            <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-muted)' }}>
              {analysis.unknownCount}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Unknown</div>
          </div>
        )}
      </div>

      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
            <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Mechanism</th>
            <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Programs</th>
            <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Avg Loss</th>
            <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Avg Novelty</th>
            <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Best (Loss)</th>
          </tr>
        </thead>
        <tbody>
          {analysis.sorted.map(row => (
            <tr key={row.family} style={{ borderBottom: '1px solid var(--border)' }}>
              <td style={{ padding: '6px 8px' }}>
                <span style={{
                  display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                  background: FAMILY_COLORS[row.family] || 'var(--text-muted)',
                  marginRight: 6,
                }} />
                {FAMILY_LABELS[row.family] || row.family}
              </td>
              <td style={{ padding: '6px 8px', fontWeight: 600 }}>{row.count}</td>
              <td style={{
                padding: '6px 8px',
                color: row.avgLoss != null && row.avgLoss < 0.6 ? 'var(--accent-green)' : 'var(--text-secondary)',
              }}>
                {row.avgLoss != null ? row.avgLoss.toFixed(4) : '--'}
              </td>
              <td style={{
                padding: '6px 8px',
                color: row.avgNovelty != null && row.avgNovelty > 0.5 ? 'var(--accent-green)' : 'var(--text-secondary)',
              }}>
                {row.avgNovelty != null ? row.avgNovelty.toFixed(3) : '--'}
              </td>
              <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontSize: 11 }}>
                {row.bestLoss < Infinity ? (
                  <span title={`Best: ${row.bestFingerprint}`}>
                    {row.bestLoss.toFixed(4)} ({row.bestFingerprint})
                  </span>
                ) : '--'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
        A program can use multiple mechanisms (e.g., conv + SSM). QKV-free means no attention primitives
        (local_window_attn, sliding_window_mask, multi_head_mix) are present in the graph.
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Per-candidate tags use: <strong>Full QKV</strong> (standard attention), <strong>Q=K=V</strong> (shared-projection variant),
        and <strong>QKV-free</strong> (non-attention token mixing).
      </div>
    </div>
  );
}
