import React, { useMemo } from 'react';
import { classifyTokenMixing, FAMILY_LABELS, FAMILY_COLORS } from './reportUtils';
import useInteractiveTable from '../shared/useInteractiveTable';
import SortIndicator from '../shared/SortIndicator';

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
        const lr = p.validation_loss_ratio != null ? p.validation_loss_ratio : p.loss_ratio;
        familyStats[fam].count++;
        if (lr != null) familyStats[fam].totalLoss += lr;
        if (p.novelty_score != null) familyStats[fam].totalNovelty += p.novelty_score;
        if (lr != null && lr < familyStats[fam].bestLoss) {
          familyStats[fam].bestLoss = lr;
          familyStats[fam].bestFingerprint = (p.graph_fingerprint || '').slice(0, 12);
        }
        familyPrograms[fam].push(p);
      }
    }

    const rows = Object.entries(familyStats)
      .map(([fam, stats]) => ({
        family: fam,
        ...stats,
        avgLoss: stats.count > 0 ? stats.totalLoss / stats.count : null,
        avgNovelty: stats.count > 0 ? stats.totalNovelty / stats.count : null,
      }));

    return { rows, qkvFreeCount, qkvCount, unknownCount, total: programs.length };
  }, [programs]);

  const { sortKey, sortDesc, filterQuery, setFilterQuery, sortedRows, handleSort } = useInteractiveTable({
    rows: analysis.rows,
    filterFields: ['family'],
    initialSortKey: 'count',
    initialSortDesc: true,
  });

  if (sortedRows.length === 0) return null;

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

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Filter:</div>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter mechanisms"
          className="filter-input"
        />
      </div>
      <table className="data-table table-compact">
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
            <th onClick={() => handleSort('family')} style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer' }}>
              Mechanism<SortIndicator active={sortKey === 'family'} desc={sortDesc} />
            </th>
            <th onClick={() => handleSort('count')} style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer' }}>
              Programs<SortIndicator active={sortKey === 'count'} desc={sortDesc} />
            </th>
            <th onClick={() => handleSort('avgLoss')} style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer' }}>
              Avg Loss<SortIndicator active={sortKey === 'avgLoss'} desc={sortDesc} />
            </th>
            <th onClick={() => handleSort('avgNovelty')} style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer' }}>
              Avg Novelty<SortIndicator active={sortKey === 'avgNovelty'} desc={sortDesc} />
            </th>
            <th onClick={() => handleSort('bestLoss')} style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer' }}>
              Best (Loss)<SortIndicator active={sortKey === 'bestLoss'} desc={sortDesc} />
            </th>
          </tr>
        </thead>
        <tbody>
          {sortedRows.map(row => (
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
