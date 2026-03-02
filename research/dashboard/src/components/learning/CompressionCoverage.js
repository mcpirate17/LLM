import React, { useState, useMemo } from 'react';
import { filterRowsByQuery } from '../../utils/tableFiltering';

const COMPRESSION_FACTORS = {
  low_rank: 0.55, shared_basis: 0.5, hash_trick: 0.35,
  structured_sparse: 0.4, kronecker: 0.5, polynomial: 0.6,
  residual_quantized: 0.3,
};

const WEIGHT_STORAGE_LABELS = {
  dense_matrix: 'Dense (baseline)', low_rank: 'Low-Rank (UV)',
  hypernetwork: 'Hypernetwork', shared_basis: 'Shared Basis',
  hash_trick: 'Hash Trick', kronecker: 'Kronecker',
  polynomial: 'Polynomial', structured_sparse: 'Structured Sparse',
};

const TOKEN_REP_LABELS = {
  standard_float: 'Standard Float', binary_hash: 'Binary Hash',
  residual_quantized: 'Residual Quantized', complex_valued: 'Complex',
  quaternion: 'Quaternion', multi_resolution: 'Multi-Resolution',
  mixture_embedding: 'Mixture Embedding',
};

function parseArchSpec(value) {
  if (!value || typeof value !== 'string') return null;
  try {
    const p = JSON.parse(value);
    return p && typeof p === 'object' ? p : null;
  } catch { return null; }
}

export function CompressionCoverage({ data, programs }) {
  const [sortKey, setSortKey] = useState('count');
  const [sortDesc, setSortDesc] = useState(true);
  const [filterQuery, setFilterQuery] = useState('');
  const analysis = useMemo(() => {
    if (data && Array.isArray(data.techniques)) {
      const totals = data.totals || {};
      const rows = [...data.techniques]
        .map((row) => ({
          technique: row.technique,
          label: WEIGHT_STORAGE_LABELS[row.technique] || TOKEN_REP_LABELS[row.technique] || row.technique,
          count: row.n_survived ?? 0,
          tested: row.n_tested ?? 0,
          avgLoss: row.avg_loss_ratio,
          bestLoss: row.best_loss_ratio,
          avgRatio: row.avg_compression_ratio,
          avgMemoryMb: row.avg_estimated_memory_mb,
          avgRetention: row.avg_quality_retention,
          survivalRate: row.survival_rate,
        }));

      return {
        rows,
        denseCount: Math.max(0, (totals.n_survived || 0) - (totals.n_compressed_survived || 0)),
        compressedCount: totals.n_compressed_survived || 0,
        total: totals.n_survived || 0,
        testedTotal: totals.n_tested || 0,
        compressedTested: totals.n_compressed_tested || 0,
      };
    }

    if (!programs || programs.length === 0) return null;
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
        byTechnique[key] = { count: 0, totalLoss: 0, lossCount: 0, bestLoss: Infinity };
      }
      const m = byTechnique[key];
      m.count++;
      if (p.loss_ratio != null) { m.totalLoss += p.loss_ratio; m.lossCount++; }
      if (p.loss_ratio != null && p.loss_ratio < m.bestLoss) m.bestLoss = p.loss_ratio;
    }

    const rows = Object.entries(byTechnique)
      .map(([technique, m]) => ({
        technique,
        label: WEIGHT_STORAGE_LABELS[technique] || TOKEN_REP_LABELS[technique] || technique,
        count: m.count,
        avgLoss: m.lossCount > 0 ? m.totalLoss / m.lossCount : null,
        factor: COMPRESSION_FACTORS[technique] || 1.0,
        bestLoss: m.bestLoss < Infinity ? m.bestLoss : null,
      }));

    return { rows, denseCount, compressedCount, total: programs.length };
  }, [data, programs]);

  const filtered = useMemo(() => (
    filterRowsByQuery(analysis?.rows || [], filterQuery, ['technique', 'label'])
  ), [analysis?.rows, filterQuery]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      const va = a?.[sortKey];
      const vb = b?.[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [filtered, sortKey, sortDesc]);

  const handleSort = (key) => {
    if (sortKey === key) { setSortDesc(!sortDesc); } else { setSortKey(key); setSortDesc(true); }
  };

  if (!analysis || analysis.compressedCount === 0) {
    return (
      <div className="card">
        <div className="card-title">Compression Technique Coverage</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          No compressed architectures among survivors yet. All current stage-1 survivors use dense
          weight matrices. Compression coverage will appear when the system generates and evaluates
          architectures with non-standard weight storage (low-rank, hash trick, sparse, etc.).
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Compression Technique Coverage</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Weight storage techniques across stage-1 survivors with explicit compression ratio,
        memory footprint, and quality-retention tradeoff summaries.
      </p>
      <div style={{ display: 'flex', gap: 12, marginBottom: 10, fontSize: 12, color: 'var(--text-secondary)' }}>
        <span><strong style={{ color: 'var(--accent-green)' }}>Compressed:</strong> {analysis.compressedCount}</span>
        <span><strong style={{ color: 'var(--text-muted)' }}>Dense:</strong> {analysis.denseCount}</span>
        <span style={{ color: 'var(--text-muted)' }}>({analysis.total} total)</span>
        {analysis.testedTotal != null && (
          <span style={{ color: 'var(--text-muted)' }}>
            tested {analysis.compressedTested}/{analysis.testedTotal} compressed
          </span>
        )}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Filter:</div>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter techniques"
          style={{
            fontSize: 11,
            padding: '4px 8px',
            borderRadius: 4,
            border: '1px solid var(--border)',
            background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)',
            minWidth: 160,
          }}
        />
      </div>
      <div style={{ maxHeight: 260, overflow: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('label')} style={{ cursor: 'pointer' }}>
                Technique{sortKey === 'label' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('tested')} style={{ cursor: 'pointer' }}>
                Tested{sortKey === 'tested' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('count')} style={{ cursor: 'pointer' }}>
                N{sortKey === 'count' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('survivalRate')} style={{ cursor: 'pointer' }}>
                Survival %{sortKey === 'survivalRate' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('avgLoss')} style={{ cursor: 'pointer' }}>
                Avg Loss{sortKey === 'avgLoss' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('bestLoss')} style={{ cursor: 'pointer' }}>
                Best Loss{sortKey === 'bestLoss' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('avgRatio')} style={{ cursor: 'pointer' }}>
                Avg Ratio{sortKey === 'avgRatio' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('avgMemoryMb')} style={{ cursor: 'pointer' }}>
                Avg Mem (MB){sortKey === 'avgMemoryMb' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('avgRetention')} style={{ cursor: 'pointer' }}>
                Quality Retention{sortKey === 'avgRetention' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(row => (
              <tr key={row.technique}>
                <td style={{ color: (row.avgRatio != null && row.avgRatio < 1) ? 'var(--accent-green)' : 'var(--text-secondary)', fontWeight: 600 }}>
                  {row.label}
                </td>
                <td>{row.tested ?? '--'}</td>
                <td>{row.count}</td>
                <td>{row.survivalRate != null ? `${(row.survivalRate * 100).toFixed(1)}%` : '--'}</td>
                <td style={{ color: row.avgLoss != null && row.avgLoss < 0.6 ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
                  {row.avgLoss != null ? row.avgLoss.toFixed(4) : '--'}
                </td>
                <td>{row.bestLoss != null ? row.bestLoss.toFixed(4) : '--'}</td>
                <td style={{ color: row.avgRatio != null && row.avgRatio < 1 ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                  {row.avgRatio != null ? `${(row.avgRatio * 100).toFixed(0)}%` : '--'}
                </td>
                <td>{row.avgMemoryMb != null ? row.avgMemoryMb.toFixed(2) : '--'}</td>
                <td>{row.avgRetention != null ? `${(row.avgRetention * 100).toFixed(0)}%` : '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default CompressionCoverage;
