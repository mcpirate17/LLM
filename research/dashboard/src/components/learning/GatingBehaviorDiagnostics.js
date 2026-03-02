import React, { useState, useMemo } from 'react';
import { filterRowsByQuery } from '../../utils/tableFiltering';

export function GatingBehaviorDiagnostics({ data }) {
  if (!data || data.available === false) {
    return (
      <div className="card">
        <div className="card-title">Gating Behavior Diagnostics</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          No gating diagnostics available yet. This section appears once routed or recursive candidates are evaluated.
        </p>
      </div>
    );
  }

  const rows = Array.isArray(data.by_mode) ? data.by_mode : [];
  const [sortKey, setSortKey] = useState('n_programs');
  const [sortDesc, setSortDesc] = useState(true);
  const [filterQuery, setFilterQuery] = useState('');

  const filtered = useMemo(() => (
    filterRowsByQuery(rows, filterQuery, ['routing_mode'])
  ), [rows, filterQuery]);

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
  return (
    <div className="card">
      <div className="card-title">Gating Behavior Diagnostics</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Canonical diagnostics for gate entropy, route-collapse risk, and token-retention curves across routing modes.
      </p>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8 }}>
        <strong style={{ color: 'var(--accent-purple)' }}>Routed candidates:</strong> {data.total_routed_programs || 0}
        <span style={{ marginLeft: 10 }}>
          <strong style={{ color: 'var(--accent-purple)' }}>Avg entropy:</strong>{' '}
          {data.avg_gate_entropy != null ? Number(data.avg_gate_entropy).toFixed(3) : 'not measured'}
        </span>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
        Collapse risk modes — high: {data?.collapse_risk_counts?.high || 0}, medium: {data?.collapse_risk_counts?.medium || 0}, low: {data?.collapse_risk_counts?.low || 0}
      </div>
      {data.explanation && (
        <div style={{ marginBottom: 10, padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6, borderLeft: '3px solid var(--accent-purple)', fontSize: 12, color: 'var(--text-secondary)' }}>
          {data.explanation}
        </div>
      )}
      {rows.length > 0 && (
        <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Filter:</div>
          <input
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            placeholder="Filter modes"
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
      )}
      {rows.length > 0 && (
        <div style={{ maxHeight: 260, overflow: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th onClick={() => handleSort('routing_mode')} style={{ cursor: 'pointer' }}>
                  Mode{sortKey === 'routing_mode' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
                </th>
                <th onClick={() => handleSort('n_programs')} style={{ cursor: 'pointer' }}>
                  N{sortKey === 'n_programs' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
                </th>
                <th onClick={() => handleSort('avg_gate_entropy')} style={{ cursor: 'pointer' }}>
                  Entropy{sortKey === 'avg_gate_entropy' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
                </th>
                <th onClick={() => handleSort('collapse_risk_label')} style={{ cursor: 'pointer' }}>
                  Collapse Risk{sortKey === 'collapse_risk_label' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
                </th>
                <th onClick={() => handleSort('avg_token_retention')} style={{ cursor: 'pointer' }}>
                  Retention (avg){sortKey === 'avg_token_retention' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
                </th>
                <th>Retention Curve</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((row) => (
                <tr key={row.routing_mode}>
                  <td style={{ color: 'var(--accent-blue)' }}>{row.routing_mode}</td>
                  <td>{row.n_programs ?? 0}</td>
                  <td>{row.avg_gate_entropy != null ? Number(row.avg_gate_entropy).toFixed(3) : 'not measured'}</td>
                  <td style={{ textTransform: 'uppercase', fontSize: 11 }}>{row.collapse_risk_label || 'unknown'}</td>
                  <td>{row.avg_token_retention != null ? `${(Number(row.avg_token_retention) * 100).toFixed(1)}%` : 'not measured'}</td>
                  <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {Array.isArray(row.token_retention_curve) && row.token_retention_curve.length > 0
                      ? row.token_retention_curve.map(point => `${point.quantile}:${(Number(point.retention) * 100).toFixed(0)}%`).join(' · ')
                      : 'not measured'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default GatingBehaviorDiagnostics;
