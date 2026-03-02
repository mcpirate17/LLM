import React, { useState, useMemo } from 'react';
import { filterRowsByQuery } from '../../utils/tableFiltering';

export function MathFamilyCoverage({ data }) {
  const rows = Array.isArray(data?.families) ? data.families : [];
  const totals = data?.totals || {};
  const [sortKey, setSortKey] = useState('n_tested');
  const [sortDesc, setSortDesc] = useState(true);
  const [filterQuery, setFilterQuery] = useState('');

  const filtered = useMemo(() => (
    filterRowsByQuery(rows, filterQuery, ['family'])
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

  if (rows.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Math Family Coverage</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          No program-family coverage data yet.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Math Family Coverage</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Share of evaluated and Stage-1 surviving programs by math family. Use this to verify the search is exploring beyond standard Euclidean patterns.
      </p>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>
        <strong style={{ color: 'var(--accent-purple)' }}>Totals:</strong>{' '}
        {totals.n_tested ?? 0} tested, {totals.n_survived ?? 0} Stage-1 survivors
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Filter:</div>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter families"
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
              <th onClick={() => handleSort('family')} style={{ cursor: 'pointer' }}>
                Family{sortKey === 'family' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('n_tested')} style={{ cursor: 'pointer' }}>
                Tested{sortKey === 'n_tested' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('n_survived')} style={{ cursor: 'pointer' }}>
                Survivors{sortKey === 'n_survived' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('survival_rate')} style={{ cursor: 'pointer' }}>
                Survival %{sortKey === 'survival_rate' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('tested_share')} style={{ cursor: 'pointer' }}>
                Test Share{sortKey === 'tested_share' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('survivor_share')} style={{ cursor: 'pointer' }}>
                Survivor Share{sortKey === 'survivor_share' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(row => (
              <tr key={row.family}>
                <td style={{ textTransform: 'capitalize', color: 'var(--accent-blue)' }}>{row.family}</td>
                <td>{row.n_tested ?? 0}</td>
                <td>{row.n_survived ?? 0}</td>
                <td>{((row.survival_rate || 0) * 100).toFixed(1)}%</td>
                <td>{((row.tested_share || 0) * 100).toFixed(1)}%</td>
                <td>{((row.survivor_share || 0) * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default MathFamilyCoverage;
