import React from 'react';
import useInteractiveTable from '../shared/useInteractiveTable';
import SortIndicator from '../shared/SortIndicator';

export function MathFamilyCoverage({ data }) {
  const rows = Array.isArray(data?.families) ? data.families : [];
  const totals = data?.totals || {};

  const { sortKey, sortDesc, filterQuery, setFilterQuery, sortedRows: sorted, handleSort } = useInteractiveTable({
    rows,
    filterFields: ['family'],
    initialSortKey: 'n_tested',
    initialSortDesc: true,
  });

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
          className="filter-input"
        />
      </div>
      <div style={{ maxHeight: 260, overflow: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('family')} style={{ cursor: 'pointer' }}>
                Family<SortIndicator active={sortKey === 'family'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('n_tested')} style={{ cursor: 'pointer' }}>
                Tested<SortIndicator active={sortKey === 'n_tested'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('n_survived')} style={{ cursor: 'pointer' }}>
                Survivors<SortIndicator active={sortKey === 'n_survived'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('survival_rate')} style={{ cursor: 'pointer' }}>
                Survival %<SortIndicator active={sortKey === 'survival_rate'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('tested_share')} style={{ cursor: 'pointer' }}>
                Test Share<SortIndicator active={sortKey === 'tested_share'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('survivor_share')} style={{ cursor: 'pointer' }}>
                Survivor Share<SortIndicator active={sortKey === 'survivor_share'} desc={sortDesc} />
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
