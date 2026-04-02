import React from 'react';
import useInteractiveTable from './useInteractiveTable';
import SortIndicator from './SortIndicator';

export function MathspaceImpact({ data }) {
  const rows = Array.isArray(data?.by_operator) ? data.by_operator : [];
  const families = Array.isArray(data?.by_family) ? data.by_family : [];
  const topTrust = Array.isArray(data?.top_trustworthy_operators) ? data.top_trustworthy_operators : [];
  const totals = data?.totals || {};
  const {
    sortKey, sortDesc, filterQuery, setFilterQuery, sortedRows: sorted, handleSort,
  } = useInteractiveTable({
    rows,
    filterFields: ['op_name'],
    initialSortKey: 'n_tested',
    initialSortDesc: true,
  });

  if (!data || data.available === false || rows.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Mathspace Operator Impact</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          No mathspace operator impact data yet. This appears once programs include hyperbolic/tropical/p-adic/clifford operators.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Mathspace Operator Impact</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Canonical impact slice for mathspace operators and families across Stage-1 pass, validation pass, and novelty signals.
      </p>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>
        <strong style={{ color: 'var(--accent-purple)' }}>Coverage:</strong>{' '}
        {totals.n_programs_with_mathspace ?? 0}/{totals.n_programs_with_graph ?? 0} programs with graph traces include mathspace ops
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Trust score = (50% S1 pass + 30% validation pass + 20% baseline wins) × sample reliability,
        where sample reliability scales with tested count up to 25 programs.
      </div>

      {topTrust.length > 0 && (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
          {topTrust.map((row) => (
            <span
              key={row.op_name}
              style={{
                fontSize: 11,
                padding: '4px 8px',
                borderRadius: 999,
                border: `1px solid ${row.trust_label === 'high' ? 'var(--accent-green)' : row.trust_label === 'medium' ? 'var(--accent-yellow)' : 'var(--text-muted)'}`,
                color: row.trust_label === 'high' ? 'var(--accent-green)' : row.trust_label === 'medium' ? 'var(--accent-yellow)' : 'var(--text-muted)',
                background: 'var(--bg-tertiary)',
              }}
            >
              {row.op_name} · trust {(Number(row.trust_score || 0) * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Filter:</div>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter operators"
          className="filter-input"
        />
      </div>
      <div style={{ maxHeight: 220, overflow: 'auto', marginBottom: 10 }}>
        <table className="data-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('op_name')} style={{ cursor: 'pointer' }}>
                Operator<SortIndicator active={sortKey === 'op_name'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('n_tested')} style={{ cursor: 'pointer' }}>
                Tested<SortIndicator active={sortKey === 'n_tested'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('stage1_pass_rate')} style={{ cursor: 'pointer' }}>
                S1 %<SortIndicator active={sortKey === 'stage1_pass_rate'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('validation_pass_rate')} style={{ cursor: 'pointer' }}>
                Validation %<SortIndicator active={sortKey === 'validation_pass_rate'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('baseline_win_rate')} style={{ cursor: 'pointer' }}>
                Baseline Win %<SortIndicator active={sortKey === 'baseline_win_rate'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('trust_score')} style={{ cursor: 'pointer' }}>
                Trust %<SortIndicator active={sortKey === 'trust_score'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('avg_novelty_score')} style={{ cursor: 'pointer' }}>
                Avg Novelty<SortIndicator active={sortKey === 'avg_novelty_score'} desc={sortDesc} />
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.slice(0, 10).map((row) => (
              <tr key={row.op_name}>
                <td style={{ color: 'var(--accent-blue)' }}>{row.op_name}</td>
                <td>{row.n_tested ?? 0}</td>
                <td>{((row.stage1_pass_rate || 0) * 100).toFixed(1)}%</td>
                <td>{((row.validation_pass_rate || 0) * 100).toFixed(1)}%</td>
                <td>{((row.baseline_win_rate || 0) * 100).toFixed(1)}%</td>
                <td>{((row.trust_score || 0) * 100).toFixed(1)}%</td>
                <td>{row.avg_novelty_score != null ? Number(row.avg_novelty_score).toFixed(3) : '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {families.length > 0 && (
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', fontSize: 11, color: 'var(--text-muted)' }}>
          {families.map((row) => (
            <span key={row.family}>
              <strong style={{ color: 'var(--accent-purple)' }}>{row.family}:</strong> S1 {(row.stage1_pass_rate * 100).toFixed(0)}% · V {(row.validation_pass_rate * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default MathspaceImpact;
