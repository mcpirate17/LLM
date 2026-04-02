import React from 'react';
import { reliabilityBand } from './reportUtils';
import useInteractiveTable from '../shared/useInteractiveTable';
import SortIndicator from '../shared/SortIndicator';

export default function FunctionalFamilyEvidence({ coverage }) {
  const families = Array.isArray(coverage?.families) ? coverage.families : [];
  const totals = coverage?.totals || {};

  const { sortKey, sortDesc, filterQuery, setFilterQuery, sortedRows, handleSort } = useInteractiveTable({
    rows: families,
    filterFields: ['family'],
    initialSortKey: 'n_tested',
    initialSortDesc: true,
  });

  const functional = families.find(row => row.family === 'functional') || null;
  const exoticFamilies = families.filter(row => row.family !== 'euclidean');
  const exoticTested = exoticFamilies.reduce((sum, row) => sum + (row.n_tested || 0), 0);
  const exoticSurvived = exoticFamilies.reduce((sum, row) => sum + (row.n_survived || 0), 0);
  const testedBand = reliabilityBand(functional?.n_tested || 0);
  const survivedBand = reliabilityBand(functional?.n_survived || 0);

  if (families.length === 0) return null;

  return (
    <div className="card">
      <div className="card-title">Functional-Family Search Coverage</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Decision-focused evidence of whether exotic mathematical families, especially functional operators,
        are actually being explored and surviving stage-1 checks.
      </p>

      <div style={{ display: 'flex', gap: 14, marginBottom: 14, flexWrap: 'wrap' }}>
        <div style={{ padding: '8px 12px', borderRadius: 6, background: 'var(--bg-tertiary)', borderLeft: '3px solid var(--accent-purple)' }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-purple)' }}>{totals.n_tested || 0}</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Total Tested</div>
        </div>
        <div style={{ padding: '8px 12px', borderRadius: 6, background: 'var(--bg-tertiary)', borderLeft: '3px solid var(--accent-green)' }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-green)' }}>{totals.n_survived || 0}</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Total Survivors</div>
        </div>
        <div style={{ padding: '8px 12px', borderRadius: 6, background: 'var(--bg-tertiary)', borderLeft: '3px solid var(--accent-yellow)' }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-yellow)' }}>{exoticTested}</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Exotic Tested</div>
        </div>
      </div>

      {functional && (
        <div style={{ marginBottom: 12, fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.55 }}>
          <div>
            <strong>Functional family tested:</strong> {functional.n_tested} ({(functional.tested_share * 100).toFixed(1)}% of all programs)
            {' · '}
            <strong style={{ color: testedBand.color, textTransform: 'uppercase', fontSize: 10 }}>{testedBand.label} sample depth</strong>
          </div>
          <div>
            <strong>Functional survivors:</strong> {functional.n_survived} (S1 rate {(functional.survival_rate * 100).toFixed(1)}%)
            {' · '}
            <strong style={{ color: survivedBand.color, textTransform: 'uppercase', fontSize: 10 }}>{survivedBand.label} survivor evidence</strong>
          </div>
          <div>
            <strong>Exotic family survivors:</strong> {exoticSurvived} across hyperbolic/tropical/p-adic/clifford/functional.
          </div>
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Filter:</div>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter families"
          className="filter-input"
        />
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table table-compact">
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
              <th onClick={() => handleSort('family')} style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer' }}>
                Family<SortIndicator active={sortKey === 'family'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('n_tested')} style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer' }}>
                Tested<SortIndicator active={sortKey === 'n_tested'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('n_survived')} style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer' }}>
                Survived<SortIndicator active={sortKey === 'n_survived'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('survival_rate')} style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer' }}>
                S1 Rate<SortIndicator active={sortKey === 'survival_rate'} desc={sortDesc} />
              </th>
              <th onClick={() => handleSort('tested_share')} style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer' }}>
                Share of Tested<SortIndicator active={sortKey === 'tested_share'} desc={sortDesc} />
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedRows.map(row => {
              const isFunctional = row.family === 'functional';
              const testedShare = Number(row.tested_share || 0) * 100;
              const survivalRate = Number(row.survival_rate || 0) * 100;
              return (
                <tr key={row.family} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '6px 8px', fontWeight: isFunctional ? 700 : 500, color: isFunctional ? 'var(--accent-purple)' : 'var(--text-secondary)' }}>
                    {row.family}
                  </td>
                  <td style={{ padding: '6px 8px' }}>{row.n_tested}</td>
                  <td style={{ padding: '6px 8px' }}>{row.n_survived}</td>
                  <td style={{ padding: '6px 8px', color: survivalRate >= 10 ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
                    {survivalRate.toFixed(1)}%
                  </td>
                  <td style={{ padding: '6px 8px' }}>{testedShare.toFixed(1)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
