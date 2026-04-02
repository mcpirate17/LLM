import React from 'react';
import useInteractiveTable from '../shared/useInteractiveTable';
import SortIndicator from '../shared/SortIndicator';
import { MetricChipList } from '../shared/MetricChipBadge';

import { routingMetricChips } from '../../utils/metricChips';

export function RoutingHealth({ data }) {
  const { sortKey, sortDesc, filterQuery, setFilterQuery, sortedRows: sorted, handleSort } = useInteractiveTable({
    rows: data?.by_mode || [],
    filterFields: ['routing_mode'],
    initialSortKey: 'n_programs',
    initialSortDesc: true,
  });

  if (!data || data.available === false || !data.by_mode || data.by_mode.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Routing Health</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          No routing telemetry available yet. Routing health tracks how well mixture-of-experts
          architectures distribute work across their expert paths. It will appear once the system
          generates and evaluates routed architectures.
        </p>
      </div>
    );
  }

  const routingCols = [
    { key: 'routing_mode', label: 'Mode' },
    { key: 'n_programs', label: 'N' },
    { key: 'sample_size_label', label: 'Sample' },
    { key: 'stage1_pass_rate', label: 'S1%' },
    { key: 'avg_drop_rate', label: 'Drop%' },
    { key: 'avg_utilization_entropy', label: 'Entropy' },
    { key: 'avg_confidence_mean', label: 'Conf' },
    { key: 'confidence_label', label: 'Conf Label' },
    { key: 'stability_label', label: 'Stability' },
    { key: '_quality', label: 'Metric Quality' },
  ];

  return (
    <div className="card">
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <span>Routing Health ({data.n_modes} modes)</span>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter modes"
          className="filter-input"
        />
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Aggregated routing telemetry by mode. Lower drop rate and higher confidence generally indicate healthier routing.
      </p>
      {data.explanation && (
        <div style={{ marginBottom: 10, padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6, borderLeft: '3px solid var(--accent-purple)' }}>
          <div style={{ fontSize: 11, color: 'var(--accent-purple)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 4 }}>
            Plain-language interpretation
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            {data.explanation}
          </div>
        </div>
      )}
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>
        <strong style={{ color: 'var(--accent-purple)' }}>Overall S1 pass:</strong>{' '}
        {((data.overall_stage1_pass_rate || 0) * 100).toFixed(1)}%
        <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>
          ({data.total_programs} programs)
        </span>
      </div>
      <div style={{ maxHeight: 260, overflow: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              {routingCols.map(col => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}
                  aria-label={`Sort by ${col.label}`}
                >
                  {col.label}
                  <SortIndicator active={sortKey === col.key} desc={sortDesc} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => {
              const chips = routingMetricChips(row);
              return (
              <tr key={row.routing_mode}>
                <td style={{ color: 'var(--accent-blue)' }}>{row.routing_mode}</td>
                <td>{row.n_programs ?? 0}</td>
                <td style={{ textTransform: 'uppercase', fontSize: 11 }}>{row.sample_size_label || 'unknown'}</td>
                <td>{((row.stage1_pass_rate || 0) * 100).toFixed(1)}%</td>
                <td>{((row.avg_drop_rate || 0) * 100).toFixed(1)}%</td>
                <td>{row.avg_utilization_entropy != null ? Number(row.avg_utilization_entropy).toFixed(3) : 'not measured'}</td>
                <td>{row.avg_confidence_mean != null ? Number(row.avg_confidence_mean).toFixed(3) : 'not measured'}</td>
                <td style={{ textTransform: 'uppercase', fontSize: 11 }}>{row.confidence_label || 'unknown'}</td>
                <td style={{ textTransform: 'uppercase', fontSize: 11 }}>{row.stability_label || 'unknown'}</td>
                <td>
                    <MetricChipList chips={chips} />
                </td>
              </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default RoutingHealth;
