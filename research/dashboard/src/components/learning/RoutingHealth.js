import React, { useState, useMemo } from 'react';
import { reliabilityColor } from '../../utils/colors';
import { filterRowsByQuery } from '../../utils/tableFiltering';

function routingMetricChips(row) {
  const conf = row.avg_confidence_mean;
  return [
    {
      label: 'Routing',
      source: 'telemetry',
      reliability: conf != null
        ? (conf >= 0.7 ? 'high' : conf >= 0.4 ? 'medium' : 'low')
        : 'low',
    },
    {
      label: 'Sample',
      source: 'mode-aggregate',
      reliability: (row.n_programs || 0) >= 80 ? 'high' : (row.n_programs || 0) >= 30 ? 'medium' : 'low',
    },
  ];
}

export function RoutingHealth({ data }) {
  const [sortKey, setSortKey] = useState('n_programs');
  const [sortDesc, setSortDesc] = useState(true);
  const [filterQuery, setFilterQuery] = useState('');

  const handleSort = (key) => {
    if (sortKey === key) { setSortDesc(!sortDesc); } else { setSortKey(key); setSortDesc(true); }
  };

  const filtered = useMemo(() => (
    filterRowsByQuery(data?.by_mode || [], filterQuery, ['routing_mode'])
  ), [data?.by_mode, filterQuery]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      let va = a[sortKey], vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [filtered, sortKey, sortDesc]);

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
                  {sortKey === col.key && (
                    <span style={{ marginLeft: 4, fontSize: 10 }}>
                      {sortDesc ? '\u25BC' : '\u25B2'}
                    </span>
                  )}
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
                  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', maxWidth: 220 }}>
                    {chips.map(chip => (
                      <span
                        key={`${row.routing_mode}-${chip.label}`}
                        title={`${chip.label}: ${chip.source}, ${chip.reliability} reliability`}
                        style={{
                          fontSize: 10,
                          padding: '1px 5px',
                          borderRadius: 4,
                          border: `1px solid ${reliabilityColor(chip.reliability)}55`,
                          color: reliabilityColor(chip.reliability),
                          background: `${reliabilityColor(chip.reliability)}22`,
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {chip.label}: {chip.source}
                      </span>
                    ))}
                  </div>
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
