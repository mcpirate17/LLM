import React from 'react';
import useInteractiveTable from '../shared/useInteractiveTable';
import SortIndicator from '../shared/SortIndicator';

export function ExperimentClusters({ clustersData }) {
  const { sortKey, sortDesc, filterQuery, setFilterQuery, sortedRows: sorted, handleSort } = useInteractiveTable({
    rows: clustersData?.clusters || [],
    filterFields: ['cluster_id', 'description'],
    initialSortKey: 'avg_s1_rate',
    initialSortDesc: true,
  });

  const clusterCols = [
    { key: 'cluster_id', label: 'Cluster' },
    { key: 'size', label: 'Size' },
    { key: 'avg_s1_rate', label: 'Avg S1%' },
    { key: 'avg_best_novelty', label: 'Avg Novelty' },
    { key: 'avg_best_loss_ratio', label: 'Avg Loss Ratio' },
  ];

  if (!clustersData || !clustersData.clusters || clustersData.clusters.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Experiment Clusters</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          Need more completed experiments to compute stable clusters.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <span>Experiment Clusters ({clustersData.n_clusters})</span>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter clusters"
          className="filter-input"
        />
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Deterministic grouping of completed experiments by outcome profile (S1 rate, novelty, loss, duration).
        Stability score indicates how well-separated clusters are.
      </p>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>
        <strong style={{ color: 'var(--accent-purple)' }}>Stability:</strong>{' '}
        {(clustersData.stability_score ?? 0).toFixed(3)}
        <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>
          ({clustersData.n_experiments} experiments)
        </span>
      </div>
      <div style={{ maxHeight: 260, overflow: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              {clusterCols.map(col => (
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
            {sorted.map(c => (
              <React.Fragment key={c.cluster_id}>
                <tr>
                  <td style={{ color: 'var(--accent-blue)' }}>#{c.cluster_id}</td>
                  <td>{c.size}</td>
                  <td>{((c.avg_s1_rate || 0) * 100).toFixed(1)}%</td>
                  <td>{(c.avg_best_novelty || 0).toFixed(3)}</td>
                  <td>{(c.avg_best_loss_ratio || 0).toFixed(3)}</td>
                </tr>
                {c.description && (
                  <tr>
                    <td colSpan={5} style={{
                      fontSize: 11, color: 'var(--text-muted)',
                      fontStyle: 'italic', paddingTop: 0, paddingBottom: 8,
                      borderBottom: '1px solid var(--border)',
                    }}>
                      {c.description}
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default ExperimentClusters;
