import React, { useState, useMemo } from 'react';

const CATEGORY_COLORS = {
  pattern: 'var(--accent-blue)',
  failure_mode: 'var(--accent-red)',
  success_factor: 'var(--accent-green)',
  hypothesis: 'var(--accent-purple)',
};

const CATEGORY_ORDER = {
  success_factor: 4,
  pattern: 3,
  hypothesis: 2,
  failure_mode: 1,
};

const STATUS_ORDER = {
  confirmed: 3,
  active: 2,
  superseded: 1,
  refuted: 0,
};

/**
 * Score an insight 0-100.
 * Weights: confidence (40%), category importance (30%), status (20%), has evidence (10%)
 */
function insightScore(insight) {
  const conf = (insight.confidence || 0.5) * 40;
  const cat = ((CATEGORY_ORDER[insight.category] || 0) / 4) * 30;
  const status = ((STATUS_ORDER[insight.status] || 0) / 3) * 20;
  const evidence = insight.supporting_evidence ? 10 : 0;
  return Math.round(Math.max(0, Math.min(100, conf + cat + status + evidence)));
}

function scoreColor(score) {
  if (score >= 70) return 'var(--accent-green)';
  if (score >= 40) return 'var(--accent-yellow)';
  if (score >= 20) return 'var(--accent-orange, #f0883e)';
  return 'var(--text-muted)';
}

const COLUMNS_FULL = [
  { key: '_score', label: 'Score' },
  { key: 'category', label: 'Category' },
  { key: 'content', label: 'Content' },
  { key: 'confidence', label: 'Confidence' },
  { key: 'status', label: 'Status' },
  { key: 'timestamp', label: 'Time' },
];

const COLUMNS_COMPACT = [
  { key: '_score', label: 'Score' },
  { key: 'category', label: 'Category' },
  { key: 'content', label: 'Content' },
  { key: 'confidence', label: 'Conf' },
];

function formatTime(timestamp) {
  if (!timestamp) return '';
  return new Date(timestamp * 1000).toLocaleString();
}

function InsightsPanel({ insights, compact }) {
  const [sortKey, setSortKey] = useState('_score');
  const [sortDesc, setSortDesc] = useState(true);
  const [expandedId, setExpandedId] = useState(null);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDesc(!sortDesc);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  const augmented = useMemo(() => {
    if (!insights) return [];
    return insights.map(ins => ({ ...ins, _score: insightScore(ins) }));
  }, [insights]);

  const sorted = useMemo(() => {
    const arr = [...augmented];
    arr.sort((a, b) => {
      let va, vb;
      if (sortKey === '_score') {
        va = a._score; vb = b._score;
      } else if (sortKey === 'category') {
        va = CATEGORY_ORDER[a.category] || 0;
        vb = CATEGORY_ORDER[b.category] || 0;
      } else if (sortKey === 'status') {
        va = STATUS_ORDER[a.status] || 0;
        vb = STATUS_ORDER[b.status] || 0;
      } else {
        va = a[sortKey]; vb = b[sortKey];
      }
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') {
        return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      }
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [augmented, sortKey, sortDesc]);

  const columns = compact ? COLUMNS_COMPACT : COLUMNS_FULL;

  if (!insights || insights.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Insights {compact ? '(Preview)' : ''}</div>
        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
          No insights recorded yet. Run experiments to generate insights.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">
        Insights {compact ? `(${insights.length})` : `— ${insights.length} Active`}
      </div>
      {!compact && (
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
          Patterns and conclusions discovered during experiments. These inform future experiment
          design — for example, if a certain operation type consistently fails, the system
          reduces its weight. Confidence reflects how much data supports each insight.
        </p>
      )}
      <table className="data-table">
        <thead>
          <tr>
            {columns.map(col => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                aria-label={`Sort insights by ${col.label}${sortKey === col.key ? `, currently ${sortDesc ? 'descending' : 'ascending'}` : ''}`}
                style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}
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
          {sorted.map((insight, i) => {
            const score = insight._score;
            const isExpanded = expandedId === (insight.insight_id || i);
            const contentPreview = (insight.content || '').length > 120
              ? insight.content.slice(0, 120) + '...'
              : insight.content;

            return (
              <React.Fragment key={insight.insight_id || i}>
                <tr
                  style={{ cursor: 'pointer' }}
                  role="button"
                  tabIndex={0}
                  aria-expanded={isExpanded}
                  aria-label={`${isExpanded ? 'Collapse' : 'Expand'} insight ${(insight.category || 'item').replace('_', ' ')}`}
                  onClick={() => setExpandedId(isExpanded ? null : (insight.insight_id || i))}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setExpandedId(isExpanded ? null : (insight.insight_id || i));
                    }
                  }}
                >
                  <td style={{ fontWeight: 600, color: scoreColor(score) }}>
                    {score}
                  </td>
                  <td>
                    <span style={{
                      fontSize: 11,
                      fontWeight: 600,
                      color: CATEGORY_COLORS[insight.category] || 'var(--accent-yellow)',
                      textTransform: 'uppercase',
                    }}>
                      {(insight.category || '').replace('_', ' ')}
                    </span>
                  </td>
                  <td style={{ color: 'var(--text-secondary)', maxWidth: compact ? 200 : 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 12 }}>
                    {contentPreview}
                  </td>
                  <td>
                    <span style={{
                      color: (insight.confidence || 0) >= 0.7 ? 'var(--accent-green)'
                        : (insight.confidence || 0) >= 0.4 ? 'var(--accent-yellow)' : 'var(--text-muted)',
                      fontWeight: (insight.confidence || 0) >= 0.7 ? 600 : 400,
                    }}>
                      {((insight.confidence || 0.5) * 100).toFixed(0)}%
                    </span>
                  </td>
                  {!compact && (
                    <td>
                      {insight.status && (
                        <span className={`badge ${
                          insight.status === 'confirmed' ? 'pass'
                          : insight.status === 'active' ? 'running'
                          : 'fail'
                        }`}>
                          {insight.status}
                        </span>
                      )}
                    </td>
                  )}
                  {!compact && (
                    <td style={{ fontSize: 12, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                      {formatTime(insight.timestamp)}
                    </td>
                  )}
                </tr>
                {isExpanded && (
                  <tr>
                    <td colSpan={columns.length} style={{ padding: 0 }}>
                      <div style={{
                        padding: '12px 16px',
                        background: 'var(--bg-tertiary)',
                        borderLeft: `3px solid ${CATEGORY_COLORS[insight.category] || 'var(--accent-yellow)'}`,
                        fontSize: 13,
                        color: 'var(--text-secondary)',
                        lineHeight: 1.6,
                        whiteSpace: 'pre-wrap',
                      }}>
                        {insight.content}
                        {insight.supporting_evidence && (
                          <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid var(--border)', fontSize: 12, color: 'var(--text-muted)' }}>
                            <strong>Evidence:</strong> {insight.supporting_evidence}
                          </div>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default InsightsPanel;
