import React, { useState, useEffect, useMemo } from 'react';
import { formatTime, scoreColor } from '../utils/format';
import { confidenceColor } from '../utils/colors';

const API_BASE = process.env.REACT_APP_API_URL || '';

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

const INSIGHTS_SORT_PREFS_KEY = 'dashboard.insights.sort.v1';


function NegativeResultsSection() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/api/analytics/negative-results`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="card"><p style={{ color: 'var(--text-muted)', fontSize: 12 }}>Loading negative results...</p></div>;
  if (error) return <div className="card"><p style={{ color: 'var(--accent-red)', fontSize: 12 }}>Error: {error}</p></div>;
  if (!data) return null;

  const hasContent = (data.failed_ops?.length > 0) || (data.dominant_errors?.length > 0) ||
    (data.anti_patterns?.length > 0) || (data.refuted_hypotheses?.length > 0);
  if (!hasContent) return null;

  return (
    <div className="card">
      <div
        onClick={() => setOpen(!open)}
        style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center', userSelect: 'none' }}
      >
        <div className="card-title" style={{ margin: 0 }}>Do Not Pursue</div>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          {open ? '\u25BE collapse' : '\u25B8 expand'}
        </span>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, marginBottom: open ? 12 : 0, lineHeight: 1.5 }}>
        {data.summary}
      </p>
      {open && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Failed Ops */}
          {data.failed_ops?.length > 0 && (
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
                Zero-Success Ops ({data.failed_ops.length})
              </div>
              {data.failed_ops.map(op => (
                <div key={op.op_name} style={{
                  display: 'flex', justifyContent: 'space-between', padding: '4px 0',
                  borderBottom: '1px solid var(--border)', alignItems: 'center',
                }}>
                  <span style={{ fontSize: 12, fontFamily: 'monospace', color: 'var(--accent-red)' }}>{op.op_name}</span>
                  <span style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 11 }}>
                    <span style={{ color: 'var(--text-muted)' }}>0/{op.n_used} S1</span>
                    <span style={{ color: 'var(--text-muted)' }}>fails at {op.failure_stage}</span>
                    <span style={{ color: confidenceColor(op.confidence), fontWeight: 600, fontSize: 10 }}>
                      {(op.confidence * 100).toFixed(0)}%
                    </span>
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Dominant Errors */}
          {data.dominant_errors?.length > 0 && (
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
                Top Error Types ({data.dominant_errors.length})
              </div>
              {data.dominant_errors.slice(0, 8).map(err => (
                <div key={err.error_type} style={{
                  display: 'flex', justifyContent: 'space-between', padding: '4px 0',
                  borderBottom: '1px solid var(--border)', alignItems: 'center', gap: 8,
                }}>
                  <span style={{ fontSize: 11, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {err.error_type}
                  </span>
                  <span style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 11, flexShrink: 0 }}>
                    <span style={{ color: 'var(--accent-red)' }}>{err.count}</span>
                    <span style={{ color: 'var(--text-muted)' }}>{err.percentage}%</span>
                    <span style={{ color: 'var(--text-muted)' }}>@ {err.primary_stage}</span>
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Anti-Patterns */}
          {data.anti_patterns?.length > 0 && (
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
                Anti-Correlated Features ({data.anti_patterns.length})
              </div>
              {data.anti_patterns.map(ap => (
                <div key={ap.metric} style={{
                  display: 'flex', justifyContent: 'space-between', padding: '4px 0',
                  borderBottom: '1px solid var(--border)', alignItems: 'center',
                }}>
                  <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{ap.feature}</span>
                  <span style={{ fontSize: 11, color: 'var(--accent-red)', fontWeight: 600 }}>
                    {ap.correlation.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Refuted Hypotheses */}
          {data.refuted_hypotheses?.length > 0 && (
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
                Refuted Hypotheses ({data.refuted_hypotheses.length})
              </div>
              {data.refuted_hypotheses.map((h, i) => (
                <div key={i} style={{
                  padding: '6px 8px', borderLeft: '3px solid var(--accent-red)',
                  marginBottom: 4, fontSize: 12, color: 'var(--text-secondary)',
                  background: 'rgba(248, 81, 73, 0.05)',
                }}>
                  {h.content}
                  {h.evidence && (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                      Evidence: {h.evidence}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function InsightsPanel({ insights, compact }) {
  const [sortKey, setSortKey] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(INSIGHTS_SORT_PREFS_KEY) || '{}');
      const validKeys = new Set([...COLUMNS_FULL, ...COLUMNS_COMPACT].map((column) => column.key));
      if (typeof stored.sortKey === 'string' && validKeys.has(stored.sortKey)) {
        return stored.sortKey;
      }
    } catch {}
    return '_score';
  });
  const [sortDesc, setSortDesc] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(INSIGHTS_SORT_PREFS_KEY) || '{}');
      if (typeof stored.sortDesc === 'boolean') {
        return stored.sortDesc;
      }
    } catch {}
    return true;
  });
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
    try {
      localStorage.setItem(INSIGHTS_SORT_PREFS_KEY, JSON.stringify({ sortKey, sortDesc }));
    } catch {}
  }, [sortKey, sortDesc]);

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
                      color: confidenceColor(insight.confidence),
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
      {!compact && <NegativeResultsSection />}
    </div>
  );
}

export default InsightsPanel;
