import React, { useState, useMemo } from 'react';

function formatTime(timestamp) {
  if (!timestamp) return '';
  return new Date(timestamp * 1000).toLocaleString();
}

const TYPE_ORDER = {
  insight: 6,
  analysis: 5,
  result: 4,
  decision: 3,
  hypothesis: 2,
  observation: 1,
  error: 1,
  note: 0,
};

const TYPE_COLORS = {
  insight: 'var(--accent-green)',
  analysis: 'var(--accent-purple)',
  result: 'var(--accent-blue)',
  decision: 'var(--accent-yellow)',
  hypothesis: 'var(--accent-blue)',
  observation: 'var(--text-secondary)',
  error: 'var(--accent-red)',
  note: 'var(--text-muted)',
};

/**
 * Score a notebook entry 0-100 by importance.
 * Weights: entry type (50%), content length/richness (30%), has tags (10%), has experiment (10%)
 */
function entryScore(entry) {
  const typeScore = ((TYPE_ORDER[entry.entry_type] || 0) / 6) * 50;

  const contentLen = (entry.content || '').length;
  const contentScore = Math.min(contentLen / 500, 1.0) * 30;

  const tagScore = entry.tags ? 10 : 0;
  const expScore = entry.experiment_id ? 10 : 0;

  return Math.round(Math.max(0, Math.min(100, typeScore + contentScore + tagScore + expScore)));
}

function scoreColor(score) {
  if (score >= 70) return 'var(--accent-green)';
  if (score >= 40) return 'var(--accent-yellow)';
  if (score >= 20) return 'var(--accent-orange, #f0883e)';
  return 'var(--text-muted)';
}

const COLUMNS = [
  { key: '_score', label: 'Score' },
  { key: 'entry_type', label: 'Type' },
  { key: 'title', label: 'Title' },
  { key: 'content', label: 'Content' },
  { key: 'tags', label: 'Tags' },
  { key: 'timestamp', label: 'Time' },
];

function LabNotebook({ entries }) {
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
    if (!entries) return [];
    return entries.map(e => ({ ...e, _score: entryScore(e) }));
  }, [entries]);

  const sorted = useMemo(() => {
    const arr = [...augmented];
    arr.sort((a, b) => {
      let va, vb;
      if (sortKey === '_score') {
        va = a._score; vb = b._score;
      } else if (sortKey === 'entry_type') {
        va = TYPE_ORDER[a.entry_type] || 0;
        vb = TYPE_ORDER[b.entry_type] || 0;
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

  if (!entries || entries.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Lab Notebook</div>
        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
          No entries yet. The lab notebook will fill as Aria runs experiments.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Lab Notebook — Recent Entries</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Chronological research log of hypotheses, observations, results, and decisions. Use this to understand why
        the system changed strategy and what evidence supported each step.
      </p>
      <table className="data-table">
        <thead>
          <tr>
            {COLUMNS.map(col => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                aria-label={`Sort notebook entries by ${col.label}${sortKey === col.key ? `, currently ${sortDesc ? 'descending' : 'ascending'}` : ''}`}
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
          {sorted.map((entry, i) => {
            const score = entry._score;
            const isExpanded = expandedId === (entry.entry_id || i);
            const contentPreview = (entry.content || '').length > 120
              ? entry.content.slice(0, 120) + '...'
              : entry.content;

            return (
              <React.Fragment key={entry.entry_id || i}>
                <tr
                  style={{ cursor: 'pointer' }}
                  role="button"
                  tabIndex={0}
                  aria-expanded={isExpanded}
                  aria-label={`${isExpanded ? 'Collapse' : 'Expand'} notebook entry ${entry.title || entry.entry_type || 'row'}`}
                  onClick={() => setExpandedId(isExpanded ? null : (entry.entry_id || i))}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setExpandedId(isExpanded ? null : (entry.entry_id || i));
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
                      color: TYPE_COLORS[entry.entry_type] || 'var(--text-muted)',
                      textTransform: 'uppercase',
                    }}>
                      {entry.entry_type}
                    </span>
                  </td>
                  <td style={{ fontWeight: 500, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {entry.title}
                  </td>
                  <td style={{ color: 'var(--text-secondary)', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 12 }}>
                    {contentPreview}
                  </td>
                  <td>
                    {entry.tags && entry.tags.split(',').filter(Boolean).map((tag, j) => (
                      <span key={j} style={{
                        fontSize: 10,
                        padding: '1px 5px',
                        background: 'var(--bg-primary)',
                        borderRadius: 3,
                        marginRight: 3,
                        color: 'var(--text-muted)',
                        whiteSpace: 'nowrap',
                      }}>
                        {tag.trim()}
                      </span>
                    ))}
                  </td>
                  <td style={{ fontSize: 12, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                    {formatTime(entry.timestamp)}
                  </td>
                </tr>
                {isExpanded && (
                  <tr>
                    <td colSpan={COLUMNS.length} style={{ padding: 0 }}>
                      <div style={{
                        padding: '12px 16px',
                        background: 'var(--bg-tertiary)',
                        borderLeft: `3px solid ${TYPE_COLORS[entry.entry_type] || 'var(--border)'}`,
                        fontSize: 13,
                        color: 'var(--text-secondary)',
                        lineHeight: 1.6,
                        whiteSpace: 'pre-wrap',
                      }}>
                        {entry.content}
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

export default LabNotebook;
