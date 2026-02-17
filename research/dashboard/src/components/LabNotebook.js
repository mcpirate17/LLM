import React, { useEffect, useState, useMemo } from 'react';
import { formatTime, scoreColor } from '../utils/format';
import useCopyToClipboard from '../hooks/useCopyToClipboard';

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

const CRITIQUE_VERDICT_STYLES = {
  proceed: { color: 'var(--accent-green)', label: 'Proceed', icon: '\u2714' },
  caution: { color: 'var(--accent-yellow)', label: 'Caution', icon: '\u26A0' },
  revise: { color: 'var(--accent-red)', label: 'Revise', icon: '\u2718' },
};

function hypothesisProvenanceLabel(source) {
  if (source === 'llm_context') return 'LLM + Context';
  if (source === 'structured_hypothesis') return 'LLM Structured';
  if (source === 'rule_based_fallback') return 'Rule Fallback';
  if (source === 'rule_based') return 'Rule-Based';
  if (source === 'user_input') return 'User Input';
  if (source === 'runner_template') return 'Runner Template';
  return null;
}

function hypothesisConfidence(metadata) {
  if (metadata?.confidence != null) return metadata.confidence;
  if (metadata?.critique_confidence != null) return metadata.critique_confidence;
  return 'not provided';
}

function hypothesisCritiqueValue(metadata) {
  const critique = metadata?.preflight_critique || metadata?.critique;
  if (!critique) return null;
  if (typeof critique === 'string') return critique;
  if (typeof critique === 'object') {
    const verdict = critique.verdict || 'unknown';
    const gate = critique.gate || 'n/a';
    const concerns = Array.isArray(critique.concerns) ? critique.concerns : [];
    return `${verdict} (gate=${gate})${concerns.length ? ` — ${concerns[0]}` : ''}`;
  }
  return null;
}

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

function entryScoreBreakdown(entry) {
  const typeScore = ((TYPE_ORDER[entry.entry_type] || 0) / 6) * 50;
  const contentLen = (entry.content || '').length;
  const contentScore = Math.min(contentLen / 500, 1.0) * 30;
  const tagScore = entry.tags ? 10 : 0;
  const expScore = entry.experiment_id ? 10 : 0;
  return {
    type: typeScore,
    content: contentScore,
    tags: tagScore,
    experiment: expScore,
  };
}

const COLUMNS = [
  { key: '_score', label: 'Score' },
  { key: 'entry_type', label: 'Type' },
  { key: 'title', label: 'Title' },
  { key: 'content', label: 'Content' },
  { key: 'tags', label: 'Tags' },
  { key: 'timestamp', label: 'Time' },
];

const LAB_NOTEBOOK_SORT_PREFS_KEY = 'dashboard.lab-notebook.sort.v1';

function LabNotebook({ entries, onSelectExperiment }) {
  const [sortKey, setSortKey] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(LAB_NOTEBOOK_SORT_PREFS_KEY) || '{}');
      if (typeof stored.sortKey === 'string' && COLUMNS.some((column) => column.key === stored.sortKey)) {
        return stored.sortKey;
      }
    } catch {}
    return '_score';
  });
  const [sortDesc, setSortDesc] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(LAB_NOTEBOOK_SORT_PREFS_KEY) || '{}');
      if (typeof stored.sortDesc === 'boolean') {
        return stored.sortDesc;
      }
    } catch {}
    return true;
  });
  const [expandedId, setExpandedId] = useState(null);
  const [copiedValue, copyText] = useCopyToClipboard();

  useEffect(() => {
    localStorage.setItem(LAB_NOTEBOOK_SORT_PREFS_KEY, JSON.stringify({ sortKey, sortDesc }));
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
    if (!entries) return [];
    return entries.map(e => {
      let parsedMetadata = e.metadata;
      if (!parsedMetadata || typeof parsedMetadata !== 'object') {
        try {
          parsedMetadata = e.metadata_json ? JSON.parse(e.metadata_json) : {};
        } catch {
          parsedMetadata = {};
        }
      }
      return { ...e, metadata: parsedMetadata || {}, _score: entryScore(e) };
    });
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

  const latestTimestamp = useMemo(() => {
    if (!entries || entries.length === 0) return null;
    const timestamps = entries.map((entry) => entry.timestamp).filter((timestamp) => timestamp != null);
    if (timestamps.length === 0) return null;
    return Math.max(...timestamps);
  }, [entries]);

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
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
        Latest entry: {latestTimestamp ? formatTime(latestTimestamp) : 'not available'}
      </div>
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
            const provenanceSource = entry.metadata?.source;
            const provenanceLabel = hypothesisProvenanceLabel(provenanceSource);
            const critique = entry.metadata?.preflight_critique || entry.metadata?.critique;
            const critiqueObject = critique && typeof critique === 'object' ? critique : null;
            const critiqueText = hypothesisCritiqueValue(entry.metadata);
            const critiqueStyle = critiqueObject?.verdict
              ? (CRITIQUE_VERDICT_STYLES[critiqueObject.verdict] || CRITIQUE_VERDICT_STYLES.caution)
              : null;

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
                    <span title={`Type ${(entryScoreBreakdown(entry).type || 0).toFixed(1)}/50 | Content ${(entryScoreBreakdown(entry).content || 0).toFixed(1)}/30 | Tags ${(entryScoreBreakdown(entry).tags || 0).toFixed(1)}/10 | Experiment ${(entryScoreBreakdown(entry).experiment || 0).toFixed(1)}/10`}>
                      {score}
                    </span>
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
                    {entry.entry_type === 'hypothesis' && provenanceLabel && (
                      <span style={{
                        marginLeft: 6,
                        fontSize: 10,
                        color: 'var(--text-muted)',
                        border: '1px solid var(--border)',
                        borderRadius: 3,
                        padding: '1px 5px',
                        whiteSpace: 'nowrap',
                      }}>
                        {provenanceLabel}
                      </span>
                    )}
                    {entry.entry_type === 'hypothesis' && critiqueStyle && (
                      <span
                        title={`Preflight review: ${critiqueStyle.label}${critiqueObject?.concerns?.length ? ' — ' + critiqueObject.concerns[0] : ''}`}
                        style={{
                          marginLeft: 6,
                          fontSize: 10,
                          color: critiqueStyle.color,
                          border: `1px solid ${critiqueStyle.color}55`,
                          background: `${critiqueStyle.color}11`,
                          borderRadius: 3,
                          padding: '1px 5px',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {critiqueStyle.icon} {critiqueStyle.label}
                      </span>
                    )}
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
                        {entry.entry_type === 'hypothesis' && provenanceLabel && (
                          <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-muted)' }}>
                            <div><strong>Hypothesis provenance:</strong> {provenanceLabel}</div>
                            <div><strong>Context used:</strong> {entry.metadata?.used_context ? 'yes' : 'no'}</div>
                            <div><strong>Review status:</strong> {entry.metadata?.review_status || 'not provided'}</div>
                            <div><strong>Confidence:</strong> {hypothesisConfidence(entry.metadata)}</div>
                          </div>
                        )}
                        {entry.entry_type === 'hypothesis' && critiqueStyle && critiqueObject && (
                          <div style={{
                            marginTop: 10,
                            padding: '8px 10px',
                            borderRadius: 6,
                            border: `1px solid ${critiqueStyle.color}`,
                            background: `${critiqueStyle.color}11`,
                            fontSize: 12,
                            lineHeight: 1.5,
                          }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: (critiqueObject.concerns?.length || critiqueObject.suggestions?.length) ? 6 : 0 }}>
                              <span style={{ fontSize: 14 }}>{critiqueStyle.icon}</span>
                              <strong style={{ color: critiqueStyle.color }}>Preflight Review: {critiqueStyle.label}</strong>
                              {critiqueObject.confidence != null && (
                                <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 'auto' }}>
                                  confidence {(critiqueObject.confidence * 100).toFixed(0)}%
                                </span>
                              )}
                            </div>
                            {critiqueObject.concerns?.length > 0 && (
                              <div style={{ marginBottom: critiqueObject.suggestions?.length ? 4 : 0 }}>
                                {critiqueObject.concerns.map((c, ci) => (
                                  <div key={ci} style={{ color: 'var(--text-secondary)', paddingLeft: 20 }}>
                                    &bull; {c}
                                  </div>
                                ))}
                              </div>
                            )}
                            {critiqueObject.suggestions?.length > 0 && (
                              <div>
                                {critiqueObject.suggestions.map((s, si) => (
                                  <div key={si} style={{ color: 'var(--text-muted)', paddingLeft: 20, fontStyle: 'italic' }}>
                                    &rarr; {s}
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                        {entry.entry_type === 'hypothesis' && !critiqueStyle && critiqueText && (
                          <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-muted)' }}>
                            <strong>Critique:</strong> {critiqueText}
                          </div>
                        )}
                        <div style={{ marginTop: 10, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                          {entry.experiment_id && onSelectExperiment && (
                            <button
                              className="refresh-btn"
                              style={{ fontSize: 11, padding: '3px 8px' }}
                              onClick={(e) => {
                                e.stopPropagation();
                                onSelectExperiment(entry.experiment_id);
                              }}
                              aria-label={`Open linked experiment ${entry.experiment_id}`}
                            >
                              Open Experiment
                            </button>
                          )}
                          {entry.experiment_id && (
                            <button
                              className="refresh-btn"
                              style={{ fontSize: 11, padding: '3px 8px' }}
                              onClick={(e) => {
                                e.stopPropagation();
                                copyText(entry.experiment_id);
                              }}
                              aria-label={`Copy experiment id ${entry.experiment_id}`}
                            >
                              {copiedValue === entry.experiment_id ? 'Copied Exp ID' : 'Copy Exp ID'}
                            </button>
                          )}
                          {entry.entry_id && (
                            <button
                              className="refresh-btn"
                              style={{ fontSize: 11, padding: '3px 8px' }}
                              onClick={(e) => {
                                e.stopPropagation();
                                copyText(entry.entry_id);
                              }}
                              aria-label={`Copy entry id ${entry.entry_id}`}
                            >
                              {copiedValue === entry.entry_id ? 'Copied Entry ID' : 'Copy Entry ID'}
                            </button>
                          )}
                        </div>
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
