import React, { useEffect, useState, useMemo } from 'react';
import { formatTime, formatDuration, scoreColor } from '../utils/format';
import { noveltyColor, reliabilityColor } from '../utils/colors';
import { experimentScore, experimentScoreBreakdown } from '../utils/scoringEngine';
import { filterRowsByQuery } from '../utils/tableFiltering';
import useCopyToClipboard from '../hooks/useCopyToClipboard';

/** Color-code experiment outcome: green (good), amber (ok), red (bad) */
function experimentRating(exp) {
  if (exp?.status === 'running' && exp?.experiment_type === 'validation') {
    return {
      color: 'var(--accent-blue)',
      label: 'Validating',
      tip: 'Validation run in progress — rating shown after aggregate seed results arrive',
    };
  }

  const n = exp.n_programs_generated || 0;
  const s1 = exp.n_stage1_passed || 0;
  const rate = n > 0 ? s1 / n : 0;

  if (s1 > 2 || rate > 0.05) return { color: 'var(--accent-green)', label: 'Strong', tip: 'Multiple architectures learned — productive experiment' };
  if (s1 > 0) return { color: 'var(--accent-yellow)', label: 'Some', tip: 'At least one learnable architecture found' };
  if ((exp.n_stage0_passed || 0) > n * 0.3) return { color: 'var(--accent-orange, #f0883e)', label: 'Weak', tip: 'Programs compiled but none learned — need better op combinations' };
  return { color: 'var(--accent-red)', label: 'Poor', tip: 'Most programs failed to compile — grammar too aggressive' };
}

function metricText(value, fallbackReason, formatter) {
  if (value == null) return fallbackReason;
  return formatter(value);
}


function experimentMetricChips(exp) {
  const nPrograms = exp.n_programs_generated || 0;
  const s1 = exp.n_stage1_passed || 0;
  const evidenceReliability = nPrograms >= 100 ? 'high' : nPrograms >= 30 ? 'medium' : 'low';
  return [
    {
      label: 'Loss',
      source: exp.best_loss_ratio != null ? 'measured' : 'not-evaluated',
      reliability: exp.best_loss_ratio != null ? evidenceReliability : 'low',
    },
    {
      label: 'Novelty',
      source: exp.best_novelty_score != null ? 'heuristic' : 'insufficient-data',
      reliability: s1 > 0 ? evidenceReliability : 'low',
    },
    {
      label: 'Baseline',
      source: 'not-available',
      reliability: 'low',
    },
  ];
}

const COLUMNS = [
  { key: 'score', label: 'Score' },
  { key: 'rating', label: 'Rating' },
  { key: 'experiment_id', label: 'ID' },
  { key: 'experiment_type', label: 'Type' },
  { key: 'hypothesis', label: 'Hypothesis' },
  { key: 'status', label: 'Status' },
  { key: 'stage_funnel', label: 'Funnel' },
  { key: 'n_stage1_passed', label: 'S1 Pass' },
  { key: 'best_loss_ratio', label: 'Best Loss' },
  { key: 'best_novelty_score', label: 'Best Novelty' },
  { key: 'aria_summary', label: 'Outcome' },
  { key: 'duration_seconds', label: 'Duration' },
  { key: 'timestamp', label: 'Time' },
];

/** Mini stage funnel: generated -> compiled -> stage0.5 -> S1 */
function StageFunnel({ generated, s0, s05, s1 }) {
  if (!generated) return <span style={{ color: 'var(--text-muted)' }}>--</span>;
  const stages = [
    { label: 'Gen', value: generated, color: 'var(--text-secondary)' },
  ];
  if (s0 != null) stages.push({ label: 'S0', value: s0, color: s0 > 0 ? 'var(--accent-blue)' : 'var(--text-muted)' });
  if (s05 != null) stages.push({ label: 'S0.5', value: s05, color: s05 > 0 ? 'var(--accent-yellow)' : 'var(--text-muted)' });
  stages.push({ label: 'S1', value: s1, color: s1 > 0 ? 'var(--accent-green)' : 'var(--text-muted)' });

  return (
    <span style={{ fontSize: 11, whiteSpace: 'nowrap' }}>
      {stages.map((s, i) => (
        <React.Fragment key={s.label}>
          {i > 0 && <span style={{ color: 'var(--text-muted)', margin: '0 2px' }}>{'\u203A'}</span>}
          <span style={{ color: s.color, fontWeight: s.label === 'S1' ? 600 : 400 }}>{s.value}</span>
        </React.Fragment>
      ))}
    </span>
  );
}

const EXPERIMENT_LIST_SORT_PREFS_KEY = 'dashboard.experiment-list.sort.v1';
const EXPERIMENT_LIST_EXPERT_KEY = 'dashboard.experiment-list.expert.v1';

function ExperimentList({ experiments, onSelectExperiment, onRefresh }) {
  const [showExpertColumns, setShowExpertColumns] = useState(() => {
    try {
      return localStorage.getItem(EXPERIMENT_LIST_EXPERT_KEY) === 'true';
    } catch {
      return false;
    }
  });

  const [sortKey, setSortKey] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(EXPERIMENT_LIST_SORT_PREFS_KEY) || '{}');
      if (typeof stored.sortKey === 'string' && COLUMNS.some((column) => column.key === stored.sortKey)) {
        return stored.sortKey;
      }
    } catch {}
    return 'score';
  });
  const [sortDesc, setSortDesc] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(EXPERIMENT_LIST_SORT_PREFS_KEY) || '{}');
      if (typeof stored.sortDesc === 'boolean') {
        return stored.sortDesc;
      }
    } catch {}
    return true;
  });
  const [filterQuery, setFilterQuery] = useState('');
  const [copiedValue, copyText] = useCopyToClipboard();
  const [cancellingId, setCancellingId] = useState(null);
  const [rerunningId, setRerunningId] = useState(null);
  const [confirmingAction, setConfirmingAction] = useState(null); // { id, type }
  const [inlineError, setInlineError] = useState(null); // { id, message }

  const handleCancel = async (e, experimentId) => {
    e.stopPropagation();
    if (!confirmingAction || confirmingAction.id !== experimentId || confirmingAction.type !== 'cancel') {
      setConfirmingAction({ id: experimentId, type: 'cancel' });
      return;
    }
    setConfirmingAction(null);
    setCancellingId(experimentId);
    try {
      const res = await fetch(`/api/experiments/${experimentId}/cancel`, { method: 'POST' });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setInlineError({ id: experimentId, message: data.error || 'Failed to cancel experiment' });
      } else if (onRefresh) {
        onRefresh();
      }
    } catch (err) {
      setInlineError({ id: experimentId, message: 'Network error cancelling experiment' });
    } finally {
      setCancellingId(null);
    }
  };

  const handleRerun = async (e, experimentId) => {
    e.stopPropagation();
    if (!confirmingAction || confirmingAction.id !== experimentId || confirmingAction.type !== 'rerun') {
      setConfirmingAction({ id: experimentId, type: 'rerun' });
      return;
    }
    setConfirmingAction(null);
    setRerunningId(experimentId);
    try {
      const res = await fetch(`/api/experiments/${experimentId}/rerun`, { method: 'POST' });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setInlineError({ id: experimentId, message: data.error || 'Failed to rerun experiment' });
      } else if (onRefresh) {
        onRefresh();
      }
    } catch (err) {
      setInlineError({ id: experimentId, message: 'Network error rerunning experiment' });
    } finally {
      setRerunningId(null);
    }
  };

  useEffect(() => {
    localStorage.setItem(EXPERIMENT_LIST_SORT_PREFS_KEY, JSON.stringify({ sortKey, sortDesc }));
  }, [sortKey, sortDesc]);

  useEffect(() => {
    localStorage.setItem(EXPERIMENT_LIST_EXPERT_KEY, String(showExpertColumns));
  }, [showExpertColumns]);

  const signalKeys = new Set(['score', 'n_stage1_passed', 'best_loss_ratio', 'best_novelty_score', 'status', 'timestamp', 'experiment_id']);
  const visibleColumns = COLUMNS.filter(col => showExpertColumns || signalKeys.has(col.key));

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDesc(!sortDesc);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  // Augment experiments with computed score
  const augmented = useMemo(() => {
    if (!experiments) return [];
    return experiments.map(exp => ({
      ...exp,
      _score: experimentScore(exp),
      _rating: experimentRating(exp),
    }));
  }, [experiments]);

  const filtered = useMemo(() => (
    filterRowsByQuery(augmented, filterQuery, [
      'experiment_id',
      'experiment_type',
      'hypothesis',
      'status',
      'aria_summary',
    ])
  ), [augmented, filterQuery]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      let va, vb;
      if (sortKey === 'score') {
        va = a._score; vb = b._score;
      } else if (sortKey === 'rating') {
        // Map label to numeric for sorting
        const order = { Strong: 4, Some: 3, Validating: 2, Weak: 1, Poor: 0 };
        va = order[a._rating.label] ?? -1;
        vb = order[b._rating.label] ?? -1;
      } else if (sortKey === 'stage_funnel') {
        // Sort by compilation rate (stage0/generated)
        va = (a.n_programs_generated || 0) > 0 ? (a.n_stage0_passed || 0) / a.n_programs_generated : 0;
        vb = (b.n_programs_generated || 0) > 0 ? (b.n_stage0_passed || 0) / b.n_programs_generated : 0;
      } else {
        va = a[sortKey]; vb = b[sortKey];
      }
      // Nulls/undefined sort to bottom
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') {
        return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      }
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [filtered, sortKey, sortDesc]);

  if (!experiments || experiments.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Experiments</div>
        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
          No experiments yet. Run: python -m research --mode=synthesize --n 100
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <span>Experiments</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            placeholder="Filter experiments"
            style={{
              fontSize: 11,
              padding: '4px 8px',
              borderRadius: 4,
              border: '1px solid var(--border)',
              background: 'var(--bg-tertiary)',
              color: 'var(--text-primary)',
              minWidth: 180,
            }}
          />
          <button
            className="refresh-btn"
            style={{ fontSize: 11, padding: '3px 10px' }}
            onClick={() => setShowExpertColumns(!showExpertColumns)}
          >
            {showExpertColumns ? 'Hide noise' : 'Show expert columns'}
          </button>
        </div>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Each experiment generates a batch of random computation graphs, then tests whether they can
        function as LLM layers. S1 Pass = architectures that actually learned from data.
        The system formulates a hypothesis before each experiment and adjusts strategy based on outcomes.
        Click any row for the full breakdown.
      </p>
      <table className="data-table">
        <thead>
          <tr>
            {visibleColumns.map(col => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                aria-label={`Sort op success table by ${col.label}${sortKey === col.key ? `, currently ${sortDesc ? 'descending' : 'ascending'}` : ''}`}
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
          {sorted.map(exp => {
            const rating = exp._rating;
            const score = exp._score;
            const isActiveValidation = exp.status === 'running' && exp.experiment_type === 'validation';
            const nUsed = exp.n_programs_generated || 0;
            const s1Count = exp.n_stage1_passed || 0;
            const chips = experimentMetricChips(exp);

            return (
              <tr key={exp.experiment_id}
                style={{ cursor: onSelectExperiment ? 'pointer' : 'default' }}
                onClick={() => onSelectExperiment && onSelectExperiment(exp.experiment_id)}>
                {visibleColumns.map(col => {
                  if (col.key === 'score') {
                    return (
                      <td key="score" style={{ fontWeight: 600, color: isActiveValidation ? 'var(--accent-blue)' : scoreColor(score) }}>
                        {isActiveValidation ? (
                          'running validation'
                        ) : (
                          <span title={`S1 rate ${(experimentScoreBreakdown(exp).passRate || 0).toFixed(1)}/40 | Loss ${(experimentScoreBreakdown(exp).loss || 0).toFixed(1)}/30 | Novelty ${(experimentScoreBreakdown(exp).novelty || 0).toFixed(1)}/20 | Completion ${(experimentScoreBreakdown(exp).completion || 0).toFixed(1)}/10`}>
                            {score}
                          </span>
                        )}
                      </td>
                    );
                  }
                  if (col.key === 'rating') {
                    return (
                      <td key="rating" title={rating.tip}>
                        <span style={{
                          display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
                          background: rating.color, marginRight: 6,
                        }} />
                        <span style={{ fontSize: 11, color: rating.color }}>{rating.label}</span>
                      </td>
                    );
                  }
                  if (col.key === 'experiment_id') {
                    return (
                      <td key="id" style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--accent-blue)' }}>
                        {exp.experiment_id}
                        {exp.experiment_id && (
                          <button
                            className="refresh-btn"
                            style={{ fontSize: 10, padding: '1px 5px', marginLeft: 6 }}
                            onClick={(e) => {
                              e.stopPropagation();
                              copyText(exp.experiment_id);
                            }}
                            aria-label={`Copy experiment id ${exp.experiment_id}`}
                          >
                            {copiedValue === exp.experiment_id ? 'Copied' : 'Copy'}
                          </button>
                        )}
                      </td>
                    );
                  }
                  if (col.key === 'experiment_type') {
                    return <td key="type">{exp.experiment_type}</td>;
                  }
                  if (col.key === 'hypothesis') {
                    return (
                      <td key="hypothesis" style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 12, color: 'var(--text-secondary)' }}
                          title={exp.hypothesis || 'No hypothesis'}>
                        {exp.hypothesis
                          ? (exp.hypothesis.length > 60 ? exp.hypothesis.slice(0, 60) + '...' : exp.hypothesis)
                          : <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>none</span>
                        }
                      </td>
                    );
                  }
                  if (col.key === 'status') {
                    return (
                      <td key="status">
                        <span className={`badge ${exp.status === 'completed' ? 'pass' :
                          exp.status === 'running' ? 'running' : 'fail'}`}>
                          {exp.status}
                        </span>
                        {exp.status === 'running' && (
                          confirmingAction?.id === exp.experiment_id && confirmingAction?.type === 'cancel' ? (
                            <span style={{ fontSize: 10, marginLeft: 6 }}>
                              <span style={{ color: 'var(--accent-yellow)' }}>Cancel?</span>
                              <button className="refresh-btn" style={{ fontSize: 10, padding: '1px 5px', marginLeft: 4, color: 'var(--accent-red)', borderColor: 'var(--accent-red)' }} onClick={(e) => handleCancel(e, exp.experiment_id)}>Yes</button>
                              <button className="refresh-btn" style={{ fontSize: 10, padding: '1px 5px', marginLeft: 2 }} onClick={(e) => { e.stopPropagation(); setConfirmingAction(null); }}>No</button>
                            </span>
                          ) : (
                            <button
                              className="refresh-btn"
                              style={{
                                fontSize: 10, padding: '1px 5px', marginLeft: 6,
                                color: 'var(--accent-red)', borderColor: 'var(--accent-red)',
                              }}
                              disabled={cancellingId === exp.experiment_id}
                              onClick={(e) => handleCancel(e, exp.experiment_id)}
                              aria-label="Cancel experiment"
                            >
                              {cancellingId === exp.experiment_id ? '...' : 'Cancel'}
                            </button>
                          )
                        )}
                        {(exp.status === 'running' || exp.status === 'failed') && (
                          confirmingAction?.id === exp.experiment_id && confirmingAction?.type === 'rerun' ? (
                            <span style={{ fontSize: 10, marginLeft: 6 }}>
                              <span style={{ color: 'var(--accent-yellow)' }}>Rerun?</span>
                              <button className="refresh-btn" style={{ fontSize: 10, padding: '1px 5px', marginLeft: 4 }} onClick={(e) => handleRerun(e, exp.experiment_id)}>Yes</button>
                              <button className="refresh-btn" style={{ fontSize: 10, padding: '1px 5px', marginLeft: 2 }} onClick={(e) => { e.stopPropagation(); setConfirmingAction(null); }}>No</button>
                            </span>
                          ) : (
                            <button
                              className="refresh-btn"
                              style={{ fontSize: 10, padding: '1px 5px', marginLeft: 6 }}
                              disabled={rerunningId === exp.experiment_id}
                              onClick={(e) => handleRerun(e, exp.experiment_id)}
                              aria-label="Rerun experiment"
                            >
                              {rerunningId === exp.experiment_id ? '...' : 'Rerun'}
                            </button>
                          )
                        )}
                        {inlineError?.id === exp.experiment_id && (
                          <span style={{ fontSize: 10, marginLeft: 6, color: 'var(--accent-red)' }}>
                            {inlineError.message}
                            <button className="refresh-btn" style={{ fontSize: 9, padding: '0 4px', marginLeft: 4 }} onClick={(e) => { e.stopPropagation(); setInlineError(null); }}>&times;</button>
                          </span>
                        )}
                      </td>
                    );
                  }
                  if (col.key === 'stage_funnel') {
                    return (
                      <td key="funnel" title={`${exp.n_programs_generated || 0} generated \u2192 ${exp.n_stage0_passed ?? '?'} compiled \u2192 ${exp.n_stage05_passed ?? '?'} stage0.5 \u2192 ${exp.n_stage1_passed || 0} S1`}>
                        <StageFunnel
                          generated={exp.n_programs_generated || 0}
                          s0={exp.n_stage0_passed}
                          s05={exp.n_stage05_passed}
                          s1={exp.n_stage1_passed || 0}
                        />
                      </td>
                    );
                  }
                  if (col.key === 'n_stage1_passed') {
                    return (
                      <td key="s1" style={{ color: s1Count > 0 ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                        {s1Count}
                        <span style={{ marginLeft: 4, fontSize: 11, color: 'var(--text-muted)' }}>
                          / {nUsed}
                          {nUsed > 0 && ` (${((s1Count / nUsed) * 100).toFixed(1)}%)`}
                        </span>
                      </td>
                    );
                  }
                  if (col.key === 'best_loss_ratio') {
                    return (
                      <td key="loss" style={{
                        color: exp.best_loss_ratio != null
                          ? (exp.best_loss_ratio < 0.5 ? 'var(--accent-green)' : exp.best_loss_ratio < 0.8 ? 'var(--accent-yellow)' : 'var(--text-muted)')
                          : 'var(--text-muted)'
                      }}>
                        {metricText(
                          exp.best_loss_ratio,
                          (exp.n_stage1_passed || 0) > 0 ? 'not computed' : 'not yet evaluated',
                          (v) => v.toFixed(3),
                        )}
                      </td>
                    );
                  }
                  if (col.key === 'best_novelty_score') {
                    return (
                      <td key="novelty" style={{ color: noveltyColor(exp.best_novelty_score) }}>
                        {metricText(
                          exp.best_novelty_score,
                          (exp.n_stage1_passed || 0) > 0 ? 'not computed' : 'insufficient data',
                          (v) => v.toFixed(3),
                        )}
                        <div style={{ marginTop: 4, display: 'flex', gap: 4, flexWrap: 'wrap', maxWidth: 220 }}>
                          {chips.map(chip => (
                            <span
                              key={`${exp.experiment_id}-${chip.label}`}
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
                    );
                  }
                  if (col.key === 'aria_summary') {
                    return (
                      <td key="outcome" style={{ maxWidth: 240, fontSize: 12, color: 'var(--text-secondary)' }}
                          title={exp.aria_summary || exp.research_question || ''}>
                        {exp.aria_summary
                          ? (exp.aria_summary.length > 80
                              ? exp.aria_summary.slice(0, 80) + '...'
                              : exp.aria_summary)
                          : exp.research_question
                            ? <span style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>
                                {exp.research_question.length > 60
                                  ? exp.research_question.slice(0, 60) + '...'
                                  : exp.research_question}
                              </span>
                            : <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>--</span>
                        }
                      </td>
                    );
                  }
                  if (col.key === 'duration_seconds') {
                    return <td key="duration">{formatDuration(exp.duration_seconds)}</td>;
                  }
                  if (col.key === 'timestamp') {
                    return (
                      <td key="time" style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                        {formatTime(exp.timestamp)}
                      </td>
                    );
                  }
                  return <td key={col.key}>--</td>;
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8, display: 'flex', gap: 16 }}>
        <span><span style={{ color: 'var(--accent-green)' }}>Green</span> = good results (learnable architectures found)</span>
        <span><span style={{ color: 'var(--accent-yellow)' }}>Amber</span> = some results (limited learning)</span>
        <span><span style={{ color: 'var(--accent-red)' }}>Red</span> = no learning (grammar needs adjustment)</span>
        {onSelectExperiment && <span style={{ marginLeft: 'auto' }}>Click a row for details</span>}
      </div>
    </div>
  );
}

export default ExperimentList;
