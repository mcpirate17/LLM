import React, { useEffect, useState, useMemo } from 'react';
import { formatTime, formatDuration, scoreColor } from '../utils/format';
import { noveltyColor } from '../utils/colors';
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

/**
 * Compute a 0-100 numeric score for an experiment.
 * Weights: S1 pass rate (40%), best loss ratio (30%), best novelty (20%), completion (10%)
 */
function experimentScore(exp) {
  if (exp?.status === 'running' && exp?.experiment_type === 'validation') {
    return 25;
  }

  const n = exp.n_programs_generated || 0;
  const s1 = exp.n_stage1_passed || 0;

  // S1 pass rate: 0-1, scaled so 10% pass rate = 1.0
  const passRate = n > 0 ? Math.min(s1 / n / 0.10, 1.0) : 0;

  // Best loss ratio: lower is better, 0.2 = perfect, 1.0 = bad
  const lossScore = exp.best_loss_ratio != null
    ? Math.max(0, 1 - (exp.best_loss_ratio - 0.2) / 0.8)
    : 0;

  // Novelty: 0-1, already scaled
  const noveltyScore = exp.best_novelty_score != null
    ? Math.min(exp.best_novelty_score, 1.0)
    : 0;

  // Completion bonus
  const completionScore = exp.status === 'completed' ? 1.0 : 0;

  const score = (passRate * 40 + lossScore * 30 + noveltyScore * 20 + completionScore * 10);
  return Math.round(Math.max(0, Math.min(100, score)));
}

function experimentScoreBreakdown(exp) {
  const n = exp.n_programs_generated || 0;
  const s1 = exp.n_stage1_passed || 0;
  const passRate = n > 0 ? Math.min(s1 / n / 0.10, 1.0) : 0;
  const lossScore = exp.best_loss_ratio != null
    ? Math.max(0, 1 - (exp.best_loss_ratio - 0.2) / 0.8)
    : 0;
  const noveltyScore = exp.best_novelty_score != null
    ? Math.min(exp.best_novelty_score, 1.0)
    : 0;
  const completionScore = exp.status === 'completed' ? 1.0 : 0;

  return {
    passRate: passRate * 40,
    loss: lossScore * 30,
    novelty: noveltyScore * 20,
    completion: completionScore * 10,
  };
}

function metricText(value, fallbackReason, formatter) {
  if (value == null) return fallbackReason;
  return formatter(value);
}

function reliabilityColor(level) {
  if (level === 'high') return 'var(--accent-green)';
  if (level === 'medium') return 'var(--accent-yellow)';
  return 'var(--accent-red)';
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
  { key: 'n_programs_generated', label: 'Programs' },
  { key: 'n_stage1_passed', label: 'S1 Pass' },
  { key: 'best_loss_ratio', label: 'Best Loss' },
  { key: 'best_novelty_score', label: 'Best Novelty' },
  { key: 'duration_seconds', label: 'Duration' },
  { key: 'timestamp', label: 'Time' },
];

const EXPERIMENT_LIST_SORT_PREFS_KEY = 'dashboard.experiment-list.sort.v1';

function ExperimentList({ experiments, onSelectExperiment }) {
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
  const [copiedValue, copyText] = useCopyToClipboard();

  useEffect(() => {
    localStorage.setItem(EXPERIMENT_LIST_SORT_PREFS_KEY, JSON.stringify({ sortKey, sortDesc }));
  }, [sortKey, sortDesc]);

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

  const sorted = useMemo(() => {
    const arr = [...augmented];
    arr.sort((a, b) => {
      let va, vb;
      if (sortKey === 'score') {
        va = a._score; vb = b._score;
      } else if (sortKey === 'rating') {
        // Map label to numeric for sorting
        const order = { Strong: 4, Some: 3, Validating: 2, Weak: 1, Poor: 0 };
        va = order[a._rating.label] ?? -1;
        vb = order[b._rating.label] ?? -1;
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
  }, [augmented, sortKey, sortDesc]);

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
      <div className="card-title">Experiments</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Each experiment generates a batch of random computation graphs, then tests whether they can
        function as LLM layers. S1 Pass = architectures that actually learned from data.
        The system formulates a hypothesis before each experiment and adjusts strategy based on outcomes.
        Click any row for the full breakdown.
      </p>
      <table className="data-table">
        <thead>
          <tr>
            {COLUMNS.map(col => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
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
            const chips = experimentMetricChips(exp);
            return (
              <tr key={exp.experiment_id}
                style={{ cursor: onSelectExperiment ? 'pointer' : 'default' }}
                onClick={() => onSelectExperiment && onSelectExperiment(exp.experiment_id)}>
                <td style={{ fontWeight: 600, color: isActiveValidation ? 'var(--accent-blue)' : scoreColor(score) }}>
                  {isActiveValidation ? (
                    'running validation'
                  ) : (
                    <span title={`S1 rate ${(experimentScoreBreakdown(exp).passRate || 0).toFixed(1)}/40 | Loss ${(experimentScoreBreakdown(exp).loss || 0).toFixed(1)}/30 | Novelty ${(experimentScoreBreakdown(exp).novelty || 0).toFixed(1)}/20 | Completion ${(experimentScoreBreakdown(exp).completion || 0).toFixed(1)}/10`}>
                      {score}
                    </span>
                  )}
                </td>
                <td title={rating.tip}>
                  <span style={{
                    display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
                    background: rating.color, marginRight: 6,
                  }} />
                  <span style={{ fontSize: 11, color: rating.color }}>{rating.label}</span>
                </td>
                <td style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--accent-blue)' }}>
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
                <td>{exp.experiment_type}</td>
                <td style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 12, color: 'var(--text-secondary)' }}
                    title={exp.hypothesis || 'No hypothesis'}>
                  {exp.hypothesis
                    ? (exp.hypothesis.length > 60 ? exp.hypothesis.slice(0, 60) + '...' : exp.hypothesis)
                    : <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>none</span>
                  }
                </td>
                <td>
                  <span className={`badge ${exp.status === 'completed' ? 'pass' :
                    exp.status === 'running' ? 'running' : 'fail'}`}>
                    {exp.status}
                  </span>
                </td>
                <td>{exp.n_programs_generated || 0}</td>
                <td style={{ color: (exp.n_stage1_passed || 0) > 0 ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                  {exp.n_stage1_passed || 0}
                  <span style={{ marginLeft: 4, fontSize: 11, color: 'var(--text-muted)' }}>
                    / {exp.n_programs_generated || 0}
                    {(exp.n_programs_generated || 0) > 0 && ` (${(((exp.n_stage1_passed || 0) / exp.n_programs_generated) * 100).toFixed(1)}%)`}
                  </span>
                </td>
                <td style={{
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
                <td style={{ color: noveltyColor(exp.best_novelty_score) }}>
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
                <td>{formatDuration(exp.duration_seconds)}</td>
                <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  {formatTime(exp.timestamp)}
                </td>
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
