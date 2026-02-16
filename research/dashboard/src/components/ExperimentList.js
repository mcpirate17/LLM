import React, { useState, useMemo } from 'react';

function formatTime(timestamp) {
  if (!timestamp) return '--';
  return new Date(timestamp * 1000).toLocaleString();
}

function formatDuration(seconds) {
  if (!seconds) return '--';
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

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

function scoreColor(score) {
  if (score >= 70) return 'var(--accent-green)';
  if (score >= 40) return 'var(--accent-yellow)';
  if (score >= 20) return 'var(--accent-orange, #f0883e)';
  return 'var(--accent-red)';
}

const COLUMNS = [
  { key: 'score', label: 'Score' },
  { key: 'rating', label: 'Rating' },
  { key: 'experiment_id', label: 'ID' },
  { key: 'experiment_type', label: 'Type' },
  { key: 'status', label: 'Status' },
  { key: 'n_programs_generated', label: 'Programs' },
  { key: 'n_stage1_passed', label: 'S1 Pass' },
  { key: 'best_loss_ratio', label: 'Best Loss' },
  { key: 'best_novelty_score', label: 'Best Novelty' },
  { key: 'duration_seconds', label: 'Duration' },
  { key: 'timestamp', label: 'Time' },
];

function ExperimentList({ experiments, onSelectExperiment }) {
  const [sortKey, setSortKey] = useState('score');
  const [sortDesc, setSortDesc] = useState(true);

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
            return (
              <tr key={exp.experiment_id}
                style={{ cursor: onSelectExperiment ? 'pointer' : 'default' }}
                onClick={() => onSelectExperiment && onSelectExperiment(exp.experiment_id)}>
                <td style={{ fontWeight: 600, color: isActiveValidation ? 'var(--accent-blue)' : scoreColor(score) }}>
                  {isActiveValidation ? '--' : score}
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
                </td>
                <td>{exp.experiment_type}</td>
                <td>
                  <span className={`badge ${exp.status === 'completed' ? 'pass' :
                    exp.status === 'running' ? 'running' : 'fail'}`}>
                    {exp.status}
                  </span>
                </td>
                <td>{exp.n_programs_generated || 0}</td>
                <td style={{ color: (exp.n_stage1_passed || 0) > 0 ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                  {exp.n_stage1_passed || 0}
                </td>
                <td style={{
                  color: exp.best_loss_ratio != null
                    ? (exp.best_loss_ratio < 0.5 ? 'var(--accent-green)' : exp.best_loss_ratio < 0.8 ? 'var(--accent-yellow)' : 'var(--text-muted)')
                    : 'var(--text-muted)'
                }}>
                  {exp.best_loss_ratio?.toFixed(3) || '--'}
                </td>
                <td style={{
                  color: exp.best_novelty_score != null
                    ? (exp.best_novelty_score > 0.8 ? 'var(--accent-green)' : exp.best_novelty_score > 0.5 ? 'var(--accent-yellow)' : 'var(--text-muted)')
                    : 'var(--text-muted)'
                }}>
                  {exp.best_novelty_score?.toFixed(3) || '--'}
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
