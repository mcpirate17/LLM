import React, { useState, useMemo } from 'react';

/** Rate a program: green (excellent), amber (promising), red (weak) */
function programRating(p) {
  const lr = p.loss_ratio;
  const nov = p.novelty_score || 0;
  const bl = p.baseline_loss_ratio;

  // Beat the transformer baseline = green
  if (bl != null && bl < 1.0) return { color: 'var(--accent-green)', label: 'Excellent', tip: 'Outperforms a standard transformer of the same size', order: 4 };
  // Low loss ratio + high novelty
  if (lr != null && lr < 0.5 && nov > 0.7) return { color: 'var(--accent-green)', label: 'Strong', tip: 'Learns fast and is structurally novel', order: 3 };
  if (lr != null && lr < 0.6) return { color: 'var(--accent-yellow)', label: 'Promising', tip: 'Learns but hasn\'t beaten the transformer baseline yet', order: 2 };
  if (nov > 0.8) return { color: 'var(--accent-yellow)', label: 'Novel', tip: 'Very different structure but learning is modest', order: 1 };
  return { color: 'var(--accent-orange, #f0883e)', label: 'Marginal', tip: 'Passed all stages but performance is weak', order: 0 };
}

/**
 * Compute a 0-100 score for a program.
 * Weights: loss ratio (35%), novelty (25%), baseline ratio (25%), throughput (15%)
 */
function programScore(p) {
  // Loss ratio: lower is better, 0.2 = perfect, 1.0 = bad
  const lossScore = p.loss_ratio != null
    ? Math.max(0, 1 - (p.loss_ratio - 0.2) / 0.8)
    : 0;

  // Novelty: 0-1
  const noveltyScore = p.novelty_score != null
    ? Math.min(p.novelty_score, 1.0)
    : 0;

  // Baseline ratio: < 1 means beats transformer
  const baselineScore = p.baseline_loss_ratio != null
    ? Math.max(0, Math.min(1, 1.5 - p.baseline_loss_ratio))
    : 0;

  // Throughput: normalize roughly (5000 tok/s = great)
  const tpScore = p.throughput_tok_s != null
    ? Math.min(p.throughput_tok_s / 5000, 1.0)
    : 0;

  const score = lossScore * 35 + noveltyScore * 25 + baselineScore * 25 + tpScore * 15;
  return Math.round(Math.max(0, Math.min(100, score)));
}

function scoreColor(score) {
  if (score >= 70) return 'var(--accent-green)';
  if (score >= 40) return 'var(--accent-yellow)';
  if (score >= 20) return 'var(--accent-orange, #f0883e)';
  return 'var(--accent-red)';
}

const COLUMNS_FULL = [
  { key: 'score', label: 'Score' },
  { key: 'rating', label: 'Rating' },
  { key: 'graph_fingerprint', label: 'Fingerprint' },
  { key: 'novelty_score', label: 'Novelty' },
  { key: 'structural_novelty', label: 'Structural' },
  { key: 'behavioral_novelty', label: 'Behavioral' },
  { key: 'loss_ratio', label: 'Loss Ratio' },
  { key: 'param_count', label: 'Params' },
  { key: 'most_similar_to', label: 'Similar To' },
  { key: 'throughput_tok_s', label: 'Throughput' },
];

const COLUMNS_COMPACT = [
  { key: 'score', label: 'Score' },
  { key: 'graph_fingerprint', label: 'Fingerprint' },
  { key: 'novelty_score', label: 'Novelty' },
  { key: 'loss_ratio', label: 'Loss Ratio' },
  { key: 'param_count', label: 'Params' },
];

function TopPrograms({ programs, compact, onSelectProgram }) {
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

  const augmented = useMemo(() => {
    if (!programs) return [];
    return programs.map(p => ({
      ...p,
      _score: programScore(p),
      _rating: programRating(p),
    }));
  }, [programs]);

  const sorted = useMemo(() => {
    const arr = [...augmented];
    arr.sort((a, b) => {
      let va, vb;
      if (sortKey === 'score') {
        va = a._score; vb = b._score;
      } else if (sortKey === 'rating') {
        va = a._rating.order; vb = b._rating.order;
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

  if (!programs || programs.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Top Programs {compact ? '(Preview)' : ''}</div>
        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
          No surviving programs yet.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">
        Top Programs {compact ? `(${programs.length})` : `— ${programs.length} Survivors`}
      </div>
      {!compact && (
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
          These are the best-performing alternative architectures discovered so far. Each one passed all
          evaluation stages and demonstrated actual learning ability. Lower loss ratio = learned faster.
          Higher novelty = more structurally different from known architectures (transformers, SSMs, convnets).
          Click any row to see its full computation graph and metrics.
        </p>
      )}
      <table className="data-table">
        <thead>
          <tr>
            {columns.map(col => (
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
          {sorted.map((p, i) => {
            const rating = p._rating;
            const score = p._score;
            return (
              <tr key={p.result_id || i}
                style={{ cursor: onSelectProgram ? 'pointer' : 'default' }}
                onClick={() => onSelectProgram && onSelectProgram(p.result_id)}>
                <td style={{ fontWeight: 600, color: scoreColor(score) }}>
                  {score}
                </td>
                {!compact && (
                  <td title={rating.tip}>
                    <span style={{
                      display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
                      background: rating.color, marginRight: 6,
                    }} />
                    <span style={{ fontSize: 11, color: rating.color }}>{rating.label}</span>
                  </td>
                )}
                <td style={{ fontFamily: 'monospace', fontSize: 12, color: onSelectProgram ? 'var(--accent-blue)' : 'inherit' }}>
                  {p.graph_fingerprint?.slice(0, 10) || '--'}
                </td>
                <td>
                  <span style={{
                    color: (p.novelty_score || 0) > 0.8 ? 'var(--accent-green)'
                      : (p.novelty_score || 0) > 0.5 ? 'var(--accent-yellow)' : 'var(--text-muted)'
                  }}>
                    {p.novelty_score?.toFixed(3) || '--'}
                  </span>
                </td>
                {!compact && <td>{p.structural_novelty?.toFixed(3) || '--'}</td>}
                {!compact && <td>{p.behavioral_novelty?.toFixed(3) || '--'}</td>}
                <td style={{
                  color: p.loss_ratio != null
                    ? (p.loss_ratio < 0.5 ? 'var(--accent-green)' : p.loss_ratio < 0.7 ? 'var(--accent-yellow)' : 'var(--accent-orange, #f0883e)')
                    : 'var(--text-muted)'
                }}>
                  {p.loss_ratio?.toFixed(4) || '--'}
                </td>
                <td>{p.param_count ? `${(p.param_count / 1e6).toFixed(1)}M` : '--'}</td>
                {!compact && <td>{p.most_similar_to || '--'}</td>}
                {!compact && <td>{p.throughput_tok_s ? `${Number(p.throughput_tok_s).toFixed(0)} tok/s` : '--'}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
      {!compact && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8, display: 'flex', gap: 16 }}>
          <span><span style={{ color: 'var(--accent-green)' }}>Green</span> = outperforms transformer or high novelty + fast learning</span>
          <span><span style={{ color: 'var(--accent-yellow)' }}>Amber</span> = promising but hasn't beaten baseline</span>
          <span>Loss ratio: lower = better (how much loss decreased during training)</span>
        </div>
      )}
      {onSelectProgram && (
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: compact ? 8 : 0, textAlign: 'right' }}>
          Click a row to view program details
        </div>
      )}
    </div>
  );
}

export default TopPrograms;
