import React, { useState, useEffect, useMemo } from 'react';

const API_BASE = process.env.REACT_APP_API_URL || '';

/**
 * LearningPanel — Shows grammar weight evolution, op success rates,
 * learning log timeline, and efficiency frontier.
 */

function GrammarWeightsChart({ defaultWeights, learnedWeights }) {
  if (!defaultWeights) return null;

  const categories = Object.keys(defaultWeights).sort();
  const maxWeight = Math.max(
    ...categories.map(c => Math.max(defaultWeights[c] || 0, (learnedWeights || {})[c] || 0)),
    1
  );

  return (
    <div className="card">
      <div className="card-title">Grammar Weights (Default vs Learned)</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        How likely each type of operation is to appear in a newly generated architecture.
        The system adjusts these weights based on which operation categories produced architectures
        that actually learned. Green = increased (working well), Red = decreased (underperforming).
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {categories.map(cat => {
          const def = defaultWeights[cat] || 0;
          const learned = (learnedWeights || {})[cat];
          const hasLearned = learned !== undefined && learned !== null;
          return (
            <div key={cat} style={{ fontSize: 13 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span style={{ color: 'var(--text-secondary)' }}>
                  {cat.replace(/_/g, ' ')}
                </span>
                <span>
                  <span style={{ color: 'var(--text-muted)' }}>{def.toFixed(1)}</span>
                  {hasLearned && (
                    <span style={{
                      color: learned > def ? 'var(--accent-green)' : learned < def ? 'var(--accent-red)' : 'var(--text-muted)',
                      marginLeft: 8,
                    }}>
                      {learned > def ? '+' : ''}{(learned - def).toFixed(1)} = {learned.toFixed(1)}
                    </span>
                  )}
                </span>
              </div>
              <div style={{ position: 'relative', height: 16, background: 'var(--bg-tertiary)', borderRadius: 4 }}>
                <div style={{
                  position: 'absolute', height: '100%', borderRadius: 4,
                  width: `${(def / maxWeight) * 100}%`,
                  background: 'rgba(88, 166, 255, 0.3)',
                  border: '1px solid var(--accent-blue)',
                }} />
                {hasLearned && (
                  <div style={{
                    position: 'absolute', height: '100%', borderRadius: 4,
                    width: `${(learned / maxWeight) * 100}%`,
                    background: learned > def
                      ? 'rgba(63, 185, 80, 0.3)'
                      : 'rgba(248, 81, 73, 0.3)',
                    border: `1px solid ${learned > def ? 'var(--accent-green)' : 'var(--accent-red)'}`,
                  }} />
                )}
              </div>
            </div>
          );
        })}
      </div>
      {!learnedWeights && (
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, fontStyle: 'italic' }}>
          No learned weights yet. Run more experiments to enable learning.
        </p>
      )}
    </div>
  );
}

/** Rate an op's contribution: green (strong), amber (some), red (weak) */
function opRating(stats) {
  const s1 = stats.s1_rate || 0;
  const s0 = stats.s0_rate || 0;
  if (s1 > 0.15) return { color: 'var(--accent-green)', label: 'Strong', tip: 'This op frequently appears in architectures that learn — a key building block' };
  if (s1 > 0.05) return { color: 'var(--accent-green)', label: 'Good', tip: 'This op contributes to some learnable architectures' };
  if (s1 > 0) return { color: 'var(--accent-yellow)', label: 'Some', tip: 'Rarely leads to learning but has produced at least one survivor' };
  if (s0 > 0.5) return { color: 'var(--accent-orange, #f0883e)', label: 'Compiles', tip: 'Compiles reliably but hasn\'t produced a learnable architecture yet' };
  return { color: 'var(--accent-red)', label: 'Weak', tip: 'Rarely compiles or leads to learning — may be deprioritized' };
}

/**
 * Score an op 0-100.
 * Weights: S1 rate (40%), S0.5 rate (20%), S0 rate (10%), novelty (20%), usage (10%)
 */
function opScore(stats) {
  const s1 = Math.min((stats.s1_rate || 0) / 0.15, 1.0) * 40;
  const s05 = Math.min((stats.s05_rate || 0), 1.0) * 20;
  const s0 = Math.min((stats.s0_rate || 0), 1.0) * 10;
  const nov = Math.min((stats.avg_novelty || 0), 1.0) * 20;
  const usage = Math.min((stats.n_used || 0) / 100, 1.0) * 10;
  return Math.round(Math.max(0, Math.min(100, s1 + s05 + s0 + nov + usage)));
}

function opScoreColor(score) {
  if (score >= 70) return 'var(--accent-green)';
  if (score >= 40) return 'var(--accent-yellow)';
  if (score >= 20) return 'var(--accent-orange, #f0883e)';
  return 'var(--accent-red)';
}

const OP_COLUMNS = [
  { key: '_score', label: 'Score' },
  { key: 'rating', label: 'Rating' },
  { key: 'op', label: 'Op' },
  { key: 'n_used', label: 'Used' },
  { key: 's0_rate', label: 'S0 %' },
  { key: 's05_rate', label: 'S0.5 %' },
  { key: 's1_rate', label: 'S1 %' },
  { key: 'avg_novelty', label: 'Avg Novelty' },
];

const RATING_ORDER = { Strong: 4, Good: 3, Some: 2, Compiles: 1, Weak: 0 };

function OpSuccessTable({ opRates }) {
  const [sortKey, setSortKey] = useState('_score');
  const [sortDesc, setSortDesc] = useState(true);

  const handleSort = (key) => {
    if (sortKey === key) setSortDesc(!sortDesc);
    else { setSortKey(key); setSortDesc(true); }
  };

  const augmented = useMemo(() => {
    if (!opRates || Object.keys(opRates).length === 0) return [];
    return Object.entries(opRates).map(([op, stats]) => ({
      op,
      ...stats,
      _score: opScore(stats),
      _rating: opRating(stats),
    }));
  }, [opRates]);

  const sorted = useMemo(() => {
    const arr = [...augmented];
    arr.sort((a, b) => {
      let va, vb;
      if (sortKey === '_score') { va = a._score; vb = b._score; }
      else if (sortKey === 'rating') { va = RATING_ORDER[a._rating.label] || 0; vb = RATING_ORDER[b._rating.label] || 0; }
      else if (sortKey === 'op') { va = a.op; vb = b.op; }
      else { va = a[sortKey]; vb = b[sortKey]; }
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [augmented, sortKey, sortDesc]);

  if (!opRates || Object.keys(opRates).length === 0) {
    return (
      <div className="card">
        <div className="card-title">Op Success Rates</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>No data yet.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Op Success Rates</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Every candidate architecture is built by combining these primitive operations.
        This table shows how often each operation appears in architectures that survive each
        evaluation stage. S0 = compiles and runs. S0.5 = numerically stable. S1 = actually
        learns (loss decreases). Higher S1% means this operation contributes to learnable
        architectures. The system uses this to evolve better combinations over time.
      </p>
      <div style={{ maxHeight: 400, overflow: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              {OP_COLUMNS.map(col => (
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
            {sorted.map((row) => {
              const rating = row._rating;
              return (
                <tr key={row.op}>
                  <td style={{ fontWeight: 600, color: opScoreColor(row._score) }}>
                    {row._score}
                  </td>
                  <td title={rating.tip}>
                    <span style={{
                      display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
                      background: rating.color, marginRight: 6,
                    }} />
                    <span style={{ fontSize: 11, color: rating.color }}>{rating.label}</span>
                  </td>
                  <td style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--accent-blue)' }}>{row.op}</td>
                  <td>{row.n_used}</td>
                  <td style={{
                    color: row.s0_rate > 0.7 ? 'var(--accent-green)' : row.s0_rate > 0.4 ? 'var(--accent-yellow)' : 'var(--accent-red)'
                  }}>
                    {(row.s0_rate * 100).toFixed(0)}%
                  </td>
                  <td style={{
                    color: row.s05_rate > 0.5 ? 'var(--accent-green)' : row.s05_rate > 0.2 ? 'var(--accent-yellow)' : 'var(--accent-red)'
                  }}>
                    {(row.s05_rate * 100).toFixed(0)}%
                  </td>
                  <td style={{
                    fontWeight: row.s1_rate > 0.05 ? 600 : 'normal',
                    color: row.s1_rate > 0.15 ? 'var(--accent-green)' : row.s1_rate > 0.05 ? 'var(--accent-yellow)' : row.s1_rate > 0 ? 'var(--accent-orange, #f0883e)' : 'var(--text-muted)'
                  }}>
                    {(row.s1_rate * 100).toFixed(1)}%
                  </td>
                  <td style={{
                    color: (row.avg_novelty || 0) > 0.7 ? 'var(--accent-green)' : (row.avg_novelty || 0) > 0.4 ? 'var(--accent-yellow)' : 'var(--text-muted)'
                  }}>
                    {row.avg_novelty?.toFixed(3) || '--'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8, display: 'flex', gap: 16 }}>
        <span><span style={{ color: 'var(--accent-green)' }}>Green</span> = op contributes to learnable architectures (S1 {'>'} 5%)</span>
        <span><span style={{ color: 'var(--accent-yellow)' }}>Amber</span> = some contribution or compiles well</span>
        <span><span style={{ color: 'var(--accent-red)' }}>Red</span> = rarely useful — system will deprioritize</span>
      </div>
    </div>
  );
}

function LearningLog({ log }) {
  if (!log || log.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Learning Log</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>No learning events yet.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Learning Log</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Timeline of when the system adapted its strategy — e.g., adjusting grammar weights
        based on which operations led to successful architectures.
      </p>
      <div style={{ maxHeight: 300, overflow: 'auto' }}>
        {log.map((entry, i) => (
          <div key={entry.id || i} style={{
            padding: '8px 12px',
            borderLeft: '3px solid var(--accent-purple)',
            marginBottom: 8,
            background: 'var(--bg-tertiary)',
            borderRadius: '0 4px 4px 0',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-purple)', textTransform: 'uppercase' }}>
                {entry.event_type?.replace(/_/g, ' ')}
              </span>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                {entry.timestamp ? new Date(entry.timestamp * 1000).toLocaleString() : ''}
              </span>
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              {entry.description}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EfficiencyFrontier({ frontier }) {
  if (!frontier || frontier.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Efficiency Frontier</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          Need Stage 1 survivors with FLOP data to compute frontier.
        </p>
      </div>
    );
  }

  // Simple scatter plot using SVG
  const W = 400, H = 200;
  const pad = 40;

  const losses = frontier.map(p => p.final_loss);
  const flops = frontier.map(p => Math.log10(Math.max(p.flops_forward, 1)));
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const minFlops = Math.min(...flops);
  const maxFlops = Math.max(...flops);
  const rangeL = maxLoss - minLoss || 1;
  const rangeF = maxFlops - minFlops || 1;

  const points = frontier.map((p, i) => ({
    x: pad + ((flops[i] - minFlops) / rangeF) * (W - 2 * pad),
    y: H - pad - ((losses[i] - minLoss) / rangeL) * (H - 2 * pad),
    label: p.graph_fingerprint?.slice(0, 8),
    novelty: p.novelty_score || 0,
  }));

  return (
    <div className="card">
      <div className="card-title">Efficiency Frontier ({frontier.length} Pareto-optimal)</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
        Architectures that are the best trade-off between compute cost (FLOPs) and learning
        quality (loss). Points on the frontier can't be beaten on both axes simultaneously —
        these are the most promising candidates for scaling up.
      </p>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        {/* Axes */}
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="var(--border)" />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="var(--border)" />
        <text x={W / 2} y={H - 5} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>log10(FLOPs)</text>
        <text x={10} y={H / 2} textAnchor="middle" fill="var(--text-muted)" fontSize={10}
          transform={`rotate(-90, 10, ${H / 2})`}>Loss</text>

        {/* Frontier line */}
        {points.length > 1 && (
          <polyline
            points={points.sort((a, b) => a.x - b.x).map(p => `${p.x},${p.y}`).join(' ')}
            fill="none" stroke="var(--accent-purple)" strokeWidth={1.5} strokeDasharray="4 2"
          />
        )}

        {/* Points */}
        {points.map((p, i) => (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r={5}
              fill={`rgba(188, 140, 255, ${0.3 + p.novelty * 0.7})`}
              stroke="var(--accent-purple)" strokeWidth={1.5} />
            <title>{p.label}: loss={losses[i]?.toFixed(4)}, FLOPs={frontier[i].flops_forward}</title>
          </g>
        ))}
      </svg>
    </div>
  );
}

function ExperimentClusters({ clustersData }) {
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
      <div className="card-title">Experiment Clusters ({clustersData.n_clusters})</div>
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
              <th>Cluster</th>
              <th>Size</th>
              <th>Avg S1%</th>
              <th>Avg Novelty</th>
              <th>Avg Loss Ratio</th>
            </tr>
          </thead>
          <tbody>
            {clustersData.clusters.map(c => (
              <tr key={c.cluster_id}>
                <td style={{ color: 'var(--accent-blue)' }}>#{c.cluster_id}</td>
                <td>{c.size}</td>
                <td>{((c.avg_s1_rate || 0) * 100).toFixed(1)}%</td>
                <td>{(c.avg_best_novelty || 0).toFixed(3)}</td>
                <td>{(c.avg_best_loss_ratio || 0).toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function LearningPanel() {
  const [weights, setWeights] = useState(null);
  const [opRates, setOpRates] = useState(null);
  const [log, setLog] = useState(null);
  const [frontier, setFrontier] = useState(null);
  const [clusters, setClusters] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/api/analytics/grammar-weights`).then(r => r.json()).catch(() => null),
      fetch(`${API_BASE}/api/analytics/op-success`).then(r => r.json()).catch(() => null),
      fetch(`${API_BASE}/api/analytics/learning-log`).then(r => r.json()).catch(() => null),
      fetch(`${API_BASE}/api/analytics/efficiency-frontier`).then(r => r.json()).catch(() => null),
      fetch(`${API_BASE}/api/analytics/experiment-clusters`).then(r => r.json()).catch(() => null),
    ]).then(([w, ops, lg, fr, cl]) => {
      setWeights(w);
      setOpRates(ops);
      setLog(lg);
      setFrontier(fr);
      setClusters(cl);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return <div className="card"><p style={{ color: 'var(--text-muted)' }}>Loading analytics...</p></div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <GrammarWeightsChart
          defaultWeights={weights?.default}
          learnedWeights={weights?.learned}
        />
        <EfficiencyFrontier frontier={frontier} />
      </div>
      <ExperimentClusters clustersData={clusters} />
      <OpSuccessTable opRates={opRates} />
      <LearningLog log={log} />
    </div>
  );
}

export default LearningPanel;
