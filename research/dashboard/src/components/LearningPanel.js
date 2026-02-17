import React, { useState, useEffect, useMemo } from 'react';
import { scoreColor } from '../utils/format';

const API_BASE = process.env.REACT_APP_API_URL || '';

/**
 * LearningPanel — Shows grammar weight evolution, op success rates,
 * learning log timeline, and efficiency frontier.
 */

function GrammarWeightsChart({ defaultWeights, learnedWeights, explanation }) {
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
      {explanation && (
        <div style={{ marginTop: 12, padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6, borderLeft: '3px solid var(--accent-purple)' }}>
          <div style={{ fontSize: 11, color: 'var(--accent-purple)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 4 }}>
            Aria's interpretation
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
            {explanation}
          </div>
        </div>
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

const OP_COLUMNS = [
  { key: '_score', label: 'Score' },
  { key: '_reliabilityOrder', label: 'Reliability' },
  { key: 'rating', label: 'Rating' },
  { key: 'op', label: 'Op' },
  { key: 'n_used', label: 'Used' },
  { key: 's0_rate', label: 'S0 %' },
  { key: 's05_rate', label: 'S0.5 %' },
  { key: 's1_rate', label: 'S1 %' },
  { key: 'avg_novelty', label: 'Avg Novelty' },
  { key: '_metricQualityOrder', label: 'Metric Quality' },
];

const RATING_ORDER = { Strong: 4, Good: 3, Some: 2, Compiles: 1, Weak: 0 };

function opScoreBreakdown(stats) {
  const s1 = Math.min((stats.s1_rate || 0) / 0.15, 1.0) * 40;
  const s05 = Math.min((stats.s05_rate || 0), 1.0) * 20;
  const s0 = Math.min((stats.s0_rate || 0), 1.0) * 10;
  const novelty = Math.min((stats.avg_novelty || 0), 1.0) * 20;
  const usage = Math.min((stats.n_used || 0) / 100, 1.0) * 10;
  return { s1, s05, s0, novelty, usage };
}

function opReliability(stats) {
  const n = stats.n_used || 0;
  if (n >= 100) return { label: 'High', color: 'var(--accent-green)', order: 3, tip: 'High confidence: large sample size' };
  if (n >= 40) return { label: 'Medium', color: 'var(--accent-yellow)', order: 2, tip: 'Moderate confidence: useful but still noisy' };
  if (n >= 15) return { label: 'Low', color: 'var(--accent-orange, #f0883e)', order: 1, tip: 'Low confidence: small sample size' };
  return { label: 'Very Low', color: 'var(--accent-red)', order: 0, tip: 'Very low confidence: treat as exploratory only' };
}

function reliabilityColor(level) {
  if (level === 'high') return 'var(--accent-green)';
  if (level === 'medium') return 'var(--accent-yellow)';
  return 'var(--accent-red)';
}

function opMetricChips(row) {
  const confidence = row.avg_novelty_confidence;
  return [
    {
      label: 'S1',
      source: 'measured',
      reliability: (row.n_used || 0) >= 100 ? 'high' : (row.n_used || 0) >= 40 ? 'medium' : 'low',
    },
    {
      label: 'Novelty',
      source: confidence != null && confidence >= 0.5 ? 'artifact-backed' : 'heuristic',
      reliability: confidence != null
        ? (confidence >= 0.7 ? 'high' : confidence >= 0.4 ? 'medium' : 'low')
        : 'low',
    },
  ];
}

function routingMetricChips(row) {
  const conf = row.avg_confidence_mean;
  return [
    {
      label: 'Routing',
      source: 'telemetry',
      reliability: conf != null
        ? (conf >= 0.7 ? 'high' : conf >= 0.4 ? 'medium' : 'low')
        : 'low',
    },
    {
      label: 'Sample',
      source: 'mode-aggregate',
      reliability: (row.n_programs || 0) >= 80 ? 'high' : (row.n_programs || 0) >= 30 ? 'medium' : 'low',
    },
  ];
}

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
      _reliability: opReliability(stats),
      _reliabilityOrder: opReliability(stats).order,
      _metricQualityOrder: (stats.n_used || 0),
    }));
  }, [opRates]);

  const sorted = useMemo(() => {
    const arr = [...augmented];
    arr.sort((a, b) => {
      let va, vb;
      if (sortKey === '_score') { va = a._score; vb = b._score; }
      else if (sortKey === '_reliabilityOrder') { va = a._reliabilityOrder || 0; vb = b._reliabilityOrder || 0; }
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
            {sorted.map((row) => {
              const rating = row._rating;
              const reliability = row._reliability;
              const nUsed = row.n_used || 0;
              const s0Count = Math.round((row.s0_rate || 0) * nUsed);
              const s05Count = Math.round((row.s05_rate || 0) * nUsed);
              const s1Count = Math.round((row.s1_rate || 0) * nUsed);
              const chips = opMetricChips(row);
              return (
                <tr key={row.op}>
                  <td style={{ fontWeight: 600, color: scoreColor(row._score) }}>
                    <span title={`S1 ${(opScoreBreakdown(row).s1 || 0).toFixed(1)}/40 | S0.5 ${(opScoreBreakdown(row).s05 || 0).toFixed(1)}/20 | S0 ${(opScoreBreakdown(row).s0 || 0).toFixed(1)}/10 | Novelty ${(opScoreBreakdown(row).novelty || 0).toFixed(1)}/20 | Usage ${(opScoreBreakdown(row).usage || 0).toFixed(1)}/10`}>
                      {row._score}
                    </span>
                  </td>
                  <td title={reliability.tip}>
                    <span style={{ color: reliability.color, fontSize: 11, fontWeight: 600 }}>
                      {reliability.label}
                    </span>
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
                    {(row.s0_rate * 100).toFixed(0)}% ({s0Count}/{nUsed})
                  </td>
                  <td style={{
                    color: row.s05_rate > 0.5 ? 'var(--accent-green)' : row.s05_rate > 0.2 ? 'var(--accent-yellow)' : 'var(--accent-red)'
                  }}>
                    {(row.s05_rate * 100).toFixed(0)}% ({s05Count}/{nUsed})
                  </td>
                  <td style={{
                    fontWeight: row.s1_rate > 0.05 ? 600 : 'normal',
                    color: row.s1_rate > 0.15 ? 'var(--accent-green)' : row.s1_rate > 0.05 ? 'var(--accent-yellow)' : row.s1_rate > 0 ? 'var(--accent-orange, #f0883e)' : 'var(--text-muted)'
                  }}>
                    {(row.s1_rate * 100).toFixed(1)}% ({s1Count}/{nUsed})
                  </td>
                  <td style={{
                    color: (row.avg_novelty || 0) > 0.7 ? 'var(--accent-green)' : (row.avg_novelty || 0) > 0.4 ? 'var(--accent-yellow)' : 'var(--text-muted)'
                  }}>
                    {row.avg_novelty != null ? row.avg_novelty.toFixed(3) : 'not computed'}
                  </td>
                  <td>
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', maxWidth: 220 }}>
                      {chips.map(chip => (
                        <span
                          key={`${row.op}-${chip.label}`}
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
  const [hover, setHover] = useState(null);

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
    data: p,
    idx: i,
  }));

  return (
    <div className="card" style={{ position: 'relative' }}>
      <div className="card-title">Efficiency Frontier ({frontier.length} Pareto-optimal)</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
        Architectures that are the best trade-off between compute cost (FLOPs) and learning
        quality (loss). Points on the frontier can't be beaten on both axes simultaneously —
        these are the most promising candidates for scaling up.
      </p>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}
        onMouseLeave={() => setHover(null)}>
        {/* Axes */}
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="var(--border)" />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="var(--border)" />
        <text x={W / 2} y={H - 5} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>log10(FLOPs)</text>
        <text x={10} y={H / 2} textAnchor="middle" fill="var(--text-muted)" fontSize={10}
          transform={`rotate(-90, 10, ${H / 2})`}>Loss</text>

        {/* Frontier line */}
        {points.length > 1 && (
          <polyline
            points={[...points].sort((a, b) => a.x - b.x).map(p => `${p.x},${p.y}`).join(' ')}
            fill="none" stroke="var(--accent-purple)" strokeWidth={1.5} strokeDasharray="4 2"
          />
        )}

        {/* Points */}
        {points.map((p, i) => (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r={hover?.idx === i ? 7 : 5}
              fill={`rgba(188, 140, 255, ${0.3 + p.novelty * 0.7})`}
              stroke={hover?.idx === i ? 'var(--accent-blue)' : 'var(--accent-purple)'}
              strokeWidth={hover?.idx === i ? 2.5 : 1.5}
              style={{ cursor: 'pointer' }}
              onMouseEnter={() => setHover(p)}
              onMouseLeave={() => setHover(null)} />
          </g>
        ))}
      </svg>

      {/* Hover card */}
      {hover && (
        <div style={{
          position: 'absolute',
          top: 60,
          right: 12,
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          padding: '10px 14px',
          fontSize: 12,
          lineHeight: 1.6,
          zIndex: 10,
          minWidth: 200,
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
        }}>
          <div style={{ fontWeight: 600, color: 'var(--accent-purple)', marginBottom: 4 }}>
            {hover.label || 'Unknown'}
          </div>
          <div><span style={{ color: 'var(--text-muted)' }}>Loss:</span> {hover.data.final_loss?.toFixed(4)}</div>
          <div><span style={{ color: 'var(--text-muted)' }}>FLOPs:</span> {hover.data.flops_forward?.toLocaleString()}</div>
          <div><span style={{ color: 'var(--text-muted)' }}>Params:</span> {hover.data.param_count?.toLocaleString()}</div>
          <div><span style={{ color: 'var(--text-muted)' }}>Novelty:</span> {(hover.data.novelty_score || 0).toFixed(3)}</div>
          {hover.data.ops && hover.data.ops.length > 0 && (
            <div style={{ marginTop: 4 }}>
              <span style={{ color: 'var(--text-muted)' }}>Ops:</span>{' '}
              <span style={{ fontFamily: 'monospace', color: 'var(--accent-blue)', fontSize: 11 }}>
                {hover.data.ops.join(', ')}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function LearningTrajectory({ trajectory }) {
  if (!trajectory || trajectory.trend === 'insufficient_data') {
    return (
      <div className="card">
        <div className="card-title">Learning Trajectory</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          Need at least 3 experiments to compute a learning trajectory.
        </p>
      </div>
    );
  }

  const trendColor = trajectory.trend === 'improving'
    ? 'var(--accent-green)'
    : trajectory.trend === 'declining'
      ? 'var(--accent-red)'
      : 'var(--accent-yellow)';

  const trendLabel = trajectory.trend === 'improving'
    ? 'Improving'
    : trajectory.trend === 'declining'
      ? 'Declining'
      : 'Plateaued';

  const points = trajectory.points || [];
  const W = 300, H = 80, pad = 4;

  let sparkline = null;
  if (points.length >= 2) {
    const rates = points.map(p => p.s1_rate);
    const maxR = Math.max(...rates, 0.01);
    const step = (W - 2 * pad) / (rates.length - 1);
    const pts = rates.map((r, i) => {
      const x = pad + i * step;
      const y = H - pad - (r / maxR) * (H - 2 * pad);
      return `${x},${y}`;
    });
    sparkline = (
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        <polyline points={pts.join(' ')} fill="none" stroke={trendColor} strokeWidth={2} />
        {pts.map((pt, i) => {
          const [x, y] = pt.split(',');
          return <circle key={i} cx={x} cy={y} r={2.5} fill={trendColor} />;
        })}
      </svg>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Learning Trajectory</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Tracks the stage-1 survival rate across recent experiments to show whether the
        AI scientist's search strategy is getting better at finding architectures that learn.
      </p>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
        <span style={{
          fontSize: 14, fontWeight: 700, color: trendColor,
          padding: '2px 10px', borderRadius: 12,
          background: trajectory.trend === 'improving'
            ? 'rgba(63,185,80,0.15)'
            : trajectory.trend === 'declining'
              ? 'rgba(248,81,73,0.15)'
              : 'rgba(210,153,34,0.15)',
        }}>
          {trendLabel}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          Recent S1 rate: {((trajectory.recent_s1_rate || 0) * 100).toFixed(1)}%
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Slope: {(trajectory.slope || 0) > 0 ? '+' : ''}{((trajectory.slope || 0) * 100).toFixed(2)}%/exp
        </span>
      </div>
      {sparkline}
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
        <span>{points.length} experiments</span>
        <span>Overall S1: {((trajectory.overall_s1_rate || 0) * 100).toFixed(1)}%</span>
        {trajectory.weight_adjustments != null && (
          <span>{trajectory.weight_adjustments} weight adjustments</span>
        )}
      </div>
    </div>
  );
}

function ExperimentClusters({ clustersData }) {
  const [sortKey, setSortKey] = useState('avg_s1_rate');
  const [sortDesc, setSortDesc] = useState(true);

  const handleSort = (key) => {
    if (sortKey === key) { setSortDesc(!sortDesc); } else { setSortKey(key); setSortDesc(true); }
  };

  const sorted = useMemo(() => {
    if (!clustersData?.clusters) return [];
    const arr = [...clustersData.clusters];
    arr.sort((a, b) => {
      let va = a[sortKey], vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [clustersData?.clusters, sortKey, sortDesc]);

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
              {clusterCols.map(col => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}
                  aria-label={`Sort by ${col.label}`}
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

function RoutingHealth({ data }) {
  const [sortKey, setSortKey] = useState('n_programs');
  const [sortDesc, setSortDesc] = useState(true);

  const handleSort = (key) => {
    if (sortKey === key) { setSortDesc(!sortDesc); } else { setSortKey(key); setSortDesc(true); }
  };

  const sorted = useMemo(() => {
    if (!data?.by_mode) return [];
    const arr = [...data.by_mode];
    arr.sort((a, b) => {
      let va = a[sortKey], vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [data?.by_mode, sortKey, sortDesc]);

  if (!data || data.available === false || !data.by_mode || data.by_mode.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Routing Health</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          No routing telemetry available yet. Routing health tracks how well mixture-of-experts
          architectures distribute work across their expert paths. It will appear once the system
          generates and evaluates routed architectures.
        </p>
      </div>
    );
  }

  const routingCols = [
    { key: 'routing_mode', label: 'Mode' },
    { key: 'n_programs', label: 'N' },
    { key: 'sample_size_label', label: 'Sample' },
    { key: 'stage1_pass_rate', label: 'S1%' },
    { key: 'avg_drop_rate', label: 'Drop%' },
    { key: 'avg_utilization_entropy', label: 'Entropy' },
    { key: 'avg_confidence_mean', label: 'Conf' },
    { key: 'confidence_label', label: 'Conf Label' },
    { key: 'stability_label', label: 'Stability' },
    { key: '_quality', label: 'Metric Quality' },
  ];

  return (
    <div className="card">
      <div className="card-title">Routing Health ({data.n_modes} modes)</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Aggregated routing telemetry by mode. Lower drop rate and higher confidence generally indicate healthier routing.
      </p>
      {data.explanation && (
        <div style={{ marginBottom: 10, padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6, borderLeft: '3px solid var(--accent-purple)' }}>
          <div style={{ fontSize: 11, color: 'var(--accent-purple)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 4 }}>
            Plain-language interpretation
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            {data.explanation}
          </div>
        </div>
      )}
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>
        <strong style={{ color: 'var(--accent-purple)' }}>Overall S1 pass:</strong>{' '}
        {((data.overall_stage1_pass_rate || 0) * 100).toFixed(1)}%
        <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>
          ({data.total_programs} programs)
        </span>
      </div>
      <div style={{ maxHeight: 260, overflow: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              {routingCols.map(col => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}
                  aria-label={`Sort by ${col.label}`}
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
              const chips = routingMetricChips(row);
              return (
              <tr key={row.routing_mode}>
                <td style={{ color: 'var(--accent-blue)' }}>{row.routing_mode}</td>
                <td>{row.n_programs ?? 0}</td>
                <td style={{ textTransform: 'uppercase', fontSize: 11 }}>{row.sample_size_label || 'unknown'}</td>
                <td>{((row.stage1_pass_rate || 0) * 100).toFixed(1)}%</td>
                <td>{((row.avg_drop_rate || 0) * 100).toFixed(1)}%</td>
                <td>{row.avg_utilization_entropy != null ? Number(row.avg_utilization_entropy).toFixed(3) : 'not measured'}</td>
                <td>{row.avg_confidence_mean != null ? Number(row.avg_confidence_mean).toFixed(3) : 'not measured'}</td>
                <td style={{ textTransform: 'uppercase', fontSize: 11 }}>{row.confidence_label || 'unknown'}</td>
                <td style={{ textTransform: 'uppercase', fontSize: 11 }}>{row.stability_label || 'unknown'}</td>
                <td>
                  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', maxWidth: 220 }}>
                    {chips.map(chip => (
                      <span
                        key={`${row.routing_mode}-${chip.label}`}
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
              </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function GatingBehaviorDiagnostics({ data }) {
  if (!data || data.available === false) {
    return (
      <div className="card">
        <div className="card-title">Gating Behavior Diagnostics</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          No gating diagnostics available yet. This section appears once routed or recursive candidates are evaluated.
        </p>
      </div>
    );
  }

  const rows = Array.isArray(data.by_mode) ? data.by_mode : [];
  return (
    <div className="card">
      <div className="card-title">Gating Behavior Diagnostics</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Canonical diagnostics for gate entropy, route-collapse risk, and token-retention curves across routing modes.
      </p>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8 }}>
        <strong style={{ color: 'var(--accent-purple)' }}>Routed candidates:</strong> {data.total_routed_programs || 0}
        <span style={{ marginLeft: 10 }}>
          <strong style={{ color: 'var(--accent-purple)' }}>Avg entropy:</strong>{' '}
          {data.avg_gate_entropy != null ? Number(data.avg_gate_entropy).toFixed(3) : 'not measured'}
        </span>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
        Collapse risk modes — high: {data?.collapse_risk_counts?.high || 0}, medium: {data?.collapse_risk_counts?.medium || 0}, low: {data?.collapse_risk_counts?.low || 0}
      </div>
      {data.explanation && (
        <div style={{ marginBottom: 10, padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6, borderLeft: '3px solid var(--accent-purple)', fontSize: 12, color: 'var(--text-secondary)' }}>
          {data.explanation}
        </div>
      )}
      {rows.length > 0 && (
        <div style={{ maxHeight: 260, overflow: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Mode</th>
                <th>N</th>
                <th>Entropy</th>
                <th>Collapse Risk</th>
                <th>Retention (avg)</th>
                <th>Retention Curve</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.routing_mode}>
                  <td style={{ color: 'var(--accent-blue)' }}>{row.routing_mode}</td>
                  <td>{row.n_programs ?? 0}</td>
                  <td>{row.avg_gate_entropy != null ? Number(row.avg_gate_entropy).toFixed(3) : 'not measured'}</td>
                  <td style={{ textTransform: 'uppercase', fontSize: 11 }}>{row.collapse_risk_label || 'unknown'}</td>
                  <td>{row.avg_token_retention != null ? `${(Number(row.avg_token_retention) * 100).toFixed(1)}%` : 'not measured'}</td>
                  <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {Array.isArray(row.token_retention_curve) && row.token_retention_curve.length > 0
                      ? row.token_retention_curve.map(point => `${point.quantile}:${(Number(point.retention) * 100).toFixed(0)}%`).join(' · ')
                      : 'not measured'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function MathFamilyCoverage({ data }) {
  const rows = Array.isArray(data?.families) ? data.families : [];
  const totals = data?.totals || {};

  if (rows.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Math Family Coverage</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          No program-family coverage data yet.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Math Family Coverage</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Share of evaluated and Stage-1 surviving programs by math family. Use this to verify the search is exploring beyond standard Euclidean patterns.
      </p>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>
        <strong style={{ color: 'var(--accent-purple)' }}>Totals:</strong>{' '}
        {totals.n_tested ?? 0} tested, {totals.n_survived ?? 0} Stage-1 survivors
      </div>
      <div style={{ maxHeight: 260, overflow: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Family</th>
              <th>Tested</th>
              <th>Survivors</th>
              <th>Survival %</th>
              <th>Test Share</th>
              <th>Survivor Share</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(row => (
              <tr key={row.family}>
                <td style={{ textTransform: 'capitalize', color: 'var(--accent-blue)' }}>{row.family}</td>
                <td>{row.n_tested ?? 0}</td>
                <td>{row.n_survived ?? 0}</td>
                <td>{((row.survival_rate || 0) * 100).toFixed(1)}%</td>
                <td>{((row.tested_share || 0) * 100).toFixed(1)}%</td>
                <td>{((row.survivor_share || 0) * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const COMPRESSION_FACTORS = {
  low_rank: 0.55, shared_basis: 0.5, hash_trick: 0.35,
  structured_sparse: 0.4, kronecker: 0.5, polynomial: 0.6,
  residual_quantized: 0.3,
};

const WEIGHT_STORAGE_LABELS = {
  dense_matrix: 'Dense (baseline)', low_rank: 'Low-Rank (UV)',
  hypernetwork: 'Hypernetwork', shared_basis: 'Shared Basis',
  hash_trick: 'Hash Trick', kronecker: 'Kronecker',
  polynomial: 'Polynomial', structured_sparse: 'Structured Sparse',
};

const TOKEN_REP_LABELS = {
  standard_float: 'Standard Float', binary_hash: 'Binary Hash',
  residual_quantized: 'Residual Quantized', complex_valued: 'Complex',
  quaternion: 'Quaternion', multi_resolution: 'Multi-Resolution',
  mixture_embedding: 'Mixture Embedding',
};

function parseArchSpec(value) {
  if (!value || typeof value !== 'string') return null;
  try {
    const p = JSON.parse(value);
    return p && typeof p === 'object' ? p : null;
  } catch { return null; }
}

function CompressionCoverage({ data, programs }) {
  const analysis = useMemo(() => {
    if (data && Array.isArray(data.techniques)) {
      const totals = data.totals || {};
      const sorted = [...data.techniques]
        .map((row) => ({
          technique: row.technique,
          label: WEIGHT_STORAGE_LABELS[row.technique] || TOKEN_REP_LABELS[row.technique] || row.technique,
          count: row.n_survived ?? 0,
          tested: row.n_tested ?? 0,
          avgLoss: row.avg_loss_ratio,
          bestLoss: row.best_loss_ratio,
          avgRatio: row.avg_compression_ratio,
          avgMemoryMb: row.avg_estimated_memory_mb,
          avgRetention: row.avg_quality_retention,
          survivalRate: row.survival_rate,
        }))
        .sort((a, b) => (b.count || 0) - (a.count || 0));

      return {
        sorted,
        denseCount: Math.max(0, (totals.n_survived || 0) - (totals.n_compressed_survived || 0)),
        compressedCount: totals.n_compressed_survived || 0,
        total: totals.n_survived || 0,
        testedTotal: totals.n_tested || 0,
        compressedTested: totals.n_compressed_tested || 0,
      };
    }

    if (!programs || programs.length === 0) return null;
    const byTechnique = {};
    let denseCount = 0;
    let compressedCount = 0;

    for (const p of programs) {
      const spec = parseArchSpec(p.arch_spec_json);
      const ws = spec?.choices?.weight_storage || 'dense_matrix';
      const tr = spec?.choices?.token_representation;
      const isDense = ws === 'dense_matrix' && (!tr || tr === 'standard_float');
      if (isDense) { denseCount++; } else { compressedCount++; }

      const key = ws !== 'dense_matrix' ? ws : (tr && tr !== 'standard_float' ? tr : 'dense_matrix');
      if (!byTechnique[key]) {
        byTechnique[key] = { count: 0, totalLoss: 0, lossCount: 0, bestLoss: Infinity };
      }
      const m = byTechnique[key];
      m.count++;
      if (p.loss_ratio != null) { m.totalLoss += p.loss_ratio; m.lossCount++; }
      if (p.loss_ratio != null && p.loss_ratio < m.bestLoss) m.bestLoss = p.loss_ratio;
    }

    const sorted = Object.entries(byTechnique)
      .map(([technique, m]) => ({
        technique,
        label: WEIGHT_STORAGE_LABELS[technique] || TOKEN_REP_LABELS[technique] || technique,
        count: m.count,
        avgLoss: m.lossCount > 0 ? m.totalLoss / m.lossCount : null,
        factor: COMPRESSION_FACTORS[technique] || 1.0,
        bestLoss: m.bestLoss < Infinity ? m.bestLoss : null,
      }))
      .sort((a, b) => b.count - a.count);

    return { sorted, denseCount, compressedCount, total: programs.length };
  }, [programs]);

  if (!analysis || analysis.compressedCount === 0) {
    return (
      <div className="card">
        <div className="card-title">Compression Technique Coverage</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          No compressed architectures among survivors yet. All current stage-1 survivors use dense
          weight matrices. Compression coverage will appear when the system generates and evaluates
          architectures with non-standard weight storage (low-rank, hash trick, sparse, etc.).
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Compression Technique Coverage</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Weight storage techniques across stage-1 survivors with explicit compression ratio,
        memory footprint, and quality-retention tradeoff summaries.
      </p>
      <div style={{ display: 'flex', gap: 12, marginBottom: 10, fontSize: 12, color: 'var(--text-secondary)' }}>
        <span><strong style={{ color: 'var(--accent-green)' }}>Compressed:</strong> {analysis.compressedCount}</span>
        <span><strong style={{ color: 'var(--text-muted)' }}>Dense:</strong> {analysis.denseCount}</span>
        <span style={{ color: 'var(--text-muted)' }}>({analysis.total} total)</span>
        {analysis.testedTotal != null && (
          <span style={{ color: 'var(--text-muted)' }}>
            tested {analysis.compressedTested}/{analysis.testedTotal} compressed
          </span>
        )}
      </div>
      <div style={{ maxHeight: 260, overflow: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Technique</th>
              <th>Tested</th>
              <th>N</th>
              <th>Survival %</th>
              <th>Avg Loss</th>
              <th>Best Loss</th>
              <th>Avg Ratio</th>
              <th>Avg Mem (MB)</th>
              <th>Quality Retention</th>
            </tr>
          </thead>
          <tbody>
            {analysis.sorted.map(row => (
              <tr key={row.technique}>
                <td style={{ color: (row.avgRatio != null && row.avgRatio < 1) ? 'var(--accent-green)' : 'var(--text-secondary)', fontWeight: 600 }}>
                  {row.label}
                </td>
                <td>{row.tested ?? '--'}</td>
                <td>{row.count}</td>
                <td>{row.survivalRate != null ? `${(row.survivalRate * 100).toFixed(1)}%` : '--'}</td>
                <td style={{ color: row.avgLoss != null && row.avgLoss < 0.6 ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
                  {row.avgLoss != null ? row.avgLoss.toFixed(4) : '--'}
                </td>
                <td>{row.bestLoss != null ? row.bestLoss.toFixed(4) : '--'}</td>
                <td style={{ color: row.avgRatio != null && row.avgRatio < 1 ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                  {row.avgRatio != null ? `${(row.avgRatio * 100).toFixed(0)}%` : '--'}
                </td>
                <td>{row.avgMemoryMb != null ? row.avgMemoryMb.toFixed(2) : '--'}</td>
                <td>{row.avgRetention != null ? `${(row.avgRetention * 100).toFixed(0)}%` : '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const GATING_OPS = new Set(['topk_gate']);

function GatingAggregate({ programs }) {
  const analysis = useMemo(() => {
    if (!programs || programs.length === 0) return null;
    let gatedCount = 0;
    let ungatedCount = 0;
    let totalDrop = 0;
    let dropCount = 0;
    let totalEntropy = 0;
    let entropyCount = 0;
    let totalConf = 0;
    let confCount = 0;
    let overflowTotal = 0;
    const byMode = {};

    for (const p of programs) {
      const hasRouting = p.routing_mode || p.routing_drop_rate != null;
      if (!hasRouting) { ungatedCount++; continue; }
      gatedCount++;

      if (p.routing_drop_rate != null) { totalDrop += p.routing_drop_rate; dropCount++; }
      if (p.routing_utilization_entropy != null) { totalEntropy += p.routing_utilization_entropy; entropyCount++; }
      if (p.routing_confidence_mean != null) { totalConf += p.routing_confidence_mean; confCount++; }
      if (p.routing_capacity_overflow_count != null) overflowTotal += p.routing_capacity_overflow_count;

      // Also detect topk_gate from graph
      let hasGateOp = false;
      if (p.graph_json) {
        try {
          const g = typeof p.graph_json === 'string' ? JSON.parse(p.graph_json) : p.graph_json;
          const nodes = g?.nodes || {};
          const nodeList = Array.isArray(nodes) ? nodes : Object.values(nodes);
          hasGateOp = nodeList.some(n => GATING_OPS.has(n?.op_name || n?.op));
        } catch { /* ignore */ }
      }

      const mode = p.routing_mode || (hasGateOp ? 'gated' : 'unknown');
      if (!byMode[mode]) byMode[mode] = { count: 0, totalLoss: 0, lossCount: 0 };
      byMode[mode].count++;
      if (p.loss_ratio != null) { byMode[mode].totalLoss += p.loss_ratio; byMode[mode].lossCount++; }
    }

    if (gatedCount === 0) return null;

    const avgDrop = dropCount > 0 ? totalDrop / dropCount : null;
    const avgEntropy = entropyCount > 0 ? totalEntropy / entropyCount : null;
    const avgConf = confCount > 0 ? totalConf / confCount : null;

    const modes = Object.entries(byMode)
      .map(([mode, m]) => ({
        mode,
        count: m.count,
        avgLoss: m.lossCount > 0 ? m.totalLoss / m.lossCount : null,
      }))
      .sort((a, b) => b.count - a.count);

    return { gatedCount, ungatedCount, avgDrop, avgEntropy, avgConf, overflowTotal, modes };
  }, [programs]);

  if (!analysis) {
    return (
      <div className="card">
        <div className="card-title">Gating Aggregate</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          No gated/routed architectures among survivors yet. Gating diagnostics will appear when the system
          generates and evaluates architectures with mixture-of-experts routing (topk_gate, etc.).
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Gating Aggregate ({analysis.gatedCount} routed programs)</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Aggregate gating health across all routed stage-1 survivors. Lower drop rate and higher
        entropy indicate healthier expert utilization system-wide.
      </p>

      <div style={{ display: 'flex', gap: 20, marginBottom: 12, flexWrap: 'wrap' }}>
        {analysis.avgDrop != null && (
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>Avg Drop Rate</div>
            <div style={{
              fontSize: 18, fontWeight: 700,
              color: analysis.avgDrop > 0.3 ? 'var(--accent-red)' : analysis.avgDrop > 0.1 ? 'var(--accent-yellow)' : 'var(--accent-green)',
            }}>
              {(analysis.avgDrop * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
              {analysis.avgDrop > 0.3 ? 'High token loss' : analysis.avgDrop > 0.1 ? 'Moderate' : 'Healthy'}
            </div>
          </div>
        )}
        {analysis.avgEntropy != null && (
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>Avg Entropy</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-secondary)' }}>
              {analysis.avgEntropy.toFixed(3)}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Utilization balance</div>
          </div>
        )}
        {analysis.avgConf != null && (
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>Avg Confidence</div>
            <div style={{
              fontSize: 18, fontWeight: 700,
              color: analysis.avgConf > 0.8 ? 'var(--accent-green)' : analysis.avgConf > 0.5 ? 'var(--accent-yellow)' : 'var(--accent-red)',
            }}>
              {analysis.avgConf.toFixed(3)}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
              {analysis.avgConf > 0.8 ? 'Decisive' : analysis.avgConf > 0.5 ? 'Moderate' : 'Uncertain'}
            </div>
          </div>
        )}
        {analysis.overflowTotal > 0 && (
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>Total Overflows</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-red)' }}>
              {analysis.overflowTotal.toLocaleString()}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Capacity overflow events</div>
          </div>
        )}
      </div>

      {analysis.modes.length > 0 && (
        <div style={{ maxHeight: 180, overflow: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Mode</th>
                <th>N</th>
                <th>Avg Loss</th>
              </tr>
            </thead>
            <tbody>
              {analysis.modes.map(row => (
                <tr key={row.mode}>
                  <td style={{ color: 'var(--accent-blue)' }}>{row.mode}</td>
                  <td>{row.count}</td>
                  <td style={{ color: row.avgLoss != null && row.avgLoss < 0.6 ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
                    {row.avgLoss != null ? row.avgLoss.toFixed(4) : '--'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function WhatIHaveLearned({ summary }) {
  if (!summary || !summary.bullets || summary.bullets.length === 0) {
    return null;
  }

  return (
    <div className="card">
      <div className="card-title">What I've learned</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Aria's synthesized takeaways across grammar adaptation, frontier quality, clusters, and recent experiment outcomes.
      </p>
      <ul style={{ margin: 0, paddingLeft: 18, color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: 6 }}>
        {summary.bullets.map((bullet, index) => (
          <li key={index} style={{ fontSize: 12, lineHeight: 1.5 }}>
            {bullet}
          </li>
        ))}
      </ul>
    </div>
  );
}

function LearningPanel() {
  const [weights, setWeights] = useState(null);
  const [opRates, setOpRates] = useState(null);
  const [log, setLog] = useState(null);
  const [frontier, setFrontier] = useState(null);
  const [clusters, setClusters] = useState(null);
  const [routingHealth, setRoutingHealth] = useState(null);
  const [routingComparison, setRoutingComparison] = useState(null);
  const [gatingDiagnostics, setGatingDiagnostics] = useState(null);
  const [mathFamilyCoverage, setMathFamilyCoverage] = useState(null);
  const [compressionCoverage, setCompressionCoverage] = useState(null);
  const [learningSummary, setLearningSummary] = useState(null);
  const [trajectory, setTrajectory] = useState(null);
  const [topPrograms, setTopPrograms] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    const safeFetch = (url) => fetch(url).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    }).catch(() => null);

    Promise.all([
      safeFetch(`${API_BASE}/api/analytics/grammar-weights`),
      safeFetch(`${API_BASE}/api/analytics/op-success`),
      safeFetch(`${API_BASE}/api/analytics/learning-log`),
      safeFetch(`${API_BASE}/api/analytics/efficiency-frontier`),
      safeFetch(`${API_BASE}/api/analytics/experiment-clusters`),
      safeFetch(`${API_BASE}/api/analytics/routing-health`),
      safeFetch(`${API_BASE}/api/analytics/routing-comparison`),
      safeFetch(`${API_BASE}/api/analytics/gating-diagnostics`),
      safeFetch(`${API_BASE}/api/analytics/math-family-coverage`),
      safeFetch(`${API_BASE}/api/analytics/compression-coverage`),
      safeFetch(`${API_BASE}/api/analytics/learning-summary`),
      safeFetch(`${API_BASE}/api/analytics/learning-trajectory`),
      safeFetch(`${API_BASE}/api/programs?n=100&sort_by=loss_ratio`),
    ]).then(([w, ops, lg, fr, cl, rh, rc, gd, mf, cc, ls, lt, tp]) => {
      if (!w && !ops && !lg && !fr && !cl && !rh && !rc && !gd && !mf && !cc && !ls && !lt) {
        setError('Failed to load analytics data. The API may be unavailable.');
      }
      setWeights(w);
      setOpRates(ops);
      setLog(lg);
      setFrontier(fr);
      setClusters(cl);
      setRoutingHealth(rh);
      setRoutingComparison(rc);
      setGatingDiagnostics(gd);
      setMathFamilyCoverage(mf);
      setCompressionCoverage(cc);
      setLearningSummary(ls);
      setTrajectory(lt);
      setTopPrograms(Array.isArray(tp) ? tp : null);
      setLastUpdated(new Date());
      setLoading(false);
    }).catch(e => {
      setError('Failed to load analytics: ' + e.message);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return <div className="card"><p style={{ color: 'var(--text-muted)' }}>Loading analytics...</p></div>;
  }

  if (error) {
    return <div className="card"><p style={{ color: 'var(--accent-red)' }}>{error}</p></div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="card" style={{ padding: '12px 16px' }}>
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
          The AI scientist searches for novel neural network layer designs by generating random
          compositions of operations, testing if they compile and learn, and evolving the search
          grammar toward successful patterns. This tab shows what the system has learned so far.
        </p>
        <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: '8px 0 0' }}>
          Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'loading'} · Sources: analytics endpoints
        </p>
      </div>
      <WhatIHaveLearned summary={learningSummary} />
      <LearningTrajectory trajectory={trajectory} />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <GrammarWeightsChart
          defaultWeights={weights?.default}
          learnedWeights={weights?.learned}
          explanation={weights?.explanation}
        />
        <EfficiencyFrontier frontier={frontier} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <ExperimentClusters clustersData={clusters} />
        <RoutingHealth data={routingComparison || routingHealth} />
      </div>
      <MathFamilyCoverage data={mathFamilyCoverage} />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <CompressionCoverage data={compressionCoverage} programs={topPrograms} />
        <GatingBehaviorDiagnostics data={gatingDiagnostics} />
      </div>
      <OpSuccessTable opRates={opRates} />
      <LearningLog log={log} />
    </div>
  );
}

export default LearningPanel;
