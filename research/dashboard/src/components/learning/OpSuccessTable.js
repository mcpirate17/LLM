import React, { useState, useMemo } from 'react';
import { scoreColor } from '../../utils/format';
import { reliabilityColor } from '../../utils/colors';
import { opScore, opScoreBreakdown } from '../../utils/scoringEngine';
import { filterRowsByQuery } from '../../utils/tableFiltering';
import Tooltip from '../shared/Tooltip';

/** Rate an op's contribution: green (strong), amber (some), red (weak) */
function opRating(stats) {
  const s1 = stats.s1_rate || 0;
  const s0 = stats.s0_rate || 0;
  if (s1 > 0.15) return { color: 'var(--accent-green)', label: 'Strong', tip: 'This op frequently appears in architectures that learn — a key building block' };
  if (s1 > 0.05) return { color: 'var(--accent-green)', label: 'Good', tip: 'This op contributes to some learnable architectures' };
  if (s1 > 0) return { color: 'var(--accent-yellow)', label: 'Some', tip: 'Rarely leads to learning but has produced at least one survivor' };
  if (s0 > 0.5) return { color: 'var(--accent-orange, #f0883e)', label: 'Compiles', tip: "Compiles reliably but hasn't produced a learnable architecture yet" };
  return { color: 'var(--accent-red)', label: 'Weak', tip: 'Rarely compiles or leads to learning — may be deprioritized' };
}

const OP_COLUMNS = [
  { key: '_score', label: 'Score', tooltip: 'Composite op score used to rank operations (S1, S0.5, S0, novelty, usage).' },
  { key: '_reliabilityOrder', label: 'Reliability', tooltip: 'Confidence based on sample size (how many architectures used this op).' },
  { key: 'rating', label: 'Rating', tooltip: 'Qualitative rating derived from S1/S0 pass rates.' },
  { key: 'op', label: 'Op', tooltip: 'Primitive operation identifier used in generated programs.' },
  { key: 'n_used', label: 'Used', tooltip: 'Number of architectures that included this op.' },
  { key: 's0_rate', label: 'S0 %', tooltip: 'Percent of architectures that compile and run.' },
  { key: 's05_rate', label: 'S0.5 %', tooltip: 'Percent of architectures that are numerically stable.' },
  { key: 's1_rate', label: 'S1 %', tooltip: 'Percent of architectures that learn (loss decreases).' },
  { key: 'avg_novelty', label: 'Avg Novelty', tooltip: 'Average novelty score for architectures using this op.' },
  { key: '_metricQualityOrder', label: 'Metric Quality', tooltip: 'Coverage of trustworthy metrics for this op (more = better).' },
];

const RATING_ORDER = { Strong: 4, Good: 3, Some: 2, Compiles: 1, Weak: 0 };

function opReliability(stats) {
  const n = stats.n_used || 0;
  if (n >= 100) return { label: 'High', color: 'var(--accent-green)', order: 3, tip: 'High confidence: large sample size' };
  if (n >= 40) return { label: 'Medium', color: 'var(--accent-yellow)', order: 2, tip: 'Moderate confidence: useful but still noisy' };
  if (n >= 15) return { label: 'Low', color: 'var(--accent-orange, #f0883e)', order: 1, tip: 'Low confidence: small sample size' };
  return { label: 'Very Low', color: 'var(--accent-red)', order: 0, tip: 'Very low confidence: treat as exploratory only' };
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

export function OpSuccessTable({ opRates }) {
  const [sortKey, setSortKey] = useState('_score');
  const [sortDesc, setSortDesc] = useState(true);
  const [filterQuery, setFilterQuery] = useState('');

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

  const filtered = useMemo(() => (
    filterRowsByQuery(augmented, filterQuery, ['op'])
  ), [augmented, filterQuery]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
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
  }, [filtered, sortKey, sortDesc]);

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
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <span>Op Success Rates</span>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter ops"
          style={{
            fontSize: 11,
            padding: '4px 8px',
            borderRadius: 4,
            border: '1px solid var(--border)',
            background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)',
            minWidth: 160,
          }}
        />
      </div>
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
                  {col.tooltip ? (
                    <Tooltip content={col.tooltip}>
                      <span>{col.label}</span>
                    </Tooltip>
                  ) : col.label}
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
                    <Tooltip content={`S1 ${(opScoreBreakdown(row).s1 || 0).toFixed(1)}/40 | S0.5 ${(opScoreBreakdown(row).s05 || 0).toFixed(1)}/20 | S0 ${(opScoreBreakdown(row).s0 || 0).toFixed(1)}/10 | Novelty ${(opScoreBreakdown(row).novelty || 0).toFixed(1)}/20 | Usage ${(opScoreBreakdown(row).usage || 0).toFixed(1)}/10`}>
                      <span>{row._score}</span>
                    </Tooltip>
                  </td>
                  <td>
                    <Tooltip content={`${reliability.tip}
Based on ${nUsed} architectures.`}>
                      <span style={{ color: reliability.color, fontSize: 11, fontWeight: 600 }}>
                        {reliability.label}
                      </span>
                    </Tooltip>
                  </td>
                  <td>
                    <Tooltip content={`${rating.tip}
Appeared in ${nUsed} architectures, ${s1Count} learned.`}>
                      <span style={{
                        display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
                        background: rating.color, marginRight: 6,
                      }} />
                      <span style={{ fontSize: 11, color: rating.color }}>{rating.label}</span>
                    </Tooltip>
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

export default OpSuccessTable;
