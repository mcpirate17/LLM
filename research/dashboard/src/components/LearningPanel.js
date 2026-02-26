import React, { useState, useEffect, useMemo } from 'react';
import { scoreColor } from '../utils/format';
import { reliabilityColor } from '../utils/colors';
import { useAriaData } from '../hooks/useAriaData';
import { NarrativeProvider, useNarrative } from '../hooks/useNarrative';
import { opScore, opScoreBreakdown } from '../utils/scoringEngine';
import { filterRowsByQuery } from '../utils/tableFiltering';
import { CHART_DEFAULTS, clampToScale, getFixedScale } from '../utils/chartScales';

const API_BASE = process.env.REACT_APP_API_URL || '';

function fmtNumber(value, digits = 0) {
  if (!Number.isFinite(value)) return '—';
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function fmtPct(value, digits = 0) {
  if (!Number.isFinite(value)) return '—';
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function Tooltip({ children, content }) {
  const [show, setShow] = useState(false);

  if (!content) return children;

  return (
    <div
      style={{ position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <div style={{
          position: 'absolute',
          bottom: '100%',
          left: '50%',
          transform: 'translateX(-50%)',
          marginBottom: 8,
          padding: '8px 12px',
          background: '#161b22',
          border: '1px solid var(--border)',
          borderRadius: 6,
          boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
          zIndex: 1000,
          minWidth: 200,
          whiteSpace: 'pre-wrap',
          fontSize: 11,
          fontWeight: 400,
          lineHeight: 1.4,
          color: 'var(--text-primary)',
          pointerEvents: 'none',
          textAlign: 'center',
        }}>
          {content}
          <div style={{
            position: 'absolute',
            top: '100%',
            left: '50%',
            marginLeft: -6,
            border: '6px solid transparent',
            borderTopColor: 'var(--border)'
          }} />
        </div>
      )}
    </div>
  );
}

function Section({ title, id, isOpen, onToggle, children }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div
        onClick={() => onToggle(id)}
        style={{
          padding: '10px 16px',
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border)',
          borderRadius: isOpen ? '6px 6px 0 0' : '6px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          cursor: 'pointer',
          userSelect: 'none'
        }}
      >
        <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{title}</h3>
        <span style={{ fontSize: 14, color: 'var(--text-muted)' }}>{isOpen ? '\u25be' : '\u25b8'}</span>
      </div>
      {isOpen && (
        <div style={{
          padding: '16px 0',
          display: 'flex',
          flexDirection: 'column',
          gap: 16
        }}>
          {children}
        </div>
      )}
    </div>
  );
}

/**
 * LearningPanel — Shows grammar weight evolution, op success rates,
 * learning log timeline, and efficiency frontier.
 */

function computeWeightedAverage(rows, key) {
  if (!Array.isArray(rows) || rows.length === 0) return null;
  let total = 0;
  let weightSum = 0;
  for (const row of rows) {
    const weight = Number(row?.n_programs || 0);
    const value = Number(row?.[key]);
    if (!Number.isFinite(value) || weight <= 0) continue;
    total += value * weight;
    weightSum += weight;
  }
  if (weightSum <= 0) return null;
  return total / weightSum;
}

function computeTargetSummary(programs, routingData) {
  const rows = Array.isArray(programs) ? programs : [];
  const takeAvg = (key) => {
    const vals = rows.map(r => Number(r?.[key])).filter(v => Number.isFinite(v));
    if (!vals.length) return null;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  };
  const takeMedian = (key) => {
    const vals = rows.map(r => Number(r?.[key])).filter(v => Number.isFinite(v)).sort((a, b) => a - b);
    if (!vals.length) return null;
    const mid = Math.floor(vals.length / 2);
    return vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
  };

  const routingRows = routingData?.by_mode || [];
  const routingRetention = computeWeightedAverage(
    routingRows.map(r => ({
      ...r,
      token_retention: r.token_retention != null ? r.token_retention
        : (r.avg_drop_rate != null ? (1 - r.avg_drop_rate) : null),
    })),
    "token_retention"
  );

  let bestMode = null;
  let bestScore = -Infinity;
  for (const row of routingRows) {
    const retention = row.token_retention != null
      ? row.token_retention
      : (row.avg_drop_rate != null ? (1 - row.avg_drop_rate) : null);
    const entropy = Number(row.avg_utilization_entropy);
    const conf = Number(row.avg_confidence_mean);
    const score = (Number.isFinite(retention) ? retention : 0)
      + (Number.isFinite(entropy) ? entropy : 0)
      + (Number.isFinite(conf) ? conf : 0);
    if (score > bestScore) {
      bestScore = score;
      bestMode = row.routing_mode;
    }
  }

  return {
    efficiency: {
      throughputMedian: takeMedian("throughput_tok_s"),
      paramsMedian: takeMedian("param_count"),
      flopsMedian: takeMedian("flops_forward"),
      sampleCount: rows.length,
    },
    routing: {
      retention: routingRetention,
      entropy: computeWeightedAverage(routingRows, "avg_utilization_entropy"),
      confidence: computeWeightedAverage(routingRows, "avg_confidence_mean"),
      overflow: computeWeightedAverage(routingRows, "avg_capacity_overflow_count"),
      bestMode,
      sampleCount: routingData?.total_programs || 0,
    },
    adaptive: {
      depthSavings: takeAvg("depth_savings_ratio"),
      effectiveDepth: takeAvg("effective_depth_ratio"),
      recursionSavings: takeAvg("recursion_savings_ratio"),
      recursionDepth: takeAvg("recursion_depth_ratio"),
      sampleCount: rows.filter(r =>
        r.depth_savings_ratio != null
        || r.effective_depth_ratio != null
        || r.recursion_savings_ratio != null
        || r.recursion_depth_ratio != null
      ).length,
    }
  };
}

function TargetBalanceCards({ summary }) {
  if (!summary) return null;
  const { efficiency, routing, adaptive } = summary;

  return (
    <div className="card">
      <div className="card-title">Balanced Targets (MoE · MoD · MoR · Mamba)</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        These KPIs track Aria’s balance across routing health (MoE), adaptive compute (MoD/MoR), and efficiency (Mamba-like throughput).
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
        <div style={{ padding: 12, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-secondary)' }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent-blue)', textTransform: 'uppercase', marginBottom: 6 }}>
            Efficiency
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Median throughput: <strong>{fmtNumber(efficiency.throughputMedian, 0)} tok/s</strong>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Median params: <strong>{efficiency.paramsMedian ? `${fmtNumber(efficiency.paramsMedian / 1e6, 2)}M` : '—'}</strong>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Median FLOPs: <strong>{efficiency.flopsMedian ? fmtNumber(efficiency.flopsMedian, 0) : '—'}</strong>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
            Samples: {efficiency.sampleCount}
          </div>
        </div>
        <div style={{ padding: 12, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-secondary)' }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent-green)', textTransform: 'uppercase', marginBottom: 6 }}>
            Routing (MoE)
          </div>
          {routing.sampleCount > 0 ? (<>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Token retention: <strong>{fmtPct(routing.retention, 1)}</strong>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Utilization entropy: <strong>{fmtNumber(routing.entropy, 3)}</strong>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Confidence: <strong>{fmtNumber(routing.confidence, 3)}</strong>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
              Best mode: {routing.bestMode || '—'} · Samples: {routing.sampleCount}
            </div>
          </>) : (
            <div style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic', marginTop: 4 }}>
              N/A — no routing architectures evaluated yet
            </div>
          )}
        </div>
        <div style={{ padding: 12, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-secondary)' }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: '#c77dff', textTransform: 'uppercase', marginBottom: 6 }}>
            Adaptive Compute
          </div>
          {adaptive.sampleCount > 0 ? (<>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Depth savings: <strong>{fmtPct(adaptive.depthSavings, 1)}</strong>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Effective depth: <strong>{fmtPct(adaptive.effectiveDepth, 1)}</strong>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Recursion savings: <strong>{fmtPct(adaptive.recursionSavings, 1)}</strong>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
              Samples with telemetry: {adaptive.sampleCount}
            </div>
          </>) : (
            <div style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic', marginTop: 4 }}>
              N/A — no adaptive compute architectures evaluated yet
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function GrammarWeightsChart({ defaultWeights, learnedWeights, explanation, onStartExperiment }) {
  if (!defaultWeights) return null;

  const categories = Object.keys(defaultWeights).sort();
  const weightValues = categories.map(c => Math.max(defaultWeights[c] || 0, (learnedWeights || {})[c] || 0));
  const weightDefaults = CHART_DEFAULTS.grammar_weight;
  const weightScale = getFixedScale('learning.grammar_weight', weightValues, {
    defaultMin: weightDefaults.min,
    defaultMax: weightDefaults.max,
  });
  const maxWeight = Math.max(weightScale.max, 1);

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
        <div style={{ marginTop: 10, padding: '10px 12px', borderRadius: 6, background: 'var(--bg-tertiary)', border: '1px solid var(--border)' }}>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '0 0 6px', lineHeight: 1.5 }}>
            No learned weights yet — only default weights are shown. The system needs at least 5 distinct
            operation categories with success data to compute learned weights. Run more diverse experiments
            to explore different op categories.
          </p>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            Categories discovered: {categories.length} · Need success data across {'\u2265'}5 categories
          </div>
          {onStartExperiment && categories.length < 5 && (
            <button
              className="refresh-btn"
              style={{ fontSize: 11, padding: '4px 10px', marginTop: 8 }}
              onClick={() => onStartExperiment({
                mode: 'continuous', n_cycles: 5,
                source: 'grammar_weights', auto_harden: true,
                preflight_override: true, enforce_preflight: true,
              })}
            >
              Run 5 Continuous
            </button>
          )}
        </div>
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
                    <Tooltip content={`${reliability.tip}\nBased on ${nUsed} architectures.`}>
                      <span style={{ color: reliability.color, fontSize: 11, fontWeight: 600 }}>
                        {reliability.label}
                      </span>
                    </Tooltip>
                  </td>
                  <td>
                    <Tooltip content={`${rating.tip}\nAppeared in ${nUsed} architectures, ${s1Count} learned.`}>
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

function AdaptationSummary({ log }) {
  const summary = useMemo(() => {
    if (!log || log.length === 0) return null;
    let improved = 0, neutral = 0, regressed = 0;
    for (const entry of log) {
      const desc = (entry.description || '').toLowerCase();
      if (desc.includes('improved') || desc.includes('better') || desc.includes('positive')) {
        improved++;
      } else if (desc.includes('regressed') || desc.includes('worse') || desc.includes('negative') || desc.includes('declined')) {
        regressed++;
      } else {
        neutral++;
      }
    }
    return { total: log.length, improved, neutral, regressed };
  }, [log]);

  if (!summary) return null;

  return (
    <div className="card">
      <div className="card-title">Adaptation Outcomes</div>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 13, color: 'var(--text-secondary)', marginBottom: 8 }}>
        <span><strong>{summary.total}</strong> grammar adaptations</span>
        <span style={{ color: 'var(--accent-green)' }}>{summary.improved} improved</span>
        <span style={{ color: 'var(--text-muted)' }}>{summary.neutral} neutral</span>
        <span style={{ color: 'var(--accent-red)' }}>{summary.regressed} regressed</span>
      </div>
      {summary.total > 0 && (
        <div style={{
          height: 8, borderRadius: 4, display: 'flex', overflow: 'hidden',
          background: 'var(--bg-tertiary)',
        }}>
          {summary.improved > 0 && (
            <div style={{ width: `${(summary.improved / summary.total) * 100}%`, background: 'var(--accent-green)', height: '100%' }} />
          )}
          {summary.neutral > 0 && (
            <div style={{ width: `${(summary.neutral / summary.total) * 100}%`, background: 'var(--text-muted)', opacity: 0.4, height: '100%' }} />
          )}
          {summary.regressed > 0 && (
            <div style={{ width: `${(summary.regressed / summary.total) * 100}%`, background: 'var(--accent-red)', height: '100%' }} />
          )}
        </div>
      )}
    </div>
  );
}

function LearningLog({ log }) {
  const [showRaw, setShowRaw] = useState(false);

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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="card-title" style={{ margin: 0 }}>Learning Log ({log.length})</div>
        <button
          className="refresh-btn"
          style={{ fontSize: 11, padding: '2px 8px' }}
          onClick={() => setShowRaw(!showRaw)}
        >
          {showRaw ? 'Hide raw entries' : 'Show raw entries'}
        </button>
      </div>
      {showRaw && (
        <div style={{ maxHeight: 300, overflow: 'auto', marginTop: 8 }}>
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
      )}
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
  const lossDefaults = CHART_DEFAULTS.loss_ratio;
  const flopsDefaults = CHART_DEFAULTS.efficiency_log_flops;
  const lossScale = getFixedScale('learning.loss_ratio', losses, {
    defaultMin: lossDefaults.min,
    defaultMax: lossDefaults.max,
  });
  const flopsScale = getFixedScale('learning.efficiency_log_flops', flops, {
    defaultMin: flopsDefaults.min,
    defaultMax: flopsDefaults.max,
  });
  const minLoss = lossScale.min;
  const maxLoss = lossScale.max;
  const minFlops = flopsScale.min;
  const maxFlops = flopsScale.max;
  const rangeL = maxLoss - minLoss || 1;
  const rangeF = maxFlops - minFlops || 1;

  const points = frontier.map((p, i) => ({
    x: pad + ((clampToScale(flops[i], flopsScale) - minFlops) / rangeF) * (W - 2 * pad),
    y: H - pad - ((clampToScale(losses[i], lossScale) - minLoss) / rangeL) * (H - 2 * pad),
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

function LearningTrajectory({ trajectory, onNavigateStrategy, onStartExperiment }) {
  const minimumExperiments = Math.max(2, Number(trajectory?.min_experiments_required) || 5);
  const windowSize = 30;

  if (!trajectory || trajectory.trend === 'insufficient_data') {
    const current = trajectory?.n_experiments || 0;
    const pct = minimumExperiments > 0 ? Math.min(100, Math.round((current / minimumExperiments) * 100)) : 0;
    return (
      <div className="card">
        <div className="card-title">Learning Trajectory</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
          Tracks the stage-1 survival rate across experiments. Need at least {minimumExperiments} experiments to compute a learning trajectory.
        </p>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8 }}>
          Progress: {current} of {minimumExperiments} experiments
        </div>
        <div style={{
          height: 6, borderRadius: 3,
          background: 'var(--bg-tertiary)',
          overflow: 'hidden',
          marginBottom: 8,
        }}>
          <div style={{
            height: '100%', borderRadius: 3,
            width: `${pct}%`,
            background: 'var(--accent-purple)',
            opacity: 0.6,
            transition: 'width 0.4s ease',
          }} />
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, textAlign: 'right' }}>
          {pct}% ({current}/{minimumExperiments})
        </div>
        {onStartExperiment && current < minimumExperiments && (
          <button
            className="refresh-btn"
            style={{ fontSize: 11, padding: '4px 10px' }}
            onClick={() => onStartExperiment({
              mode: 'continuous', n_cycles: Math.max(5, minimumExperiments - current),
              source: 'learning_trajectory', auto_harden: true,
              preflight_override: true, enforce_preflight: true,
            })}
          >
            Run {Math.max(5, minimumExperiments - current)} Experiments
          </button>
        )}
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

  const points = (trajectory.points || []).slice(-windowSize);
  const W = 600, H = 200, pad = 40, padRight = 12, padTop = 12;

  let sparkline = null;
  if (points.length >= 2) {
    const rates = points.map(p => p.s1_rate);
    const rateDefaults = CHART_DEFAULTS.s1_rate;
    const rateScale = getFixedScale('learning.s1_rate', rates, {
      defaultMin: rateDefaults.min,
      defaultMax: rateDefaults.max,
    });
    const maxR = Math.max(rateScale.max, 0.01);
    const denom = Math.max(1, windowSize - 1);
    const step = (W - pad - padRight) / denom;
    const pts = rates.map((r, i) => {
      const x = pad + i * step;
      const clamped = clampToScale(r, rateScale);
      const y = H - pad - (clamped / maxR) * (H - pad - padTop);
      return `${x},${y}`;
    });

    // Grid lines (4 horizontal)
    const gridLines = [];
    const nGrid = 4;
    for (let g = 0; g <= nGrid; g++) {
      const val = (maxR * g) / nGrid;
      const gy = H - pad - (val / maxR) * (H - pad - padTop);
      gridLines.push(
        <g key={`grid-${g}`}>
          <line x1={pad} y1={gy} x2={W - padRight} y2={gy}
            stroke="var(--border)" strokeWidth={0.5} strokeDasharray={g === 0 ? 'none' : '4 2'} />
          <text x={pad - 4} y={gy + 3} textAnchor="end"
            fill="var(--text-muted)" fontSize={9}>
            {(val * 100).toFixed(1)}%
          </text>
        </g>
      );
    }

    // X-axis labels (every ~5th experiment)
    const xLabels = [];
    const labelEvery = Math.max(1, Math.floor(windowSize / 8));
    for (let i = 0; i < points.length; i += labelEvery) {
      const x = pad + i * step;
      xLabels.push(
        <text key={`x-${i}`} x={x} y={H - pad + 14} textAnchor="middle"
          fill="var(--text-muted)" fontSize={9}>
          #{i + 1}
        </text>
      );
    }

    // Regression line
    const slope = trajectory.slope || 0;
    const meanY = trajectory.overall_s1_rate || 0;
    const midIdx = (points.length - 1) / 2;
    const regStart = Math.max(0, meanY - slope * midIdx);
    const regEnd = meanY + slope * (points.length - 1 - midIdx);
    const regY1 = H - pad - (Math.min(Math.max(regStart, 0), maxR) / maxR) * (H - pad - padTop);
    const regY2 = H - pad - (Math.min(Math.max(regEnd, 0), maxR) / maxR) * (H - pad - padTop);

    sparkline = (
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', maxWidth: 700 }}>
        {gridLines}
        {xLabels}
        <line x1={pad} y1={regY1} x2={pad + (points.length - 1) * step} y2={regY2}
          stroke={trendColor} strokeWidth={1.5} strokeDasharray="6 3" opacity={0.6} />
        <polyline points={pts.join(' ')} fill="none" stroke={trendColor} strokeWidth={2} />
        {pts.map((pt, i) => {
          const [x, y] = pt.split(',');
          return (
            <circle key={i} cx={x} cy={y} r={3} fill={trendColor}
              style={{ cursor: 'default' }}>
              <title>Exp #{i + 1}: {(rates[i] * 100).toFixed(1)}% S1 rate</title>
            </circle>
          );
        })}
        <text x={W / 2} y={H - 2} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>
          Experiment #
        </text>
        <text x={8} y={(H - pad) / 2 + padTop} textAnchor="middle"
          fill="var(--text-muted)" fontSize={10}
          transform={`rotate(-90, 8, ${(H - pad) / 2 + padTop})`}>
          S1 Rate
        </text>
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
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10, flexWrap: 'wrap' }}>
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
      {trajectory.trend === 'plateaued' && (
        <div style={{
          marginBottom: 10,
          padding: '10px 12px',
          borderRadius: 6,
          border: '1px solid var(--border)',
          borderLeft: '3px solid var(--accent-purple)',
          background: 'var(--bg-tertiary)',
          fontSize: 12,
          color: 'var(--text-secondary)',
          lineHeight: 1.5,
        }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent-purple)', marginBottom: 4, textTransform: 'uppercase' }}>
            Aria's Analysis
          </div>
          <div>
            Search productivity has <strong>plateaued</strong>. Aria will likely recommend <strong>Novelty Search</strong> to escape this local minimum.
          </div>
          <div style={{ marginTop: 6 }}>
            <button
              className="refresh-btn"
              style={{ fontSize: 11, padding: '2px 8px' }}
              onClick={onNavigateStrategy}
            >
              See Strategy Advisor &rarr;
            </button>
          </div>
        </div>
      )}
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
  const [filterQuery, setFilterQuery] = useState('');

  const handleSort = (key) => {
    if (sortKey === key) { setSortDesc(!sortDesc); } else { setSortKey(key); setSortDesc(true); }
  };

  const filtered = useMemo(() => (
    filterRowsByQuery(clustersData?.clusters || [], filterQuery, [
      'cluster_id',
      'description',
    ])
  ), [clustersData?.clusters, filterQuery]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      let va = a[sortKey], vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [filtered, sortKey, sortDesc]);

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
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <span>Experiment Clusters ({clustersData.n_clusters})</span>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter clusters"
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
  const [filterQuery, setFilterQuery] = useState('');

  const handleSort = (key) => {
    if (sortKey === key) { setSortDesc(!sortDesc); } else { setSortKey(key); setSortDesc(true); }
  };

  const filtered = useMemo(() => (
    filterRowsByQuery(data?.by_mode || [], filterQuery, ['routing_mode'])
  ), [data?.by_mode, filterQuery]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      let va = a[sortKey], vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [filtered, sortKey, sortDesc]);

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
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <span>Routing Health ({data.n_modes} modes)</span>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter modes"
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
  const [sortKey, setSortKey] = useState('n_programs');
  const [sortDesc, setSortDesc] = useState(true);
  const [filterQuery, setFilterQuery] = useState('');

  const filtered = useMemo(() => (
    filterRowsByQuery(rows, filterQuery, ['routing_mode'])
  ), [rows, filterQuery]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      const va = a?.[sortKey];
      const vb = b?.[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [filtered, sortKey, sortDesc]);

  const handleSort = (key) => {
    if (sortKey === key) { setSortDesc(!sortDesc); } else { setSortKey(key); setSortDesc(true); }
  };
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
        <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Filter:</div>
          <input
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            placeholder="Filter modes"
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
      )}
      {rows.length > 0 && (
        <div style={{ maxHeight: 260, overflow: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th onClick={() => handleSort('routing_mode')} style={{ cursor: 'pointer' }}>
                  Mode{sortKey === 'routing_mode' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
                </th>
                <th onClick={() => handleSort('n_programs')} style={{ cursor: 'pointer' }}>
                  N{sortKey === 'n_programs' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
                </th>
                <th onClick={() => handleSort('avg_gate_entropy')} style={{ cursor: 'pointer' }}>
                  Entropy{sortKey === 'avg_gate_entropy' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
                </th>
                <th onClick={() => handleSort('collapse_risk_label')} style={{ cursor: 'pointer' }}>
                  Collapse Risk{sortKey === 'collapse_risk_label' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
                </th>
                <th onClick={() => handleSort('avg_token_retention')} style={{ cursor: 'pointer' }}>
                  Retention (avg){sortKey === 'avg_token_retention' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
                </th>
                <th>Retention Curve</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((row) => (
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
  const [sortKey, setSortKey] = useState('n_tested');
  const [sortDesc, setSortDesc] = useState(true);
  const [filterQuery, setFilterQuery] = useState('');

  const filtered = useMemo(() => (
    filterRowsByQuery(rows, filterQuery, ['family'])
  ), [rows, filterQuery]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      const va = a?.[sortKey];
      const vb = b?.[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [filtered, sortKey, sortDesc]);

  const handleSort = (key) => {
    if (sortKey === key) { setSortDesc(!sortDesc); } else { setSortKey(key); setSortDesc(true); }
  };

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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Filter:</div>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter families"
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
      <div style={{ maxHeight: 260, overflow: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('family')} style={{ cursor: 'pointer' }}>
                Family{sortKey === 'family' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('n_tested')} style={{ cursor: 'pointer' }}>
                Tested{sortKey === 'n_tested' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('n_survived')} style={{ cursor: 'pointer' }}>
                Survivors{sortKey === 'n_survived' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('survival_rate')} style={{ cursor: 'pointer' }}>
                Survival %{sortKey === 'survival_rate' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('tested_share')} style={{ cursor: 'pointer' }}>
                Test Share{sortKey === 'tested_share' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('survivor_share')} style={{ cursor: 'pointer' }}>
                Survivor Share{sortKey === 'survivor_share' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(row => (
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

function MathspaceImpact({ data }) {
  const rows = Array.isArray(data?.by_operator) ? data.by_operator : [];
  const families = Array.isArray(data?.by_family) ? data.by_family : [];
  const topTrust = Array.isArray(data?.top_trustworthy_operators) ? data.top_trustworthy_operators : [];
  const totals = data?.totals || {};
  const [sortKey, setSortKey] = useState('n_tested');
  const [sortDesc, setSortDesc] = useState(true);
  const [filterQuery, setFilterQuery] = useState('');

  const filtered = useMemo(() => (
    filterRowsByQuery(rows, filterQuery, ['op_name'])
  ), [rows, filterQuery]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      const va = a?.[sortKey];
      const vb = b?.[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [filtered, sortKey, sortDesc]);

  const handleSort = (key) => {
    if (sortKey === key) { setSortDesc(!sortDesc); } else { setSortKey(key); setSortDesc(true); }
  };

  if (!data || data.available === false || rows.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Mathspace Operator Impact</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          No mathspace operator impact data yet. This appears once programs include hyperbolic/tropical/p-adic/clifford operators.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Mathspace Operator Impact</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Canonical impact slice for mathspace operators and families across Stage-1 pass, validation pass, and novelty signals.
      </p>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>
        <strong style={{ color: 'var(--accent-purple)' }}>Coverage:</strong>{' '}
        {totals.n_programs_with_mathspace ?? 0}/{totals.n_programs_with_graph ?? 0} programs with graph traces include mathspace ops
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Trust score = (50% S1 pass + 30% validation pass + 20% baseline wins) × sample reliability,
        where sample reliability scales with tested count up to 25 programs.
      </div>

      {topTrust.length > 0 && (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
          {topTrust.map((row) => (
            <span
              key={row.op_name}
              style={{
                fontSize: 11,
                padding: '4px 8px',
                borderRadius: 999,
                border: `1px solid ${row.trust_label === 'high' ? 'var(--accent-green)' : row.trust_label === 'medium' ? 'var(--accent-yellow)' : 'var(--text-muted)'}`,
                color: row.trust_label === 'high' ? 'var(--accent-green)' : row.trust_label === 'medium' ? 'var(--accent-yellow)' : 'var(--text-muted)',
                background: 'var(--bg-tertiary)',
              }}
            >
              {row.op_name} · trust {(Number(row.trust_score || 0) * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Filter:</div>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter operators"
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
      <div style={{ maxHeight: 220, overflow: 'auto', marginBottom: 10 }}>
        <table className="data-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('op_name')} style={{ cursor: 'pointer' }}>
                Operator{sortKey === 'op_name' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('n_tested')} style={{ cursor: 'pointer' }}>
                Tested{sortKey === 'n_tested' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('stage1_pass_rate')} style={{ cursor: 'pointer' }}>
                S1 %{sortKey === 'stage1_pass_rate' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('validation_pass_rate')} style={{ cursor: 'pointer' }}>
                Validation %{sortKey === 'validation_pass_rate' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('baseline_win_rate')} style={{ cursor: 'pointer' }}>
                Baseline Win %{sortKey === 'baseline_win_rate' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('trust_score')} style={{ cursor: 'pointer' }}>
                Trust %{sortKey === 'trust_score' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('avg_novelty_score')} style={{ cursor: 'pointer' }}>
                Avg Novelty{sortKey === 'avg_novelty_score' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.slice(0, 10).map((row) => (
              <tr key={row.op_name}>
                <td style={{ color: 'var(--accent-blue)' }}>{row.op_name}</td>
                <td>{row.n_tested ?? 0}</td>
                <td>{((row.stage1_pass_rate || 0) * 100).toFixed(1)}%</td>
                <td>{((row.validation_pass_rate || 0) * 100).toFixed(1)}%</td>
                <td>{((row.baseline_win_rate || 0) * 100).toFixed(1)}%</td>
                <td>{((row.trust_score || 0) * 100).toFixed(1)}%</td>
                <td>{row.avg_novelty_score != null ? Number(row.avg_novelty_score).toFixed(3) : '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {families.length > 0 && (
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', fontSize: 11, color: 'var(--text-muted)' }}>
          {families.map((row) => (
            <span key={row.family}>
              <strong style={{ color: 'var(--accent-purple)' }}>{row.family}:</strong> S1 {(row.stage1_pass_rate * 100).toFixed(0)}% · V {(row.validation_pass_rate * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      )}
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
  const [sortKey, setSortKey] = useState('count');
  const [sortDesc, setSortDesc] = useState(true);
  const [filterQuery, setFilterQuery] = useState('');
  const analysis = useMemo(() => {
    if (data && Array.isArray(data.techniques)) {
      const totals = data.totals || {};
      const rows = [...data.techniques]
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
        }));

      return {
        rows,
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

    const rows = Object.entries(byTechnique)
      .map(([technique, m]) => ({
        technique,
        label: WEIGHT_STORAGE_LABELS[technique] || TOKEN_REP_LABELS[technique] || technique,
        count: m.count,
        avgLoss: m.lossCount > 0 ? m.totalLoss / m.lossCount : null,
        factor: COMPRESSION_FACTORS[technique] || 1.0,
        bestLoss: m.bestLoss < Infinity ? m.bestLoss : null,
      }));

    return { rows, denseCount, compressedCount, total: programs.length };
  }, [data, programs]);

  const filtered = useMemo(() => (
    filterRowsByQuery(analysis?.rows || [], filterQuery, ['technique', 'label'])
  ), [analysis?.rows, filterQuery]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      const va = a?.[sortKey];
      const vb = b?.[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [filtered, sortKey, sortDesc]);

  const handleSort = (key) => {
    if (sortKey === key) { setSortDesc(!sortDesc); } else { setSortKey(key); setSortDesc(true); }
  };

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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Filter:</div>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter techniques"
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
      <div style={{ maxHeight: 260, overflow: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('label')} style={{ cursor: 'pointer' }}>
                Technique{sortKey === 'label' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('tested')} style={{ cursor: 'pointer' }}>
                Tested{sortKey === 'tested' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('count')} style={{ cursor: 'pointer' }}>
                N{sortKey === 'count' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('survivalRate')} style={{ cursor: 'pointer' }}>
                Survival %{sortKey === 'survivalRate' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('avgLoss')} style={{ cursor: 'pointer' }}>
                Avg Loss{sortKey === 'avgLoss' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('bestLoss')} style={{ cursor: 'pointer' }}>
                Best Loss{sortKey === 'bestLoss' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('avgRatio')} style={{ cursor: 'pointer' }}>
                Avg Ratio{sortKey === 'avgRatio' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('avgMemoryMb')} style={{ cursor: 'pointer' }}>
                Avg Mem (MB){sortKey === 'avgMemoryMb' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
              <th onClick={() => handleSort('avgRetention')} style={{ cursor: 'pointer' }}>
                Quality Retention{sortKey === 'avgRetention' && <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(row => (
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

function ControlComparison({ data, onStartExperiment }) {
  if (!data || data.status === 'insufficient_data') {
    const nControl = data?.control?.experiments || 0;
    const nLearned = data?.learned?.experiments || 0;
    return (
      <div className="card">
        <div className="card-title">Learning Effectiveness</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
          Compares experiments using learned grammar weights vs control experiments with default weights.
          This tells you whether Aria's learning is actually improving search quality.
        </p>
        <div style={{
          padding: '10px 12px', borderRadius: 6, marginBottom: 10,
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border)',
          fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6,
        }}>
          <div style={{ marginBottom: 4 }}>
            <strong>What's needed:</strong> {'\u2265'}2 control + {'\u2265'}2 learned experiments
          </div>
          <div style={{ marginBottom: 4 }}>
            <strong>Current:</strong> {nControl} control, {nLearned} learned
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            Control experiments run automatically every 5th continuous-mode experiment with default grammar weights.
            Run 5 continuous experiments to guarantee at least 1 control.
          </div>
        </div>
        {onStartExperiment && (
          <button
            className="refresh-btn"
            style={{ fontSize: 11, padding: '4px 10px' }}
            onClick={() => onStartExperiment({
              mode: 'continuous', n_cycles: 5,
              source: 'control_comparison', auto_harden: true,
              preflight_override: true, enforce_preflight: true,
            })}
          >
            Run 5 Continuous
          </button>
        )}
      </div>
    );
  }

  const { control, learned, s1_rate_difference, z_score, significant_at_p05, learned_is_better, interpretation, caveat, matched_pairs } = data;

  const verdictColor = significant_at_p05
    ? (learned_is_better ? 'var(--accent-green)' : 'var(--accent-red, #e74c3c)')
    : 'var(--accent-yellow)';

  return (
    <div className="card">
      <div className="card-title">Learning Effectiveness</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Statistical comparison of experiments using learned grammar weights vs control experiments
        using default weights. A positive difference means learning is helping find better architectures.
      </p>

      <div style={{
        padding: '8px 12px', borderRadius: 6, marginBottom: 12,
        background: significant_at_p05
          ? (learned_is_better ? 'rgba(63,185,80,0.12)' : 'rgba(248,81,73,0.12)')
          : 'rgba(210,153,34,0.12)',
        border: `1px solid ${verdictColor}`,
      }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: verdictColor, marginBottom: 4 }}>
          {interpretation}
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
          z-score: {z_score} {significant_at_p05 ? '(p < 0.05)' : '(not significant)'}
          {matched_pairs ? ` · ${matched_pairs} time-matched pairs` : ''}
        </div>
        {caveat && (
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4, fontStyle: 'italic' }}>
            {caveat}
          </div>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div style={{ padding: '8px 12px', borderRadius: 6, background: 'var(--bg-tertiary)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase' }}>
            Control (Default Weights)
          </div>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)' }}>
            {(control.s1_rate * 100).toFixed(2)}%
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {control.s1_passed}/{control.programs} passed | {control.experiments} experiments
          </div>
        </div>
        <div style={{ padding: '8px 12px', borderRadius: 6, background: 'var(--bg-tertiary)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase' }}>
            Learned Weights
          </div>
          <div style={{ fontSize: 18, fontWeight: 700, color: learned_is_better ? 'var(--accent-green)' : 'var(--text-primary)' }}>
            {(learned.s1_rate * 100).toFixed(2)}%
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {learned.s1_passed}/{learned.programs} passed | {learned.experiments} experiments
          </div>
        </div>
      </div>

      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>
        S1 rate difference: {s1_rate_difference > 0 ? '+' : ''}{(s1_rate_difference * 100).toFixed(2)} percentage points
      </div>
    </div>
  );
}

function ArchitectureRerunTelemetry({ telemetry }) {
  if (!telemetry) {
    return null;
  }

  const uniqueCount = Number(telemetry.unique_fingerprint_count || 0);
  const totalRows = Number(telemetry.total_result_rows || 0);
  const repeatRows = Number(telemetry.repeat_result_rows || 0);
  const rerunRatio = Number(telemetry.rerun_ratio || 0);
  const topConcentration = Number(telemetry.top_fingerprint_concentration || 0);
  const weightingMode = telemetry.weighting_mode || 'unknown';

  return (
    <div className="card">
      <div className="card-title">Unique Architectures vs Reruns</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Breadth telemetry for architecture search. High rerun ratios or high top-fingerprint concentration
        indicate learning signal is coming from repeated identities rather than broad exploration.
      </p>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 8 }}>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          <strong style={{ color: 'var(--accent-green)' }}>Unique fingerprints:</strong> {uniqueCount}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          <strong style={{ color: 'var(--text-muted)' }}>Rows:</strong> {totalRows}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          <strong style={{ color: rerunRatio >= 0.6 ? 'var(--accent-yellow)' : 'var(--text-muted)' }}>Rerun ratio:</strong>{' '}
          {(rerunRatio * 100).toFixed(1)}%
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          <strong style={{ color: topConcentration >= 0.35 ? 'var(--accent-yellow)' : 'var(--text-muted)' }}>Top fingerprint concentration:</strong>{' '}
          {(topConcentration * 100).toFixed(1)}%
        </span>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
        Repeat rows: {repeatRows} · Weighting mode: {weightingMode}
      </div>
    </div>
  );
}

function FingerprintDiagnosticsCard({ diagnostics }) {
  if (!diagnostics) {
    return null;
  }

  const total = Number(diagnostics.total || 0);
  const byReason = diagnostics.by_reason && typeof diagnostics.by_reason === 'object'
    ? diagnostics.by_reason
    : {};
  const topReasons = Object.entries(byReason)
    .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0))
    .slice(0, 3);

  return (
    <div className="card">
      <div className="card-title">Fingerprint Diagnostics</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Runtime telemetry for skipped sensitivity probes during fingerprint analysis.
      </p>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 6 }}>
        <strong style={{ color: total > 0 ? 'var(--accent-yellow)' : 'var(--accent-green)' }}>
          Sensitivity skips:
        </strong>{' '}
        {total}
      </div>
      {topReasons.length > 0 && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          Top reasons: {topReasons.map(([reason, count]) => `${reason} (${count})`).join(' · ')}
        </div>
      )}
    </div>
  );
}

const MIN_SAMPLES = 5;

function sampleCount(data, kind) {
  if (!data) return 0;
  if (kind === 'clusters') return data.n_experiments || 0;
  if (kind === 'routing') return data.total_programs || 0;
  if (kind === 'gating') return data.total_routed_programs || 0;
  return 0;
}

function DataAccumulation({ title, current, threshold, children }) {
  if (current >= threshold) return children;
  const pct = threshold > 0 ? Math.min(100, Math.round((current / threshold) * 100)) : 0;
  return (
    <div className="card" style={{ position: 'relative', overflow: 'hidden' }}>
      <div className="card-title">{title}</div>
      <div style={{ padding: '16px 0 8px' }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
          Accumulating data — {current} of {threshold} samples needed for statistically
          meaningful results.
        </div>
        <div style={{
          height: 6, borderRadius: 3,
          background: 'var(--bg-tertiary)',
          overflow: 'hidden',
        }}>
          <div style={{
            height: '100%', borderRadius: 3,
            width: `${pct}%`,
            background: 'var(--accent-purple)',
            opacity: 0.6,
            transition: 'width 0.4s ease',
          }} />
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6, textAlign: 'right' }}>
          {pct}% ({current}/{threshold})
        </div>
      </div>
      {/* Skeleton rows */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, opacity: 0.25 }}>
        {[1, 2, 3].map(i => (
          <div key={i} style={{
            height: 14, borderRadius: 4,
            background: 'var(--bg-tertiary)',
            width: `${85 - i * 12}%`,
          }} />
        ))}
      </div>
    </div>
  );
}

function AriaThoughtProcess() {
  const ctx = useNarrative();
  if (!ctx?.narrative) return null;

  const { narrative, trend } = ctx;

  const trendColor = trend === 'improving'
    ? 'var(--accent-green)'
    : trend === 'declining'
      ? 'var(--accent-red, #e74c3c)'
      : 'var(--accent-purple)';

  return (
    <div className="card" style={{
      padding: '14px 16px',
      borderLeft: `3px solid ${trendColor}`,
      background: 'var(--bg-secondary)',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8,
      }}>
        <span style={{
          fontSize: 11, fontWeight: 700, textTransform: 'uppercase',
          letterSpacing: 0.5, color: trendColor,
        }}>
          Aria's Thought Process
        </span>
        <span style={{
          fontSize: 9, fontWeight: 600,
          color: trendColor,
          background: `color-mix(in srgb, ${trendColor} 12%, transparent)`,
          border: `1px solid ${trendColor}`,
          borderRadius: 4,
          padding: '1px 5px',
        }}>
          Live
        </span>
      </div>
      <div style={{
        fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65,
      }}>
        {narrative}
      </div>
    </div>
  );
}

function FeedbackLoopSummary({ weights, trajectory, controlComparison, title }) {
  const summary = useMemo(() => {
    const parts = [];

    if (weights?.default && weights?.learned) {
      const deltas = Object.keys(weights.default).map(cat => ({
        cat,
        delta: (weights.learned[cat] || 0) - weights.default[cat]
      })).sort((a, b) => b.delta - a.delta);

      if (deltas.length > 0 && deltas[0].delta > 0.2) {
        parts.push(`Grammar is shifting toward **${deltas[0].cat.replace(/_/g, ' ')}** (+${deltas[0].delta.toFixed(1)}).`);
      }
    }

    if (trajectory?.trend) {
      const trend = trajectory.trend === 'improving' ? 'improving' : trajectory.trend === 'declining' ? 'declining' : 'plateaued';
      parts.push(`Search productivity (S1 pass rate) is currently **${trend}**.`);
    }

    if (controlComparison?.interpretation) {
      parts.push(`Aria's verdict: **${controlComparison.interpretation}**.`);
    }

    return parts;
  }, [weights, trajectory, controlComparison]);

  if (summary.length === 0) return null;

  return (
    <div className="card" style={{ background: 'var(--bg-secondary)', borderLeft: '3px solid var(--accent-purple)' }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent-purple)', marginBottom: 8, textTransform: 'uppercase' }}>
        {title || 'Feedback Loop Summary'}
      </div>
      <div style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.6 }}>
        {summary.map((p, i) => (
          <div key={i} style={{ marginBottom: 4 }}>
            {p.split('**').map((text, idx) => (
              idx % 2 === 1 ? <strong key={idx}>{text}</strong> : text
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function InsightSynergyMatrix({ data }) {
  const synergistic = Array.isArray(data?.synergistic_pairs) ? data.synergistic_pairs : [];
  const antagonistic = Array.isArray(data?.antagonistic_pairs) ? data.antagonistic_pairs : [];
  const available = Boolean(data?.available) && (synergistic.length > 0 || antagonistic.length > 0);
  const trim = (text) => {
    const t = String(text || '').trim();
    if (t.length <= 88) return t;
    return `${t.slice(0, 85)}...`;
  };

  if (!available) {
    return (
      <div className="card">
        <h3 style={{ margin: 0, marginBottom: 8 }}>Insight Synergy Matrix</h3>
        <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: 12 }}>
          Not enough resolved insight-bundle trials yet.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <h3 style={{ margin: 0, marginBottom: 8 }}>Insight Synergy Matrix</h3>
      <p style={{ margin: '0 0 10px', color: 'var(--text-secondary)', fontSize: 12 }}>
        Learns which insight combinations improve downstream outcomes and which conflict.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent-green)', marginBottom: 6 }}>
            Positive Pairs
          </div>
          {synergistic.slice(0, 5).map((row, idx) => (
            <div key={`syn-${idx}`} style={{ fontSize: 11, marginBottom: 6, color: 'var(--text-secondary)' }}>
              <div>{trim(row?.insight_a_content)} + {trim(row?.insight_b_content)}</div>
              <div style={{ color: 'var(--text-muted)' }}>
                reward {Number(row?.mean_reward || 0).toFixed(3)} · trials {row?.n_trials || 0} · {row?.confidence_label || 'low'}
              </div>
            </div>
          ))}
        </div>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent-red)', marginBottom: 6 }}>
            Conflicting Pairs
          </div>
          {antagonistic.slice(0, 5).map((row, idx) => (
            <div key={`ant-${idx}`} style={{ fontSize: 11, marginBottom: 6, color: 'var(--text-secondary)' }}>
              <div>{trim(row?.insight_a_content)} + {trim(row?.insight_b_content)}</div>
              <div style={{ color: 'var(--text-muted)' }}>
                reward {Number(row?.mean_reward || 0).toFixed(3)} · trials {row?.n_trials || 0} · {row?.confidence_label || 'low'}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DataPopulateBar({ learningTrajectory, controlComparison, onStartExperiment }) {
  if (!onStartExperiment) return null;

  const nExperiments = learningTrajectory?.n_experiments || 0;
  const hasEnoughData = nExperiments >= 10
    && controlComparison?.status !== 'insufficient_data';
  if (hasEnoughData) return null;

  const controlNeeded = controlComparison?.status === 'insufficient_data';
  const nControlExps = controlComparison?.control?.experiments || 0;
  const nLearnedExps = controlComparison?.learned?.experiments || 0;

  return (
    <div className="card" style={{
      padding: '14px 16px',
      borderLeft: '3px solid var(--accent-blue)',
      background: 'var(--bg-secondary)',
    }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>
        More experiments needed to populate learning analytics
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 10 }}>
        {nExperiments < 5
          ? `You have ${nExperiments} experiment${nExperiments === 1 ? '' : 's'}. At least 5 are needed for trajectory analysis, and control experiments run automatically every 5th continuous experiment.`
          : nExperiments < 10
            ? `${nExperiments} experiments completed. More data will improve trajectory analysis and statistical significance.`
            : ''}
        {controlNeeded && nExperiments >= 5 && (
          <span> Control comparison needs {'\u2265'}2 control + {'\u2265'}2 learned experiments (currently {nControlExps} control, {nLearnedExps} learned). Controls run automatically every 5th continuous experiment.</span>
        )}
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <button
          className="refresh-btn"
          style={{ fontSize: 12, padding: '5px 14px', fontWeight: 600 }}
          onClick={() => onStartExperiment({
            mode: 'continuous', n_cycles: 5, source: 'learning_panel',
            auto_harden: true, preflight_override: true, enforce_preflight: true,
          })}
        >
          Run 5 Continuous
        </button>
        <button
          className="refresh-btn"
          style={{ fontSize: 12, padding: '5px 14px', fontWeight: 600 }}
          onClick={() => onStartExperiment({
            mode: 'continuous', n_cycles: 10, source: 'learning_panel',
            auto_harden: true, preflight_override: true, enforce_preflight: true,
          })}
        >
          Run 10 Continuous
        </button>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          {nExperiments} experiment{nExperiments === 1 ? '' : 's'} so far
        </span>
      </div>
    </div>
  );
}

function LearningPanel({ onNavigateStrategy, onStartExperiment }) {
  const {
    learningTrajectory,
    fingerprintDiagnostics,
    mathFamilyCoverage,
    lastUpdated: sharedLastUpdated,
  } = useAriaData() || {};

  const [weights, setWeights] = useState(null);
  const [opRates, setOpRates] = useState(null);
  const [log, setLog] = useState(null);
  const [frontier, setFrontier] = useState(null);
  const [clusters, setClusters] = useState(null);
  const [routingHealth, setRoutingHealth] = useState(null);
  const [routingComparison, setRoutingComparison] = useState(null);
  const [gatingDiagnostics, setGatingDiagnostics] = useState(null);
  const [mathspaceImpact, setMathspaceImpact] = useState(null);
  const [compressionCoverage, setCompressionCoverage] = useState(null);
  const [learningSummary, setLearningSummary] = useState(null);
  const [topPrograms, setTopPrograms] = useState(null);
  const [controlComparison, setControlComparison] = useState(null);
  const [insightInteractions, setInsightInteractions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const targetSummary = useMemo(
    () => computeTargetSummary(topPrograms, routingComparison || routingHealth),
    [topPrograms, routingComparison, routingHealth]
  );

  const [openSections, setOpenSections] = useState({
    core: true,
    quality: false,
    diagnostics: false,
    raw: false
  });

  const toggleSection = (id) => {
    setOpenSections(prev => ({ ...prev, [id]: !prev[id] }));
  };

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
      safeFetch(`${API_BASE}/api/analytics/mathspace-impact`),
      safeFetch(`${API_BASE}/api/analytics/compression-coverage`),
      safeFetch(`${API_BASE}/api/analytics/learning-summary`),
      safeFetch(`${API_BASE}/api/programs?n=100&sort_by=loss_ratio`),
      safeFetch(`${API_BASE}/api/analytics/control-comparison`),
      safeFetch(`${API_BASE}/api/analytics/insight-interactions`),
    ]).then(([w, ops, lg, fr, cl, rh, rc, gd, mi, cc, ls, tp, ctrl, si]) => {
      if (!w && !ops && !lg && !fr && !cl && !rh && !rc && !gd && !mi && !cc && !ls && !si) {
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
      setMathspaceImpact(mi);
      setCompressionCoverage(cc);
      setLearningSummary(ls);
      setTopPrograms(Array.isArray(tp) ? tp : null);
      setControlComparison(ctrl);
      setInsightInteractions(si);
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
    <NarrativeProvider trajectoryData={learningTrajectory} weightData={weights}>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="card" style={{ padding: '12px 16px' }}>
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
          The AI scientist searches for novel neural network layer designs by generating random
          compositions of operations, testing if they compile and learn, and evolving the search
          grammar toward successful patterns. This tab shows what the system has learned so far.
        </p>
        <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: '8px 0 0' }}>
          Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'loading'} · Shared data: {sharedLastUpdated ? new Date(sharedLastUpdated).toLocaleTimeString() : 'loading'}
        </p>
      </div>
      <FeedbackLoopSummary
        weights={weights}
        trajectory={learningTrajectory}
        controlComparison={controlComparison}
        title="Aria's Analysis / Feedback Loop Summary"
      />
      <DataPopulateBar
        learningTrajectory={learningTrajectory}
        controlComparison={controlComparison}
        onStartExperiment={onStartExperiment}
      />

      <Section title="Core Learning" id="core" isOpen={openSections.core} onToggle={toggleSection}>
        <AriaThoughtProcess />
        <WhatIHaveLearned summary={learningSummary} />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <LearningTrajectory trajectory={learningTrajectory} onNavigateStrategy={onNavigateStrategy} onStartExperiment={onStartExperiment} />
          <ControlComparison data={controlComparison} onStartExperiment={onStartExperiment} />
        </div>
        <GrammarWeightsChart
          defaultWeights={weights?.default}
          learnedWeights={weights?.learned}
          explanation={weights?.explanation}
          onStartExperiment={onStartExperiment}
        />
      </Section>

      <Section title="Search Quality" id="quality" isOpen={openSections.quality} onToggle={toggleSection}>
        <ArchitectureRerunTelemetry telemetry={weights?.architecture_rerun_telemetry} />
        <TargetBalanceCards summary={targetSummary} />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <DataAccumulation
            title="Efficiency Frontier"
            current={Array.isArray(frontier) ? frontier.length : 0}
            threshold={3}
          >
            <EfficiencyFrontier frontier={frontier} />
          </DataAccumulation>
          <DataAccumulation
            title="Experiment Clusters"
            current={sampleCount(clusters, 'clusters')}
            threshold={MIN_SAMPLES}
          >
            <ExperimentClusters clustersData={clusters} />
          </DataAccumulation>
        </div>
      </Section>

      <Section title="Advanced Diagnostics" id="diagnostics" isOpen={openSections.diagnostics} onToggle={toggleSection}>
        <FingerprintDiagnosticsCard diagnostics={fingerprintDiagnostics} />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <DataAccumulation
            title="Routing Health"
            current={sampleCount(routingComparison || routingHealth, 'routing')}
            threshold={MIN_SAMPLES}
          >
            <RoutingHealth data={routingComparison || routingHealth} />
          </DataAccumulation>
          <DataAccumulation
            title="Gating Behavior Diagnostics"
            current={sampleCount(gatingDiagnostics, 'gating')}
            threshold={MIN_SAMPLES}
          >
            <GatingBehaviorDiagnostics data={gatingDiagnostics} />
          </DataAccumulation>
        </div>
        <MathFamilyCoverage data={mathFamilyCoverage} />
        <MathspaceImpact data={mathspaceImpact} />
        <CompressionCoverage data={compressionCoverage} programs={topPrograms} />
      </Section>

      <Section title="Raw Data" id="raw" isOpen={openSections.raw} onToggle={toggleSection}>
        <AdaptationSummary log={log} />
        <DataAccumulation
          title="Insight Synergy Matrix"
          current={insightInteractions?.total_interactions || (insightInteractions?.synergistic_pairs?.length || 0) + (insightInteractions?.antagonistic_pairs?.length || 0)}
          threshold={5}
        >
          <InsightSynergyMatrix data={insightInteractions} />
        </DataAccumulation>
        <OpSuccessTable opRates={opRates} />
        <LearningLog log={log} />
      </Section>
    </div>
    </NarrativeProvider>
  );
}

export default LearningPanel;
