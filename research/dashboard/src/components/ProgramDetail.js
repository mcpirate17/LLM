import React, { useState, useEffect } from 'react';
import GraphViewer from './GraphViewer';
import { lossColor, noveltyColor } from '../utils/colors';
import useCopyToClipboard from '../hooks/useCopyToClipboard';

const API_BASE = process.env.REACT_APP_API_URL || '';

/**
 * ProgramDetail — Modal showing computation graph, stage pipeline,
 * fingerprint radar chart, training metrics, similar architectures,
 * sandbox metrics, FLOPs, baseline comparison, training curve.
 */

function StagePipeline({ program }) {
  const stages = [
    { key: 'stage0_passed', label: 'Stage 0', sublabel: 'Compilation' },
    { key: 'stage05_passed', label: 'Stage 0.5', sublabel: 'Stability' },
    { key: 'stage1_passed', label: 'Stage 1', sublabel: 'Learning' },
  ];

  return (
    <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
      {stages.map((stage, i) => {
        const passed = program[stage.key];
        const color = passed ? 'var(--accent-green)' : 'var(--accent-red)';
        const bg = passed ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)';
        return (
          <React.Fragment key={stage.key}>
            {i > 0 && <span style={{ color: 'var(--text-muted)', fontSize: 16 }}>&rarr;</span>}
            <div style={{
              padding: '6px 12px',
              background: bg,
              border: `1px solid ${color}`,
              borderRadius: 6,
              textAlign: 'center',
              minWidth: 80,
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color }}>{passed ? 'PASS' : 'FAIL'}</div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{stage.label}</div>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}

function RadarChart({ program, size = 240 }) {
  // Extended radar with fingerprint metrics
  const axes = [
    { key: 'novelty_score', label: 'Novelty' },
    { key: 'structural_novelty', label: 'Structural' },
    { key: 'behavioral_novelty', label: 'Behavioral' },
    { key: 'fp_interaction_locality', label: 'Locality' },
    { key: 'fp_interaction_sparsity', label: 'Sparsity' },
    { key: 'fp_isotropy', label: 'Isotropy' },
    { key: 'fp_rank_ratio', label: 'Rank' },
    { key: 'fp_sensitivity_uniformity', label: 'Sensitivity' },
  ].filter(a => program[a.key] !== null && program[a.key] !== undefined);

  // Fall back to minimal if no fingerprint data
  if (axes.length < 3) {
    const fallback = [
      { key: 'novelty_score', label: 'Novelty' },
      { key: 'structural_novelty', label: 'Structural' },
      { key: 'behavioral_novelty', label: 'Behavioral' },
    ];
    axes.length = 0;
    axes.push(...fallback);
  }

  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 30;
  const n = axes.length;

  const getPoint = (i, val) => {
    const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
    const d = val * r;
    return { x: cx + d * Math.cos(angle), y: cy + d * Math.sin(angle) };
  };

  const rings = [0.25, 0.5, 0.75, 1.0];
  const values = axes.map(a => Math.min(program[a.key] || 0, 1));
  const points = values.map((v, i) => getPoint(i, v));
  const polygonPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z';

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {rings.map(ring => {
        const ringPoints = axes.map((_, i) => getPoint(i, ring));
        const d = ringPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z';
        return <path key={ring} d={d} fill="none" stroke="var(--border, #30363d)" strokeWidth={0.5} />;
      })}
      {axes.map((_, i) => {
        const end = getPoint(i, 1);
        return <line key={i} x1={cx} y1={cy} x2={end.x} y2={end.y}
          stroke="var(--border, #30363d)" strokeWidth={0.5} />;
      })}
      <path d={polygonPath} fill="rgba(188, 140, 255, 0.2)" stroke="var(--accent-purple, #bc8cff)" strokeWidth={2} />
      {points.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={3}
          fill="var(--accent-purple, #bc8cff)" stroke="var(--bg-secondary, #161b22)" strokeWidth={1.5} />
      ))}
      {axes.map((axis, i) => {
        const labelPt = getPoint(i, 1.25);
        return (
          <text key={i} x={labelPt.x} y={labelPt.y}
            textAnchor="middle" dominantBaseline="middle"
            fill="var(--text-secondary, #8b949e)" fontSize={9}>
            {axis.label}
          </text>
        );
      })}
    </svg>
  );
}

function TrainingCurve({ resultId }) {
  const [curve, setCurve] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/programs/${resultId}/training-curve`)
      .then(r => r.json())
      .then(d => setCurve(d))
      .catch(() => {});
  }, [resultId]);

  if (!curve || curve.length === 0) return null;

  const W = 350, H = 120;
  const pad = { l: 45, r: 10, t: 10, b: 25 };

  const losses = curve.map(c => c.loss).filter(l => l != null && isFinite(l));
  if (losses.length < 2) return null;

  const minL = Math.min(...losses);
  const maxL = Math.max(...losses);
  const rangeL = maxL - minL || 1;

  const xScale = i => pad.l + (i / (losses.length - 1)) * (W - pad.l - pad.r);
  const yScale = v => H - pad.b - ((v - minL) / rangeL) * (H - pad.t - pad.b);

  const pathD = losses.map((l, i) => `${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(l)}`).join(' ');

  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
        Training Curve
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="var(--border)" />
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="var(--border)" />
        <text x={pad.l - 5} y={yScale(maxL)} textAnchor="end" fill="var(--text-muted)" fontSize={9}>{maxL.toFixed(2)}</text>
        <text x={pad.l - 5} y={yScale(minL)} textAnchor="end" fill="var(--text-muted)" fontSize={9}>{minL.toFixed(2)}</text>
        <text x={W / 2} y={H - 3} textAnchor="middle" fill="var(--text-muted)" fontSize={9}>Step</text>
        <path d={pathD} fill="none" stroke="var(--accent-green)" strokeWidth={1.5} />
      </svg>
    </div>
  );
}

function MetricRow({ label, value }) {
  if (value === null || value === undefined || value === '--') return null;
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: '1px solid var(--border)' }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <span>{value}</span>
    </div>
  );
}

const TIER_COLORS = {
  screening: 'var(--accent-blue)',
  investigation: 'var(--accent-yellow)',
  validation: 'var(--accent-purple)',
  breakthrough: 'var(--accent-green)',
};

const TIER_LABELS = {
  screening: 'Screening',
  investigation: 'Investigation',
  validation: 'Validation',
  breakthrough: 'Breakthrough',
};

function TierBadge({ tier }) {
  if (!tier) return null;
  return (
    <span style={{
      padding: '2px 8px',
      borderRadius: 4,
      fontSize: 11,
      fontWeight: 600,
      color: TIER_COLORS[tier] || 'var(--text-muted)',
      background: `${TIER_COLORS[tier] || 'var(--text-muted)'}22`,
      border: `1px solid ${TIER_COLORS[tier] || 'var(--border)'}`,
      textTransform: 'uppercase',
    }}>
      {TIER_LABELS[tier] || tier}
    </span>
  );
}

function HypothesisInfo({ hypothesis }) {
  if (!hypothesis) return null;
  const colors = {
    confirmed: 'var(--accent-green)',
    refuted: 'var(--accent-red)',
    inconclusive: 'var(--accent-yellow)',
    pending: 'var(--text-muted)',
    testing: 'var(--accent-blue)',
  };
  return (
    <div style={{
      padding: 12, background: 'var(--bg-tertiary)', borderRadius: 4,
      borderLeft: `2px solid ${colors[hypothesis.status] || 'var(--border)'}`,
      fontSize: 13,
    }}>
      <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', marginBottom: 4,
        color: colors[hypothesis.status] || 'var(--text-muted)' }}>
        Hypothesis: {hypothesis.status}
      </div>
      <div style={{ marginBottom: 4 }}>{hypothesis.prediction}</div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
        <em>Metric:</em> {hypothesis.success_metric}
      </div>
      {hypothesis.outcome_summary && (
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4 }}>
          <em>Outcome:</em> {hypothesis.outcome_summary}
        </div>
      )}
    </div>
  );
}

function BenchmarkEvidenceSnapshot({ program, leaderboardEntry }) {
  const tier = leaderboardEntry?.tier;
  const isBreakthrough = tier === 'breakthrough';
  const ratio = Number(program?.baseline_loss_ratio);
  const hasRatio = Number.isFinite(ratio);
  const beatsBaseline = hasRatio && ratio < 1;

  if (!hasRatio && !isBreakthrough) return null;

  return (
    <div style={{
      marginTop: 12,
      padding: 10,
      background: 'var(--bg-tertiary)',
      borderRadius: 6,
      borderLeft: `3px solid ${beatsBaseline ? 'var(--accent-green)' : 'var(--accent-yellow)'}`,
    }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 6 }}>
        Benchmark Evidence Snapshot
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
        <div>
          <strong>Fixed-seed baseline ratio:</strong>{' '}
          {hasRatio ? ratio.toFixed(3) : 'Unavailable'}
          {hasRatio && (
            <span style={{ marginLeft: 6, color: beatsBaseline ? 'var(--accent-green)' : 'var(--accent-red)' }}>
              {beatsBaseline ? '(< 1.0, beats baseline)' : '(≥ 1.0, below baseline)'}
            </span>
          )}
        </div>
        <div>
          <strong>Interpretation:</strong>{' '}
          {hasRatio
            ? (beatsBaseline
              ? 'This architecture outperforms the fixed-seed transformer baseline on the same setup.'
              : 'This architecture does not yet beat the fixed-seed transformer baseline on this snapshot.')
            : 'Baseline comparison was not recorded for this result.'}
        </div>
        {isBreakthrough && (
          <div>
            <strong>Breakthrough note:</strong> tier promotion also requires multi-seed stability and robustness checks beyond this fixed-seed snapshot.
          </div>
        )}
      </div>
    </div>
  );
}

function EvidenceFlagChips({ flags }) {
  if (!flags) return null;
  const entries = [
    { key: 'has_baseline', label: 'Baseline' },
    { key: 'has_cka_artifact', label: 'CKA Artifact' },
    { key: 'has_multi_seed', label: 'Multi-Seed' },
    { key: 'has_hypothesis', label: 'Hypothesis' },
  ];
  return (
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
      {entries.map(({ key, label }) => {
        const ok = flags[key];
        return (
          <span key={key} style={{
            fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 4,
            color: ok ? 'var(--accent-green)' : 'var(--accent-red)',
            background: ok ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
            border: `1px solid ${ok ? 'var(--accent-green)' : 'var(--accent-red)'}44`,
          }}>
            {ok ? '\u2713' : '\u2717'} {label}
          </span>
        );
      })}
    </div>
  );
}

function HypothesisLineage({ chain }) {
  if (!chain || chain.length === 0) return <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No linked hypothesis</div>;
  const statusColors = {
    confirmed: 'var(--accent-green)',
    refuted: 'var(--accent-red)',
    inconclusive: 'var(--accent-yellow)',
    pending: 'var(--text-muted)',
    testing: 'var(--accent-blue)',
  };
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {chain.map((h, i) => (
        <div key={h.hypothesis_id || i} style={{
          fontSize: 12, padding: '4px 8px',
          borderLeft: `3px solid ${statusColors[h.status] || 'var(--border)'}`,
          color: 'var(--text-secondary)',
        }}>
          <span style={{ fontWeight: 600, color: statusColors[h.status] || 'var(--text-muted)', textTransform: 'uppercase', fontSize: 10, marginRight: 6 }}>
            [{h.status}]
          </span>
          {h.prediction || h.title || 'Untitled hypothesis'}
        </div>
      ))}
    </div>
  );
}

function OutcomesByPhase({ outcomes }) {
  if (!outcomes) return null;
  const phases = [
    { key: 'screening', label: 'Screening' },
    { key: 'investigation', label: 'Investigation' },
    { key: 'validation', label: 'Validation' },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {phases.map(({ key, label }) => {
        const data = outcomes[key];
        if (!data) return (
          <div key={key} style={{ fontSize: 12, color: 'var(--text-muted)', padding: '4px 0', borderBottom: '1px solid var(--border)' }}>
            {label}: --
          </div>
        );
        const passed = data.passed;
        return (
          <div key={key} style={{ fontSize: 12, padding: '4px 0', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
            <span style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              {data.loss_ratio != null && <span>LR: {Number(data.loss_ratio).toFixed(4)}</span>}
              {data.novelty != null && <span>Nov: {Number(data.novelty).toFixed(3)}</span>}
              {data.robustness != null && <span>Rob: {Number(data.robustness).toFixed(3)}</span>}
              {data.baseline_ratio != null && <span>BL: {Number(data.baseline_ratio).toFixed(3)}</span>}
              {data.multi_seed_std != null && <span>Std: {Number(data.multi_seed_std).toFixed(4)}</span>}
              {passed !== undefined && (
                <span style={{
                  fontSize: 10, fontWeight: 600, padding: '1px 6px', borderRadius: 3,
                  color: passed ? 'var(--accent-green)' : 'var(--accent-red)',
                  background: passed ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                }}>
                  {passed ? 'PASS' : 'FAIL'}
                </span>
              )}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function FailureContext({ context }) {
  if (!context || (!context.stage_at_death && !context.error_type)) return null;
  const topErrors = Object.entries(context.experiment_errors || {}).sort((a, b) => b[1] - a[1]).slice(0, 5);
  return (
    <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
      {context.stage_at_death && <div>Stage at death: <strong>{context.stage_at_death}</strong></div>}
      {context.error_type && <div>Error type: <strong>{context.error_type}</strong></div>}
      {topErrors.length > 0 && (
        <div style={{ marginTop: 6 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>Top Experiment Errors</div>
          {topErrors.map(([err, count]) => (
            <div key={err} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', borderBottom: '1px solid var(--border)' }}>
              <span style={{ fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 300 }}>{err}</span>
              <span style={{ fontSize: 11, color: 'var(--accent-red)', flexShrink: 0, marginLeft: 8 }}>{count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function RecommendationCard({ recommendation }) {
  if (!recommendation) return null;
  const actionColors = {
    'investigate': 'var(--accent-blue)',
    'validate': 'var(--accent-purple)',
    're-validate': 'var(--accent-purple)',
    'scale up or publish': 'var(--accent-green)',
    'publish': 'var(--accent-green)',
    're-investigate or archive': 'var(--accent-yellow)',
    'archive': 'var(--accent-red)',
  };
  const color = actionColors[recommendation.action] || 'var(--accent-blue)';
  return (
    <div style={{
      padding: 12, borderRadius: 6, border: `1px solid ${color}`,
      background: `${color}11`,
    }}>
      <div style={{ fontSize: 14, fontWeight: 700, color, textTransform: 'uppercase', marginBottom: 4 }}>
        {recommendation.action}
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
        {recommendation.rationale}
      </div>
      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
        Confidence: {recommendation.confidence}
      </div>
    </div>
  );
}

const TOKEN_MIXING_OPS = {
  local_window_attn: { label: 'Local Window Attention', desc: 'Windowed causal self-attention (Q=K=V)', family: 'attention', complexity: 'linear' },
  sliding_window_mask: { label: 'Sliding Window Mask', desc: 'Exponential distance decay for windowed mixing', family: 'attention', complexity: 'linear' },
  softmax_last: { label: 'Softmax (dim)', desc: 'Standard softmax along last dimension', family: 'attention', complexity: 'quadratic' },
  softmax_seq: { label: 'Softmax (seq)', desc: 'Softmax along sequence dimension', family: 'attention', complexity: 'quadratic' },
  causal_mask: { label: 'Causal Mask', desc: 'Lower-triangular causal masking', family: 'attention', complexity: 'quadratic' },
  multi_head_mix: { label: 'Multi-Head Mix', desc: 'Multi-head reshape + per-head normalization', family: 'attention', complexity: 'quadratic' },
  selective_scan: { label: 'Selective Scan (SSM)', desc: 'Input-dependent state scan (Mamba-style)', family: 'ssm', complexity: 'linear' },
  conv1d_seq: { label: '1D Conv (depthwise)', desc: 'Depthwise convolution along sequence', family: 'conv', complexity: 'linear' },
  rfft_seq: { label: 'FFT (forward)', desc: 'Real FFT along sequence dimension', family: 'frequency', complexity: 'linear' },
  irfft_seq: { label: 'FFT (inverse)', desc: 'Inverse real FFT along sequence', family: 'frequency', complexity: 'linear' },
  sort_seq: { label: 'Sort Mixing', desc: 'Sort along sequence by learned key', family: 'sorting', complexity: 'nlogn' },
  argsort_seq: { label: 'Argsort Mixing', desc: 'Argsort along sequence dimension', family: 'sorting', complexity: 'nlogn' },
  token_pool_restore: { label: 'Token Pooling', desc: 'Pool adjacent token pairs then restore', family: 'pooling', complexity: 'linear' },
  basis_expansion: { label: 'Basis Expansion', desc: 'Sinusoidal basis projection (neural operator)', family: 'functional', complexity: 'linear' },
  integral_kernel: { label: 'Integral Kernel', desc: 'Learned kernel mixing over positions', family: 'functional', complexity: 'quadratic' },
  fixed_point_iter: { label: 'Fixed-Point Iter', desc: 'Implicit fixed-point iteration (DEQ-style)', family: 'functional', complexity: 'linear' },
};

const FAMILY_COLORS = {
  attention: 'var(--accent-blue)',
  ssm: 'var(--accent-green)',
  conv: 'var(--accent-yellow)',
  frequency: 'var(--accent-purple)',
  sorting: 'var(--accent-red)',
  pooling: 'var(--text-muted)',
  functional: '#e0a060',
};

const FAMILY_LABELS = {
  attention: 'QKV-based',
  ssm: 'State Space',
  conv: 'Convolution',
  frequency: 'Frequency Domain',
  sorting: 'Sort-based',
  pooling: 'Pooling',
  functional: 'Functional/Operator',
};

function TokenMixingTaxonomy({ graphJson }) {
  if (!graphJson) return null;

  const rawNodes = graphJson.nodes || [];
  const nodes = Array.isArray(rawNodes) ? rawNodes : Object.values(rawNodes);
  const detected = [];
  const seen = new Set();
  for (const node of nodes) {
    if (!node || typeof node !== 'object') continue;
    const opName = node.op || node.op_name;
    if (opName && TOKEN_MIXING_OPS[opName] && !seen.has(opName)) {
      seen.add(opName);
      detected.push({ op: opName, ...TOKEN_MIXING_OPS[opName] });
    }
  }

  if (detected.length === 0) return null;

  const families = [...new Set(detected.map(d => d.family))];
  const hasQKV = families.includes('attention');
  const summary = hasQKV
    ? `Uses QKV-style attention${families.length > 1 ? ' + ' + families.filter(f => f !== 'attention').map(f => FAMILY_LABELS[f]).join(', ') : ''}`
    : `QKV-free: ${families.map(f => FAMILY_LABELS[f]).join(' + ')}`;

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
        Token Mixing Mechanism
      </div>
      <div style={{
        padding: '8px 12px', background: 'var(--bg-tertiary)', borderRadius: 6,
        borderLeft: `3px solid ${hasQKV ? 'var(--accent-blue)' : 'var(--accent-green)'}`,
      }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: hasQKV ? 'var(--accent-blue)' : 'var(--accent-green)', marginBottom: 6 }}>
          {summary}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {detected.map(d => (
            <div key={d.op} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
              <span style={{
                padding: '1px 6px', borderRadius: 3, fontSize: 10, fontWeight: 600,
                background: `${FAMILY_COLORS[d.family]}22`, color: FAMILY_COLORS[d.family],
              }}>
                {FAMILY_LABELS[d.family]}
              </span>
              <code style={{ color: 'var(--text-secondary)', fontSize: 11 }}>{d.op}</code>
              <span style={{ color: 'var(--text-muted)' }}>{d.desc}</span>
              <span style={{
                marginLeft: 'auto', fontSize: 10, color: 'var(--text-muted)',
                fontStyle: 'italic', flexShrink: 0,
              }}>
                {d.complexity === 'linear' ? 'O(n)' : d.complexity === 'nlogn' ? 'O(n log n)' : 'O(n²)'}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const GATING_OPS = new Set(['topk_gate']);
const ACTIVATION_GATING_OPS = new Set(['sigmoid', 'silu', 'gelu']);

function detectGatingOps(graphJson) {
  if (!graphJson) return { hasGating: false, gatingOps: [], activationGates: [] };
  const nodes = graphJson.nodes || {};
  const nodeList = Array.isArray(nodes) ? nodes : Object.values(nodes);
  const gatingOps = [];
  const activationGates = [];
  const seen = new Set();
  for (const node of nodeList) {
    if (!node || typeof node !== 'object') continue;
    const op = node.op_name || node.op;
    if (!op || seen.has(op)) continue;
    seen.add(op);
    if (GATING_OPS.has(op)) gatingOps.push(op);
    if (ACTIVATION_GATING_OPS.has(op)) activationGates.push(op);
  }
  return { hasGating: gatingOps.length > 0, gatingOps, activationGates };
}

function entropyInterpretation(entropy, nExperts) {
  if (entropy == null || nExperts == null || nExperts < 2) return null;
  const maxEntropy = Math.log2(nExperts);
  const ratio = maxEntropy > 0 ? entropy / maxEntropy : 0;
  if (ratio > 0.85) return { label: 'Well-distributed', color: 'var(--accent-green)', risk: 'low' };
  if (ratio > 0.5) return { label: 'Moderate skew', color: 'var(--accent-yellow)', risk: 'medium' };
  return { label: 'Route collapse risk', color: 'var(--accent-red)', risk: 'high' };
}

function GatingDiagnostics({ program }) {
  const graphJson = program.graph_json_parsed;
  const { hasGating, gatingOps, activationGates } = detectGatingOps(graphJson);

  const hasRouting = program.routing_drop_rate != null ||
    program.routing_utilization_entropy != null ||
    program.routing_confidence_mean != null;

  if (!hasGating && !hasRouting) return null;

  let expertUtil = null;
  let nExperts = null;
  if (program.routing_expert_utilization_json) {
    try {
      const parsed = typeof program.routing_expert_utilization_json === 'string'
        ? JSON.parse(program.routing_expert_utilization_json)
        : program.routing_expert_utilization_json;

      let normalized = null;
      if (Array.isArray(parsed)) {
        normalized = parsed;
      } else if (parsed && typeof parsed === 'object') {
        normalized = Object.values(parsed);
      }

      if (Array.isArray(normalized)) {
        expertUtil = normalized
          .map(v => Number(v))
          .filter(v => Number.isFinite(v));
        nExperts = expertUtil.length;
      }
    } catch { /* ignore */ }
  }

  const entropy = program.routing_utilization_entropy;
  const interpretation = entropyInterpretation(entropy, nExperts || 2);
  const dropRate = program.routing_drop_rate;
  const confMean = program.routing_confidence_mean;
  const confStd = program.routing_confidence_std;
  const tokProcessed = program.routing_tokens_processed;
  const tokSkipped = program.routing_tokens_skipped;
  const overflows = program.routing_capacity_overflow_count;

  const maxUtil = Array.isArray(expertUtil) && expertUtil.length > 0
    ? Math.max(...expertUtil)
    : 0;

  const tokenRetention = (tokProcessed != null && (tokProcessed + (tokSkipped || 0)) > 0)
    ? Number(tokProcessed) / Number(tokProcessed + (tokSkipped || 0))
    : (dropRate != null ? Math.max(0, Math.min(1, 1 - Number(dropRate))) : null);

  const tokenRetentionCurve = (() => {
    if (program.routing_expert_utilization_json) {
      try {
        const parsed = typeof program.routing_expert_utilization_json === 'string'
          ? JSON.parse(program.routing_expert_utilization_json)
          : program.routing_expert_utilization_json;
        if (Array.isArray(parsed) && parsed.length > 0 && typeof parsed[0] === 'object') {
          const points = parsed
            .map((point, idx) => ({
              step: Number(point.step ?? idx),
              retention: Number(point.retention ?? point.token_retention ?? point.value),
            }))
            .filter(point => Number.isFinite(point.step) && Number.isFinite(point.retention))
            .map(point => ({ step: point.step, retention: Math.max(0, Math.min(1, point.retention)) }));
          if (points.length >= 2) return points;
        }
      } catch {
        // ignore malformed payloads
      }
    }
    if (tokenRetention != null) {
      return [
        { step: 0, retention: 1.0 },
        { step: 1, retention: tokenRetention },
      ];
    }
    return [];
  })();

  const tokenCurvePath = (() => {
    if (!tokenRetentionCurve || tokenRetentionCurve.length < 2) return null;
    const maxStep = Math.max(...tokenRetentionCurve.map(point => point.step), 1);
    return tokenRetentionCurve
      .map((point, idx) => {
        const x = (point.step / maxStep) * 100;
        const y = (1 - point.retention) * 100;
        return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(' ');
  })();

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
        Gating & Routing Diagnostics
      </div>
      <div style={{
        padding: '10px 12px', background: 'var(--bg-tertiary)', borderRadius: 6,
        borderLeft: `3px solid ${interpretation ? interpretation.color : 'var(--accent-blue)'}`,
        display: 'flex', flexDirection: 'column', gap: 10,
      }}>
        {/* Detected ops */}
        {(gatingOps.length > 0 || activationGates.length > 0) && (
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600 }}>Gating ops:</span>
            {gatingOps.map(op => (
              <code key={op} style={{
                fontSize: 10, padding: '1px 6px', borderRadius: 3,
                background: 'rgba(188, 140, 255, 0.15)', color: 'var(--accent-purple)',
              }}>{op}</code>
            ))}
            {activationGates.map(op => (
              <code key={op} style={{
                fontSize: 10, padding: '1px 6px', borderRadius: 3,
                background: 'var(--bg-secondary)', color: 'var(--text-muted)',
              }}>{op}</code>
            ))}
          </div>
        )}

        {/* Routing mode */}
        {program.routing_mode && (
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            <span style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Mode:</span>{' '}
            <span style={{ color: 'var(--accent-blue)' }}>{program.routing_mode}</span>
          </div>
        )}

        {/* Key metrics row */}
        {hasRouting && (
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12 }}>
            {dropRate != null && (
              <div>
                <div style={{ color: 'var(--text-muted)', fontSize: 10, fontWeight: 600, marginBottom: 2 }}>Drop Rate</div>
                <span style={{
                  fontWeight: 600, fontSize: 14,
                  color: dropRate > 0.3 ? 'var(--accent-red)' : dropRate > 0.1 ? 'var(--accent-yellow)' : 'var(--accent-green)',
                }}>
                  {(dropRate * 100).toFixed(1)}%
                </span>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                  {dropRate > 0.3 ? 'High token loss' : dropRate > 0.1 ? 'Moderate' : 'Healthy'}
                </div>
              </div>
            )}
            {entropy != null && (
              <div>
                <div style={{ color: 'var(--text-muted)', fontSize: 10, fontWeight: 600, marginBottom: 2 }}>Utilization Entropy</div>
                <span style={{ fontWeight: 600, fontSize: 14, color: interpretation?.color || 'var(--text-secondary)' }}>
                  {Number(entropy).toFixed(3)}
                </span>
                {interpretation && (
                  <div style={{ fontSize: 10, color: interpretation.color }}>{interpretation.label}</div>
                )}
                {interpretation && (
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    Collapse risk: <strong style={{ color: interpretation.color }}>{interpretation.risk}</strong>
                  </div>
                )}
              </div>
            )}
            {confMean != null && (
              <div>
                <div style={{ color: 'var(--text-muted)', fontSize: 10, fontWeight: 600, marginBottom: 2 }}>Gate Confidence</div>
                <span style={{
                  fontWeight: 600, fontSize: 14,
                  color: confMean > 0.8 ? 'var(--accent-green)' : confMean > 0.5 ? 'var(--accent-yellow)' : 'var(--accent-red)',
                }}>
                  {Number(confMean).toFixed(3)}
                </span>
                {confStd != null && (
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 4 }}>
                    ±{Number(confStd).toFixed(3)}
                  </span>
                )}
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                  {confMean > 0.8 ? 'Decisive' : confMean > 0.5 ? 'Moderate' : 'Uncertain routing'}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Token-retention curve */}
        {tokenCurvePath && (
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>
              Token Retention Curve
            </div>
            <div style={{ border: '1px solid var(--border)', borderRadius: 4, background: 'var(--bg-secondary)', padding: '4px 6px' }}>
              <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: 54, display: 'block' }}>
                <path d={tokenCurvePath} fill="none" stroke="var(--accent-blue)" strokeWidth="2" />
              </svg>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
              Final retention: {tokenRetention != null ? `${(tokenRetention * 100).toFixed(1)}%` : 'not measured'}
            </div>
          </div>
        )}

        {/* Token flow */}
        {(tokProcessed != null || tokSkipped != null || overflows != null) && (
          <div style={{ display: 'flex', gap: 12, fontSize: 11, color: 'var(--text-secondary)', flexWrap: 'wrap' }}>
            {tokProcessed != null && <span>Processed: <strong>{Number(tokProcessed).toLocaleString()}</strong></span>}
            {tokSkipped != null && (
              <span>Skipped: <strong style={{ color: tokSkipped > 0 ? 'var(--accent-yellow)' : 'var(--text-secondary)' }}>
                {Number(tokSkipped).toLocaleString()}
              </strong></span>
            )}
            {overflows != null && overflows > 0 && (
              <span>Capacity overflows: <strong style={{ color: 'var(--accent-red)' }}>
                {Number(overflows).toLocaleString()}
              </strong></span>
            )}
          </div>
        )}

        {/* Expert utilization bars */}
        {expertUtil && expertUtil.length > 0 && (
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>
              Expert Utilization ({expertUtil.length} experts)
            </div>
            <div style={{ display: 'flex', gap: 2, alignItems: 'flex-end', height: 40 }}>
              {expertUtil.map((val, i) => {
                const v = Number(val);
                const pct = maxUtil > 0 ? (v / maxUtil) * 100 : 0;
                const isCollapsed = nExperts > 2 && v < (maxUtil * 0.1);
                return (
                  <div
                    key={i}
                    title={`Expert ${i}: ${(v * 100).toFixed(1)}%`}
                    style={{
                      flex: 1,
                      height: `${Math.max(pct, 4)}%`,
                      background: isCollapsed ? 'var(--accent-red)' : 'var(--accent-blue)',
                      borderRadius: '2px 2px 0 0',
                      opacity: isCollapsed ? 0.7 : 0.6,
                      minWidth: 4,
                      maxWidth: 24,
                    }}
                  />
                );
              })}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
              <span>E0</span>
              <span>E{expertUtil.length - 1}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ProgramDetail({ resultId, onClose, onActionComplete, onSelectExperiment, onViewInLeaderboard, onSelectCampaign }) {
  const [program, setProgram] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [scaleUpOpen, setScaleUpOpen] = useState(false);
  const [scaleUpConfig, setScaleUpConfig] = useState({ steps: 5000, batch_size: 8, seq_len: 512 });
  const [scaleUpStarting, setScaleUpStarting] = useState(false);
  const [leaderboardEntry, setLeaderboardEntry] = useState(null);
  const [actionStarting, setActionStarting] = useState(null);
  const [actionError, setActionError] = useState(null);
  const [linkedHypothesis, setLinkedHypothesis] = useState(null);
  const [linkedDecision, setLinkedDecision] = useState(null);
  const [linkedExperiment, setLinkedExperiment] = useState(null);
  const [linkedCampaign, setLinkedCampaign] = useState(null);
  const [provenanceOpen, setProvenanceOpen] = useState(true);
  const [decisionPacket, setDecisionPacket] = useState(null);
  const [decisionPacketLoading, setDecisionPacketLoading] = useState(false);
  const [decisionPacketError, setDecisionPacketError] = useState(null);
  const [decisionPacketOpen, setDecisionPacketOpen] = useState(true);
  const [manifestLoading, setManifestLoading] = useState(false);
  const [manifestCopied, copyManifest] = useCopyToClipboard();

  const fetchAndCopyManifest = () => {
    if (!resultId) return;
    setManifestLoading(true);
    fetch(`${API_BASE}/api/reproducibility-manifest/${resultId}`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(d => {
        copyManifest(JSON.stringify(d, null, 2));
        setManifestLoading(false);
      })
      .catch(() => { setManifestLoading(false); });
  };

  const fetchDecisionPacket = () => {
    if (!resultId) return;
    setDecisionPacketLoading(true);
    setDecisionPacketError(null);
    fetch(`${API_BASE}/api/decision-packet/${resultId}`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(d => { setDecisionPacket(d); setDecisionPacketLoading(false); })
      .catch(e => { setDecisionPacketError('Failed: ' + e.message); setDecisionPacketLoading(false); });
  };

  useEffect(() => {
    if (!resultId) return;
    setLoading(true);
    setError(null);
    setLinkedHypothesis(null);
    setLinkedDecision(null);
    setLinkedExperiment(null);
    setLinkedCampaign(null);
    fetch(`${API_BASE}/api/programs/${resultId}`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(d => {
        setProgram(d);
        setLoading(false);
        // Fetch linked hypothesis via experiment
        if (d?.experiment_id) {
          fetch(`${API_BASE}/api/experiments/${d.experiment_id}`)
            .then(r => r.ok ? r.json() : null)
            .then(expData => {
              if (expData?.experiment) {
                setLinkedExperiment(expData.experiment);
                if (expData.experiment.campaign_id) {
                  setLinkedCampaign({ campaign_id: expData.experiment.campaign_id, title: expData.experiment.campaign_title || expData.experiment.campaign_id });
                  // Find hypothesis linked to this experiment
                  fetch(`${API_BASE}/api/campaigns/${expData.experiment.campaign_id}/hypotheses`)
                    .then(r => r.ok ? r.json() : [])
                    .then(hyps => {
                      const linked = (Array.isArray(hyps) ? hyps : []).find(
                        h => h.experiment_id === d.experiment_id
                      );
                      if (linked) setLinkedHypothesis(linked);
                    })
                    .catch(() => {});
                  // Find decisions mentioning this result
                  fetch(`${API_BASE}/api/campaigns/${expData.experiment.campaign_id}/decisions`)
                    .then(r => r.ok ? r.json() : [])
                    .then(decs => {
                      const linked = (Array.isArray(decs) ? decs : []).find(d => {
                        const evidenceIds = d.evidence_ids || [];
                        return Array.isArray(evidenceIds) && evidenceIds.includes(resultId);
                      });
                      if (linked) setLinkedDecision(linked);
                    })
                    .catch(() => {});
                }
              }
            })
            .catch(() => {});
        }
      })
      .catch(e => { setError('Failed to load program: ' + e.message); setLoading(false); });
    // Fetch leaderboard entry for this result
    fetch(`${API_BASE}/api/leaderboard?limit=200`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.entries) {
          const entry = data.entries.find(e => e.result_id === resultId);
          setLeaderboardEntry(entry || null);
        }
      })
      .catch(() => {});
  }, [resultId]);

  if (!resultId) return null;

  const fmt = (v, d = 4) => v != null ? Number(v).toFixed(d) : '--';
  const fmtMs = v => v != null ? `${Number(v).toFixed(1)}ms` : '--';
  const fmtMem = v => v != null ? `${Number(v).toFixed(1)}MB` : '--';
  const fmtInt = v => v != null ? Number(v).toLocaleString() : '--';

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ fontSize: 16, margin: 0 }}>Program Detail</h3>
          <button className="refresh-btn" onClick={onClose} style={{ fontSize: 18, lineHeight: 1, padding: '4px 8px' }}>&times;</button>
        </div>

        {loading ? (
          <p style={{ color: 'var(--text-muted)' }}>Loading...</p>
        ) : error ? (
          <p style={{ color: 'var(--accent-red)' }}>{error}</p>
        ) : !program ? (
          <p style={{ color: 'var(--accent-red)' }}>Program not found</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Header info */}
            <div>
              <div style={{ fontFamily: 'monospace', fontSize: 13, color: 'var(--accent-blue)', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
                <span>{program.graph_fingerprint}</span>
                {program.stage_at_death && program.stage_at_death !== 'survived' && (
                  <span style={{ fontSize: 11, color: 'var(--accent-red)' }}>
                    died at {program.stage_at_death}
                  </span>
                )}
                {leaderboardEntry && (
                  <>
                    <TierBadge tier={leaderboardEntry.tier} />
                    <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-green)' }}>
                      Score: {Number(leaderboardEntry.composite_score).toFixed(3)}
                    </span>
                  </>
                )}
              </div>
              <StagePipeline program={program} />
            </div>

            {/* Provenance & Context */}
            {(program.experiment_id || linkedHypothesis || leaderboardEntry || linkedCampaign) && (
              <div style={{
                background: 'var(--bg-tertiary)',
                borderRadius: 6,
                border: '1px solid var(--border)',
                overflow: 'hidden',
              }}>
                <div
                  onClick={() => setProvenanceOpen(!provenanceOpen)}
                  style={{
                    padding: '8px 12px',
                    cursor: 'pointer',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    userSelect: 'none',
                  }}
                >
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                    Provenance & Context
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {provenanceOpen ? '▾ collapse' : '▸ expand'}
                  </span>
                </div>
                {provenanceOpen && (
                  <div style={{ padding: '0 12px 12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {/* Source experiment */}
                    {program.experiment_id && (
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90 }}>Experiment:</span>
                        <span style={{ fontFamily: 'monospace', fontSize: 11 }}>{program.experiment_id.slice(0, 12)}</span>
                        {linkedExperiment?.started_at && (
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            {new Date(linkedExperiment.started_at).toLocaleString()}
                          </span>
                        )}
                        {linkedExperiment?.experiment_type && (
                          <span style={{ fontSize: 10, color: 'var(--accent-blue)', border: '1px solid var(--accent-blue)', borderRadius: 3, padding: '0 4px' }}>
                            {linkedExperiment.experiment_type}
                          </span>
                        )}
                      </div>
                    )}

                    {/* Hypothesis 1-liner */}
                    {linkedHypothesis && (
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'baseline', gap: 8 }}>
                        <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90, flexShrink: 0 }}>Hypothesis:</span>
                        <span style={{
                          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 400,
                          color: linkedHypothesis.status === 'confirmed' ? 'var(--accent-green)' :
                                 linkedHypothesis.status === 'refuted' ? 'var(--accent-red)' : 'var(--text-secondary)',
                        }} title={linkedHypothesis.prediction}>
                          [{linkedHypothesis.status}] {linkedHypothesis.prediction}
                        </span>
                      </div>
                    )}

                    {/* Campaign */}
                    {linkedCampaign && (
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90 }}>Campaign:</span>
                        <span>{linkedCampaign.title || linkedCampaign.campaign_id}</span>
                      </div>
                    )}

                    {/* Leaderboard status */}
                    {leaderboardEntry && (
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90 }}>Leaderboard:</span>
                        <TierBadge tier={leaderboardEntry.tier} />
                        <span style={{ fontWeight: 600, color: 'var(--accent-green)' }}>
                          {Number(leaderboardEntry.composite_score).toFixed(3)}
                        </span>
                        {leaderboardEntry.tier === 'screening' && (
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            needs investigation to advance
                          </span>
                        )}
                        {leaderboardEntry.tier === 'investigation' && !leaderboardEntry.investigation_passed && (
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            investigation in progress
                          </span>
                        )}
                        {leaderboardEntry.tier === 'investigation' && leaderboardEntry.investigation_passed && (
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            ready for validation
                          </span>
                        )}
                      </div>
                    )}

                    {/* Quick nav links */}
                    <div style={{ display: 'flex', gap: 6, marginTop: 4, flexWrap: 'wrap' }}>
                      {program.experiment_id && onSelectExperiment && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 11, padding: '3px 10px' }}
                          onClick={() => { onClose(); onSelectExperiment(program.experiment_id); }}
                        >
                          Open Experiment
                        </button>
                      )}
                      {leaderboardEntry && onViewInLeaderboard && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 11, padding: '3px 10px' }}
                          onClick={() => { onClose(); onViewInLeaderboard(resultId); }}
                        >
                          View in Leaderboard
                        </button>
                      )}
                      {linkedCampaign && onSelectCampaign && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 11, padding: '3px 10px' }}
                          onClick={() => { onClose(); onSelectCampaign(linkedCampaign.campaign_id); }}
                        >
                          Open Campaign
                        </button>
                      )}
                      <button
                        className="refresh-btn"
                        style={{
                          fontSize: 11, padding: '3px 10px',
                          background: decisionPacket ? 'rgba(188, 140, 255, 0.15)' : undefined,
                          borderColor: 'var(--accent-purple)',
                          color: 'var(--accent-purple)',
                        }}
                        disabled={decisionPacketLoading}
                        onClick={fetchDecisionPacket}
                      >
                        {decisionPacketLoading ? 'Loading...' : 'Decision Packet'}
                      </button>
                      <button
                        className="refresh-btn"
                        style={{ fontSize: 11, padding: '3px 10px' }}
                        disabled={manifestLoading}
                        onClick={fetchAndCopyManifest}
                      >
                        {manifestLoading ? 'Loading...' : manifestCopied ? 'Copied!' : 'Copy Manifest'}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Decision Packet */}
            {decisionPacketError && (
              <div style={{ padding: 8, background: 'rgba(248, 81, 73, 0.1)', border: '1px solid var(--accent-red)', borderRadius: 4, fontSize: 12, color: 'var(--accent-red)' }}>
                {decisionPacketError}
              </div>
            )}
            {decisionPacket && (
              <div style={{
                background: 'var(--bg-tertiary)', borderRadius: 6,
                border: '1px solid var(--accent-purple)', overflow: 'hidden',
              }}>
                <div
                  onClick={() => setDecisionPacketOpen(!decisionPacketOpen)}
                  style={{
                    padding: '8px 12px', cursor: 'pointer',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    userSelect: 'none', background: 'rgba(188, 140, 255, 0.08)',
                  }}
                >
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-purple)', textTransform: 'uppercase' }}>
                    Decision Packet
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {decisionPacketOpen ? '\u25BE collapse' : '\u25B8 expand'}
                  </span>
                </div>
                {decisionPacketOpen && (
                  <div style={{ padding: '8px 12px 12px', display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {/* Evidence Flags */}
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Evidence Flags</div>
                      <EvidenceFlagChips flags={decisionPacket.evidence_flags} />
                    </div>
                    {/* Hypothesis Lineage */}
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Hypothesis Lineage</div>
                      <HypothesisLineage chain={decisionPacket.hypothesis_chain} />
                    </div>
                    {/* Outcomes by Phase */}
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Outcomes by Phase</div>
                      <OutcomesByPhase outcomes={decisionPacket.outcomes} />
                    </div>
                    {/* Failure Context */}
                    {(decisionPacket.failure_context?.stage_at_death || decisionPacket.failure_context?.error_type) && (
                      <div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Failure Context</div>
                        <FailureContext context={decisionPacket.failure_context} />
                      </div>
                    )}
                    {/* Recommendation */}
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Recommendation</div>
                      <RecommendationCard recommendation={decisionPacket.recommendation} />
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Error if failed */}
            {(program.error_message || program.stage0_error) && (
              <div style={{
                padding: 8,
                background: 'rgba(248, 81, 73, 0.1)',
                border: '1px solid var(--accent-red)',
                borderRadius: 4,
                fontSize: 12,
                fontFamily: 'monospace',
                color: 'var(--accent-red)',
              }}>
                {program.error_type && (
                  <span style={{ fontWeight: 600 }}>[{program.error_type}] </span>
                )}
                {program.error_message || program.stage0_error}
              </div>
            )}
            {actionError && (
              <div style={{
                padding: 8,
                background: 'rgba(248, 81, 73, 0.1)',
                border: '1px solid var(--accent-red)',
                borderRadius: 4,
                fontSize: 12,
                color: 'var(--accent-red)',
              }}>
                {actionError}
              </div>
            )}

            {/* Metrics + Radar side by side */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
                  Core Metrics
                </div>
                <div style={{ fontSize: 13 }}>
                  <MetricRow label="Parameters" value={program.param_count ? `${(program.param_count / 1e6).toFixed(2)}M` : null} />
                  <MetricRow label="Loss Ratio" value={program.loss_ratio != null ?
                    <span style={{
                      color: lossColor(program.loss_ratio),
                      fontWeight: program.loss_ratio < 0.5 ? 600 : 'normal',
                    }} title={program.loss_ratio < 0.5 ? 'Learned quickly — strong candidate' : program.loss_ratio < 0.7 ? 'Moderate learning' : 'Slow learning'}>
                      {fmt(program.loss_ratio)}
                    </span> : null} />
                  <MetricRow label="Final Loss" value={fmt(program.final_loss)} />
                  <MetricRow label="Baseline Ratio" value={program.baseline_loss_ratio != null ?
                    <span style={{
                      color: program.baseline_loss_ratio < 1 ? 'var(--accent-green)' : 'var(--accent-red)',
                      fontWeight: program.baseline_loss_ratio < 1 ? 600 : 'normal',
                    }} title={program.baseline_loss_ratio < 1 ? 'Beats a standard transformer!' : 'Underperforms a transformer of same size'}>
                      {fmt(program.baseline_loss_ratio)} {program.baseline_loss_ratio < 1 ? '(beats transformer)' : ''}
                    </span> : null} />
                  <MetricRow label="Throughput" value={program.throughput_tok_s != null ? `${Number(program.throughput_tok_s).toFixed(0)} tok/s` : null} />
                  <MetricRow label="Novelty" value={program.novelty_score != null ?
                    <span style={{
                      color: noveltyColor(program.novelty_score),
                    }} title={program.novelty_score > 0.8 ? 'Very different from known architectures' : program.novelty_score > 0.5 ? 'Moderately novel' : 'Similar to existing architectures'}>
                      {fmt(program.novelty_score, 3)}
                    </span> : null} />
                  <MetricRow label="Similar To" value={program.most_similar_to} />
                </div>

                <BenchmarkEvidenceSnapshot program={program} leaderboardEntry={leaderboardEntry} />

                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8, marginTop: 12 }}>
                  Sandbox Timing
                </div>
                <div style={{ fontSize: 13 }}>
                  <MetricRow label="Compile" value={fmtMs(program.compile_time_ms)} />
                  <MetricRow label="Forward" value={fmtMs(program.forward_time_ms)} />
                  <MetricRow label="Backward" value={fmtMs(program.backward_time_ms)} />
                  <MetricRow label="Peak Memory" value={fmtMem(program.peak_memory_mb)} />
                  <MetricRow label="FLOPs (fwd)" value={program.flops_forward ? fmtInt(program.flops_forward) : null} />
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <RadarChart program={program} size={220} />
              </div>
            </div>

            {/* CKA Similarity bars */}
            {(program.fp_cka_vs_transformer != null || program.fp_cka_vs_ssm != null) && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
                    CKA Similarity to Known Architectures
                  </div>
                  {(program.cka_source || program.cka_artifact_version) && (
                    <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                      {program.cka_source && (
                        <span style={{
                          fontSize: 10,
                          fontWeight: 600,
                          padding: '2px 6px',
                          borderRadius: 4,
                          background: program.cka_source === 'artifact' ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                          color: program.cka_source === 'artifact' ? 'var(--accent-green)' : 'var(--accent-red)',
                        }}>
                          {program.cka_source === 'artifact' ? 'Artifact CKA' : 'Fallback CKA'}
                        </span>
                      )}
                      {program.cka_artifact_version && (
                        <span style={{
                          fontSize: 10,
                          padding: '2px 6px',
                          borderRadius: 4,
                          background: 'var(--bg-tertiary)',
                          color: 'var(--text-muted)',
                        }}>
                          {program.cka_artifact_version}
                        </span>
                      )}
                    </div>
                  )}
                </div>
                {[
                  { label: 'Transformer', value: program.fp_cka_vs_transformer, color: 'var(--accent-blue)' },
                  { label: 'SSM', value: program.fp_cka_vs_ssm, color: 'var(--accent-green)' },
                  { label: 'Conv', value: program.fp_cka_vs_conv, color: 'var(--accent-yellow)' },
                ].map(({ label, value, color }) => value != null && (
                  <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: 12, color: 'var(--text-secondary)', minWidth: 80 }}>{label}</span>
                    <div style={{ flex: 1, height: 12, background: 'var(--bg-tertiary)', borderRadius: 3 }}>
                      <div style={{
                        width: `${Math.min(value, 1) * 100}%`, height: '100%',
                        background: color, borderRadius: 3, opacity: 0.6,
                      }} />
                    </div>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', minWidth: 40 }}>{(Number(value) * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            )}

            {/* Training metrics */}
            {program.initial_loss != null && (
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
                  Training Metrics
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 13 }}>
                  <MetricRow label="Initial Loss" value={fmt(program.initial_loss)} />
                  <MetricRow label="Min Loss" value={fmt(program.min_loss)} />
                  <MetricRow label="Steps" value={program.n_train_steps} />
                  <MetricRow label="Avg Step Time" value={fmtMs(program.avg_step_time_ms)} />
                  <MetricRow label="Mean Grad Norm" value={fmt(program.mean_grad_norm, 3)} />
                  <MetricRow label="Max Grad Norm" value={fmt(program.max_grad_norm, 3)} />
                </div>
              </div>
            )}

            {/* Training Curve */}
            {program.has_training_curve && (
              <TrainingCurve resultId={resultId} />
            )}

            {/* LLM Explanation */}
            {program.llm_explanation && (
              <div style={{
                padding: 12,
                background: 'var(--bg-tertiary)',
                borderRadius: 4,
                borderLeft: '2px solid var(--accent-purple)',
                fontSize: 13,
                color: 'var(--text-secondary)',
                fontStyle: 'italic',
              }}>
                <div style={{ fontSize: 11, color: 'var(--accent-purple)', marginBottom: 4, fontWeight: 600, fontStyle: 'normal' }}>
                  ARIA'S ANALYSIS
                </div>
                {program.llm_explanation}
              </div>
            )}

            {/* Linked Hypothesis */}
            {linkedHypothesis && (
              <HypothesisInfo hypothesis={linkedHypothesis} />
            )}

            {/* Linked Decision */}
            {linkedDecision && (
              <div style={{
                padding: 12, background: 'var(--bg-tertiary)', borderRadius: 4,
                borderLeft: `2px solid ${
                  linkedDecision.decision_type === 'go' ? 'var(--accent-green)' :
                  linkedDecision.decision_type === 'no_go' ? 'var(--accent-red)' : 'var(--accent-yellow)'
                }`,
                fontSize: 13,
              }}>
                <div style={{
                  fontSize: 11, fontWeight: 600, textTransform: 'uppercase', marginBottom: 4,
                  color: linkedDecision.decision_type === 'go' ? 'var(--accent-green)' :
                         linkedDecision.decision_type === 'no_go' ? 'var(--accent-red)' : 'var(--accent-yellow)',
                }}>
                  Decision: {linkedDecision.decision_type?.replace('_', ' ')}
                </div>
                <div>{linkedDecision.rationale}</div>
              </div>
            )}

            {/* Scale Up Button (only for S1 survivors) */}
            {program.stage1_passed && (
              <div style={{
                padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6,
                border: '1px solid var(--border)',
              }}>
                {!scaleUpOpen ? (
                  <button
                    className="start-btn"
                    onClick={() => setScaleUpOpen(true)}
                    style={{ padding: '6px 16px', fontSize: 12 }}
                  >
                    Scale Up This Architecture
                  </button>
                ) : (
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--text-secondary)' }}>
                      Scale-Up Configuration
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 8 }}>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Steps</label>
                        <input type="number" min="1000" max="50000" step="1000"
                          value={scaleUpConfig.steps}
                          onChange={e => setScaleUpConfig(c => ({ ...c, steps: parseInt(e.target.value) || 5000 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Batch Size</label>
                        <input type="number" min="4" max="16" step="1"
                          value={scaleUpConfig.batch_size}
                          onChange={e => setScaleUpConfig(c => ({ ...c, batch_size: parseInt(e.target.value) || 8 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Seq Length</label>
                        <input type="number" min="256" max="1024" step="128"
                          value={scaleUpConfig.seq_len}
                          onChange={e => setScaleUpConfig(c => ({ ...c, seq_len: parseInt(e.target.value) || 512 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button
                        className="start-btn"
                        disabled={scaleUpStarting}
                        onClick={async () => {
                          setScaleUpStarting(true);
                          try {
                            setActionError(null);
                            const res = await fetch(`${API_BASE}/api/experiments/start`, {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({
                                mode: 'scale_up',
                                result_ids: [resultId],
                                scale_up_steps: scaleUpConfig.steps,
                                scale_up_batch_size: scaleUpConfig.batch_size,
                                scale_up_seq_len: scaleUpConfig.seq_len,
                              }),
                            });
                            if (!res.ok) {
                              const err = await res.json();
                              setActionError(err.error || 'Failed to start scale-up');
                            } else {
                              setScaleUpOpen(false);
                              if (onActionComplete) onActionComplete();
                              onClose();
                            }
                          } catch (e) {
                            setActionError('Error: ' + e.message);
                          }
                          setScaleUpStarting(false);
                        }}
                        style={{ padding: '6px 16px', fontSize: 12 }}
                      >
                        {scaleUpStarting ? 'Starting...' : 'Start Scale-Up'}
                      </button>
                      <button
                        className="refresh-btn"
                        onClick={() => setScaleUpOpen(false)}
                        style={{ padding: '6px 12px', fontSize: 12 }}
                      >
                        Cancel
                      </button>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
                      Trains for {scaleUpConfig.steps} steps with batch={scaleUpConfig.batch_size}, seq={scaleUpConfig.seq_len}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Investigate / Validate actions */}
            {program.stage1_passed && (
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {(!leaderboardEntry || leaderboardEntry.tier === 'screening') && (
                  <button
                    className="start-btn"
                    disabled={actionStarting === 'investigate'}
                    onClick={async () => {
                      setActionStarting('investigate');
                      try {
                        setActionError(null);
                        const res = await fetch(`${API_BASE}/api/experiments/start`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ mode: 'investigation', result_ids: [resultId] }),
                        });
                        if (!res.ok) {
                          const err = await res.json();
                          setActionError(err.error || 'Failed to start investigation');
                        } else {
                          if (onActionComplete) onActionComplete();
                          onClose();
                        }
                      } catch (e) {
                        setActionError('Error: ' + e.message);
                      }
                      setActionStarting(null);
                    }}
                    style={{ padding: '6px 16px', fontSize: 12 }}
                    title="Deep study with multiple training programs"
                  >
                    {actionStarting === 'investigate' ? 'Starting...' : 'Investigate'}
                  </button>
                )}
                {leaderboardEntry?.tier === 'investigation' && leaderboardEntry?.investigation_passed && (
                  <button
                    className="start-btn"
                    disabled={actionStarting === 'validate'}
                    onClick={async () => {
                      setActionStarting('validate');
                      try {
                        setActionError(null);
                        const res = await fetch(`${API_BASE}/api/experiments/start`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ mode: 'validation', result_ids: [resultId] }),
                        });
                        if (!res.ok) {
                          const err = await res.json();
                          setActionError(err.error || 'Failed to start validation');
                        } else {
                          if (onActionComplete) onActionComplete();
                          onClose();
                        }
                      } catch (e) {
                        setActionError('Error: ' + e.message);
                      }
                      setActionStarting(null);
                    }}
                    style={{ padding: '6px 16px', fontSize: 12, background: 'var(--accent-purple)', borderColor: 'var(--accent-purple)' }}
                    title="Publication-grade multi-seed validation"
                  >
                    {actionStarting === 'validate' ? 'Starting...' : 'Validate'}
                  </button>
                )}
              </div>
            )}

            {/* Leaderboard training program details */}
            {leaderboardEntry?.investigation_best_training && (
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
                  Best Training Program (from Investigation)
                </div>
                <pre style={{
                  fontSize: 11, padding: 8, background: 'var(--bg-tertiary)',
                  borderRadius: 4, overflow: 'auto', maxHeight: 120,
                  color: 'var(--text-secondary)',
                }}>
                  {typeof leaderboardEntry.investigation_best_training === 'string'
                    ? leaderboardEntry.investigation_best_training
                    : JSON.stringify(leaderboardEntry.investigation_best_training, null, 2)}
                </pre>
              </div>
            )}

            <TokenMixingTaxonomy graphJson={program.graph_json_parsed} />
            <GatingDiagnostics program={program} />

            {/* Graph Viewer */}
            {program.graph_json_parsed && (
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
                  Computation Graph
                </div>
                <GraphViewer graph={program.graph_json_parsed} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default ProgramDetail;
