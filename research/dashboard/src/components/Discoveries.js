import { apiCall } from "../services/apiService";
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { scoreColor } from '../utils/format';
import { lossColor, noveltyColor, reliabilityColor } from '../utils/colors';
import { candidateScore, candidateScoreBreakdown, promotionEvidence, TIER_ORDER, TIER_COLORS, TIER_LABELS, bestLoss, percentOfReference } from '../utils/scoringEngine';

const DISCOVERIES_PREFS_KEY = 'aria_discoveries_prefs_v1';
const QUALITY_FLOOR_BEST_LOSS_MAX = 0.8;

const STATUS_LABELS = {
  screening: 'Screened',
  investigation: 'Investigation',
  validation: 'Validated',
  breakthrough: 'Breakthrough',
};

const STATUS_OPTIONS = [
  { value: 'screening', label: 'Screened' },
  { value: 'investigation', label: 'Investigating' },
  { value: 'validation', label: 'Validated' },
  { value: 'breakthrough', label: 'Breakthrough' },
];

function discoveryLossDisplay(entry) {
  if (entry?.discovery_loss_ratio != null) return Number(entry.discovery_loss_ratio);
  if (entry?.screening_loss_ratio != null) return Number(entry.screening_loss_ratio);
  if (entry?.loss_ratio != null) return Number(entry.loss_ratio);
  return null;
}

function validationLossDisplay(entry) {
  if (entry?.validation_loss_ratio != null) return Number(entry.validation_loss_ratio);
  const tier = String(entry?.tier || '').toLowerCase();
  if (tier === 'validation' || tier === 'breakthrough') {
    if (entry?.investigation_loss_ratio != null) return Number(entry.investigation_loss_ratio);
  }
  return null;
}

function finitePositiveOrNull(value) {
  if (value == null) return null;
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return null;
  return num;
}

// ── Summary Bar ────────────────────────────────────────────────────

function SummaryBar({ tierCounts }) {
  const total = tierCounts?.total_survivors || 0;
  const validated = (tierCounts?.validation || 0) + (tierCounts?.breakthrough || 0);
  const breakthroughs = tierCounts?.breakthrough || 0;

  return (
    <div style={{
      display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap',
      padding: '10px 14px', marginBottom: 12,
      background: 'var(--bg-secondary)', borderRadius: 8,
      border: '1px solid var(--border)', fontSize: 13,
    }}>
      <Stat value={total} label="unique architectures" />
      <Stat value={tierCounts?.screening || 0} label="screened" color="var(--accent-blue)" />
      <Stat value={tierCounts?.investigation || 0} label="investigation" color="var(--accent-yellow)" />
      <Stat value={validated} label="validated" color="var(--accent-purple)" />
      <Stat value={breakthroughs} label="breakthroughs" color="var(--accent-green)" />
    </div>
  );
}

function Stat({ value, label, color }) {
  return (
    <span>
      <strong style={{ fontSize: 16, color: color || 'var(--text-primary)', marginRight: 4 }}>
        {value}
      </strong>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
    </span>
  );
}

// ── Status Badge ───────────────────────────────────────────────────

function StatusBadge({ entry }) {
  const tier = entry.tier;
  const color = TIER_COLORS[tier] || 'var(--text-muted)';
  
  let label = STATUS_LABELS[tier] || tier || 'Unknown';
  
  // Refine label if phase is completed but failed
  if (tier === 'investigation' && entry.investigation_robustness != null && !entry.investigation_passed) {
    label = 'Brittle';
  } else if (tier === 'validation' && entry.validation_baseline_ratio != null && !entry.validation_passed) {
    label = 'Mediocre';
  }

  return (
    <span style={{
      padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
      color, background: `${color}22`, border: `1px solid ${color}`,
      textTransform: 'uppercase',
    }}>
      {label}
    </span>
  );
}

function StatusEditor({
  entry,
  currentValue,
  onChange,
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <select
        value={currentValue || entry.tier || 'screening'}
        onChange={(e) => onChange(e.target.value)}
        onClick={(e) => e.stopPropagation()}
        style={{
          fontSize: 11,
          padding: '2px 6px',
          borderRadius: 4,
          border: '1px solid var(--border)',
          background: 'var(--bg-tertiary)',
          color: 'var(--text-primary)',
        }}
      >
        {STATUS_OPTIONS.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </select>
    </div>
  );
}

// ── Score with hover breakdown ─────────────────────────────────────

function ScoreCell({ entry }) {
  const [show, setShow] = useState(false);
  const breakdown = candidateScoreBreakdown(entry, TIER_ORDER);
  // Keep displayed score aligned with table sorting.
  const score = (entry?._score != null)
    ? Number(entry._score)
    : (entry?.composite_score != null)
      ? Number(entry.composite_score)
      : candidateScore(entry, TIER_ORDER);

  const keyMap = {
    sLoss: { label: 'Screening Loss', color: 'var(--accent-blue)' },
    iLoss: { label: 'Investigation Loss', color: '#1f6feb' },
    loss: { label: 'Loss', color: 'var(--accent-blue)' },
    novelty: { label: 'Novelty', color: 'var(--accent-purple)' },
    vBase: { label: 'Baseline', color: 'var(--accent-green)' },
    baseline: { label: 'Baseline', color: 'var(--accent-green)' },
    robust: { label: 'Robustness', color: 'var(--accent-yellow)' },
    consistency: { label: 'Consistency', color: '#d29922' },
    tierBonus: { label: 'Tier Bonus', color: 'var(--accent-orange)' },
    throughput: { label: 'Throughput', color: 'var(--text-muted)' },
    efficiencyBonus: { label: 'Efficiency', color: '#58a6ff' },
    routingBonus: { label: 'Routing', color: '#3fb950' },
    adaptiveBonus: { label: 'Adaptive Compute', color: '#c77dff' },
    robustnessBonus: { label: 'Robustness', color: 'var(--accent-yellow)' },
    referenceDeltaBonus: { label: 'Baseline Delta', color: 'var(--accent-orange)' },
  };

  const components = Object.entries(breakdown)
    .filter(([, w]) => w > 0)
    .map(([key, weight]) => ({ key, weight, ...(keyMap[key] || { label: key, color: 'var(--border)' }) }));

  const total = components.reduce((acc, c) => acc + (Number(c.weight) || 0), 0) || 1;

  return (
    <div
      style={{ minWidth: 70, position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <div style={{ fontWeight: 600, color: scoreColor(score) }}>{Number(score).toFixed(1)}</div>
      <div style={{ display: 'flex', height: 3, borderRadius: 2, overflow: 'hidden', background: 'var(--bg-tertiary)', marginTop: 2 }}>
        {components.map(c => (
          <div key={c.key} style={{ width: `${(c.weight / total) * 100}%`, background: c.color, height: '100%' }} />
        ))}
      </div>
      {show && (
        <div style={{
          position: 'absolute', top: '100%', left: '50%', transform: 'translateX(-50%)',
          marginTop: 6, padding: '8px 10px', background: '#161b22',
          border: '1px solid var(--border)', borderRadius: 6,
          boxShadow: '0 6px 16px rgba(0,0,0,0.45)', zIndex: 1000,
          minWidth: 200, fontSize: 11, color: 'var(--text-primary)',
        }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Score Breakdown</div>
          {components.map(c => (
            <div key={c.key} style={{ marginBottom: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                <span>{c.label}</span>
                <span>{Number(c.weight).toFixed(1)}</span>
              </div>
              <div style={{ height: 3, background: 'var(--bg-tertiary)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{ width: `${(c.weight / total) * 100}%`, height: '100%', background: c.color }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Expanded Row Detail ────────────────────────────────────────────

function ExpandedDetail({
  entry,
  onInvestigate,
  onValidate,
  onQueueAdd,
  onQueueRemove,
  onDelete,
  isQueued,
  eligibility,
  statusDraft,
  onStatusDraftChange,
  onSaveStatus,
  savingStatus,
}) {
  const promotion = promotionEvidence(entry);
  const fmt = (v, d = 4) => {
    if (v == null) return '--';
    const num = Number(v);
    if (num !== 0 && Math.abs(num) < 0.0001) return num.toExponential(2);
    return num.toFixed(d);
  };

  const hasBeenInvestigated = entry.investigation_loss_ratio != null || ['investigation', 'validation', 'breakthrough'].includes(entry.tier);
  const hasBeenValidated = entry.validation_loss_ratio != null || ['validation', 'breakthrough'].includes(entry.tier);
  const canDelete = !entry.is_reference && (entry.tier === 'screening' || entry.tier === 'failed' || entry.tier === 'rejected' || entry.screening_passed === false || entry.investigation_passed === false || entry.validation_passed === false);

  return (
    <tr>
      <td colSpan={8} style={{ padding: '12px 16px', background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border)' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, fontSize: 12 }}>
          {/* Metrics detail */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 6, textTransform: 'uppercase', fontSize: 10, color: 'var(--text-muted)' }}>Full Metrics</div>
            <MetricRow label="Screening Loss" value={fmt(entry.screening_loss_ratio)} color={lossColor(entry.screening_loss_ratio)} />
            <MetricRow label="Screening Novelty" value={fmt(entry.screening_novelty, 3)} color={noveltyColor(entry.screening_novelty)} />
            <MetricRow label="Investigation Loss" value={fmt(entry.investigation_loss_ratio)} />
            <MetricRow label="Robustness" value={fmt(entry.investigation_robustness, 2)}
              color={entry.investigation_robustness != null
                ? (entry.investigation_robustness >= 0.5 ? 'var(--accent-green)' : 'var(--accent-red)')
                : undefined} />
            <MetricRow label="Validation Loss" value={fmt(entry.validation_loss_ratio)} />
            <MetricRow label="Validation Baseline" value={fmt(entry.validation_baseline_ratio)}
              color={entry.validation_baseline_ratio != null
                ? (entry.validation_baseline_ratio < 1 ? 'var(--accent-green)' : 'var(--accent-red)')
                : undefined} />
            <MetricRow label="Multi-seed Std" value={fmt(entry.validation_multi_seed_std, 3)} />
            <MetricRow label="Composite" value={fmt(entry.composite_score, 3)} color="var(--accent-green)" />
          </div>

          {/* Evidence & promotion */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 6, textTransform: 'uppercase', fontSize: 10, color: 'var(--text-muted)' }}>Evidence</div>
            <div style={{ marginBottom: 6, color: promotion.color, fontWeight: 600 }}>
              Promotion: {promotion.label} ({promotion.score}%)
            </div>
            {entry.cka_source && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                CKA: {entry.cka_source === 'artifact' ? 'artifact-backed' : 'heuristic'}
              </div>
            )}
            {entry.novelty_confidence != null && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                Novelty confidence: {Number(entry.novelty_confidence).toFixed(2)}
              </div>
            )}
            {entry.param_count != null && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                Parameters: {(entry.param_count / 1e6).toFixed(1)}M
              </div>
            )}
            {entry.graph_fingerprint && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                FP: {entry.graph_fingerprint}
              </div>
            )}
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
                Edit Status
              </div>
              <StatusEditor
                entry={entry}
                currentValue={statusDraft || entry.tier}
                onChange={(tier) => onStatusDraftChange?.(tier)}
              />
            </div>
            <div style={{ marginTop: 8 }}>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onSaveStatus?.();
                }}
                disabled={savingStatus || (statusDraft || entry.tier) === entry.tier}
                style={{
                  ...actionBtnStyle,
                  padding: '3px 9px',
                  fontSize: 10,
                  borderColor: 'var(--accent-green)',
                  color: 'var(--accent-green)',
                  opacity: (savingStatus || (statusDraft || entry.tier) === entry.tier) ? 0.6 : 1,
                  cursor: (savingStatus || (statusDraft || entry.tier) === entry.tier) ? 'not-allowed' : 'pointer',
                }}
                title="Save status change"
              >
                {savingStatus ? 'Saving…' : 'Save Status'}
              </button>
            </div>
            {entry.result_id && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                ID: {entry.result_id}
              </div>
            )}
          </div>

          {/* Actions */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 6, textTransform: 'uppercase', fontSize: 10, color: 'var(--text-muted)' }}>Actions</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {!hasBeenInvestigated && (
                <button
                  onClick={() => onInvestigate([entry.result_id])}
                  style={{
                    ...actionBtnStyle,
                    background: 'rgba(63, 185, 80, 0.12)',
                    border: '1px solid rgba(63, 185, 80, 0.4)',
                    color: 'var(--accent-green)',
                    opacity: eligibility?.investigationEligible ? 1 : 0.85,
                  }}
                  title={eligibility?.investigationEligible ? 'Start investigation' : 'Currently ineligible; click to force override'}
                >
                  {eligibility?.investigationEligible ? 'Investigate' : 'Force Investigate'}
                </button>
              )}
              {!hasBeenValidated && (
                <button
                  onClick={() => onValidate([entry.result_id])}
                  style={{
                    ...actionBtnStyle,
                    background: 'rgba(188, 140, 255, 0.12)',
                    border: '1px solid rgba(188, 140, 255, 0.4)',
                    color: 'var(--accent-purple)',
                    opacity: eligibility?.validationEligible ? 1 : 0.85,
                  }}
                  title={eligibility?.validationEligible ? 'Start validation' : 'Currently ineligible; click to force override'}
                >
                  {eligibility?.validationEligible ? 'Validate' : 'Force Validate'}
                </button>
              )}
              {entry.result_id && (onQueueAdd || onQueueRemove) && (
                <button
                  onClick={() => {
                    if (isQueued) {
                      onQueueRemove?.(entry.result_id);
                    } else if (eligibility?.queueEligible) {
                      onQueueAdd?.({
                        resultId: entry.result_id,
                        fingerprint: entry.graph_fingerprint,
                        source: 'discoveries',
                        architectureFamily: entry.architecture_family,
                        intent: eligibility?.validationEligible ? 'validation' : 'investigation',
                        queueEligible: true,
                        investigationEligible: eligibility?.investigationEligible,
                        validationEligible: eligibility?.validationEligible,
                      });
                    }
                  }}
                  disabled={!isQueued && !eligibility?.queueEligible}
                  style={{
                    ...actionBtnStyle,
                    borderColor: isQueued ? 'var(--accent-yellow)' : 'var(--accent-blue)',
                    color: isQueued ? 'var(--accent-yellow)' : 'var(--accent-blue)',
                    opacity: !isQueued && !eligibility?.queueEligible ? 0.5 : 1,
                  }}
                >
                  {isQueued 
                    ? 'Queued' 
                    : (!eligibility?.queueEligible && (entry.tier === 'validation' || entry.tier === 'breakthrough'))
                      ? 'Fully Validated'
                      : !eligibility?.queueEligible 
                        ? 'Ineligible'
                        : 'Add to Queue'}
                </button>
              )}
              {canDelete && (
                <button
                  onClick={() => {
                    if (window.confirm(`Delete ${entry.entry_id?.slice(0, 11) || entry.result_id?.slice(0, 11)} and all associated data?`)) {
                      onDelete?.(entry.entry_id || entry.result_id);
                    }
                  }}
                  style={{
                    ...actionBtnStyle,
                    borderColor: 'rgba(248, 81, 73, 0.4)',
                    background: 'rgba(248, 81, 73, 0.12)',
                    color: 'var(--accent-red, #f85149)',
                  }}
                  title="Delete entry and all associated data"
                >
                  Delete
                </button>
              )}
            </div>
          </div>
        </div>
      </td>
    </tr>
  );
}

function MetricRow({ label, value, color }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ color: color || 'var(--text-primary)', fontFamily: 'monospace' }}>{value}</span>
    </div>
  );
}

// ── Fingerprint Leaderboard Chart ─────────────────────────────────

function FingerprintLeaderboardChart({ entries }) {
  if (!entries || entries.length < 2) return null;
  
  // Take top 15 for the chart
  const top = entries.slice(0, 15);
  const W = 600;
  const H = 160;
  const PAD_X = 40;
  const PAD_Y = 20;
  const barW = (W - 2 * PAD_X) / top.length - 8;
  
  const maxScore = Math.max(...top.map(e => e._score), 80);
  
  return (
    <div style={{ marginBottom: 20, padding: '10px 0' }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', fontWeight: 600 }}>
        Fingerprint Performance Ranking (Top {top.length})
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', maxWidth: W }}>
        {/* Y-axis labels */}
        {[0, 0.5, 1].map(frac => (
          <text key={frac} x={PAD_X - 5} y={H - PAD_Y - frac * (H - 2 * PAD_Y)} 
                fontSize={9} fill="var(--text-muted)" textAnchor="end" alignmentBaseline="middle">
            {Math.round(frac * maxScore)}
          </text>
        ))}
        
        {/* Horizontal grid lines */}
        {[0, 0.5, 1].map(frac => (
          <line key={`grid-${frac}`} x1={PAD_X} y1={H - PAD_Y - frac * (H - 2 * PAD_Y)} 
                x2={W - PAD_X} y2={H - PAD_Y - frac * (H - 2 * PAD_Y)} 
                stroke="var(--border)" strokeWidth={0.5} strokeDasharray="2 2" />
        ))}

        {top.map((e, i) => {
          const score = e._score || 0;
          const barH = (score / maxScore) * (H - 2 * PAD_Y);
          const x = PAD_X + i * (barW + 8);
          const y = H - PAD_Y - barH;
          const isPinnedReference = Boolean(e?.is_reference)
            || String(e?.model_source || '').toLowerCase() === 'reference'
            || Boolean(e?.reference_name);
          const color = isPinnedReference ? 'var(--accent-purple)' : scoreColor(score);
          
          return (
            <g key={e.result_id || i}>
              <rect x={x} y={y} width={barW} height={barH} fill={`${color}88`} stroke={color} strokeWidth={1} rx={2} />
              <text x={x + barW / 2} y={H - 5} fontSize={8} fill="var(--text-muted)" 
                    textAnchor="middle" transform={`rotate(45 ${x + barW / 2} ${H - 5})`}>
                {e.display_name?.slice(0, 8) || e.graph_fingerprint?.slice(0, 6)}
              </text>
              <title>{e.display_name || e.graph_fingerprint}: Discovery Score {score}</title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────

const COLUMNS = [
  { key: '_score', label: 'Discovery Score', title: 'Internal ranking score based on novelty and performance.' },
  { key: 'display_name', label: 'Architecture', title: 'Human-readable name or fingerprint of the model topology.' },
  { key: 'architecture_family', label: 'Family', title: 'The architectural category (e.g., Attention, SSM, Hybrid).' },
  { key: 'discovery_loss_ratio', label: 'Discovery Loss', title: 'Loss ratio on random tokens (fast triage).' },
  { key: 'validation_loss_ratio', label: 'Validation Loss', title: 'Loss ratio on real micro-corpus (true causal performance).' },
  { key: '_best_loss', label: 'Best Loss', title: 'The lowest loss ratio achieved by this architecture across all tests.' },
  { key: '_vs_ref', label: 'vs Ref', title: 'How this model compares to the GPT-2 baseline (lower % is better).' },
  { key: '_novelty', label: 'Novelty', title: 'Measures how unique this model is compared to existing designs.' },
  { key: 'param_efficiency', label: 'Param Eff', title: 'Parameter efficiency: FLOPs per parameter (higher = parameters are used more efficiently).' },
  { key: 'sample_efficiency', label: 'Sample Eff', title: 'How quickly the model converges to 25% of initial loss (1.0 = instant, 0.0 = never).' },
  { key: 'investigation_robustness', label: 'Robustness', title: 'Consistency across different training recipes (higher is more stable).' },
  { key: 'robustness_long_ctx_score', label: 'LongCtx', title: 'Combined long-context score used in final evaluation.' },
  { key: 'robustness_long_ctx_scaling_score', label: 'LC-Scale', title: 'Long-context scaling component score.' },
  { key: 'robustness_long_ctx_assoc_score', label: 'LC-Assoc', title: 'Associative retrieval benchmark score.' },
  { key: 'robustness_long_ctx_multi_hop_score', label: 'LC-MHop', title: 'Multi-hop retrieval benchmark score.' },
  { key: 'robustness_long_ctx_passkey_score', label: 'LC-Passkey', title: 'Zero-shot passkey retrieval benchmark score.' },
  { key: 'robustness_long_ctx_retrieval_aggregate', label: 'LC-Retr', title: 'Aggregate retrieval score across long-context benchmarks.' },
  { key: 'max_viable_seq_len', label: 'LC-MaxLen', title: 'Maximum viable sequence length from long-context scaling sweep.' },
  { key: 'jacobian_spectral_norm', label: 'Spectral', title: 'Jacobian Spectral Norm: stability of gradient propagation (lower is better).' },
  { key: 'init_sensitivity_std', label: 'InitStd', title: 'Sensitivity to weight initialization (lower means more predictable).' },
  { key: 'tier', label: 'Status', width: 96, title: 'Current research phase of this architecture.' },
  { key: '_actions', label: '', title: 'Actions' },
];

function Discoveries({
  onSelectProgram,
  onInvestigate,
  onValidate,
  highlightResultId,
  onHighlightClear,
  onQueueAdd,
  onQueueRemove,
  queuedResultIds,
  eligibilityByResultId,
  onOpenInDesigner,
}) {
  const isPinnedReferenceRow = useCallback((entry) => (
    Boolean(entry?.is_reference)
    || String(entry?.model_source || '').toLowerCase() === 'reference'
    || Boolean(entry?.reference_name)
  ), []);

  const prefs = (() => {
    try {
      if (typeof window === 'undefined') return {};
      const stored = window.localStorage.getItem(DISCOVERIES_PREFS_KEY);
      return stored ? JSON.parse(stored) : {};
    } catch { return {}; }
  })();

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTier, setActiveTier] = useState(() =>
    ['all', 'screening', 'investigation', 'validation', 'breakthrough'].includes(prefs?.activeTier) ? prefs.activeTier : 'all'
  );
  const [sortKey, setSortKey] = useState(() => typeof prefs?.sortKey === 'string' ? prefs.sortKey : '_score');
  const [sortDesc, setSortDesc] = useState(() => typeof prefs?.sortDesc === 'boolean' ? prefs.sortDesc : true);
  const [searchQuery, setSearchQuery] = useState(() => typeof prefs?.searchQuery === 'string' ? prefs.searchQuery : '');
  const [expandedRowId, setExpandedRowId] = useState(null);
  const [highlightId, setHighlightId] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [statusDrafts, setStatusDrafts] = useState({});
  const [savingStatusRowId, setSavingStatusRowId] = useState(null);
  const [statusError, setStatusError] = useState(null);
  const [showChart, setShowChart] = useState(true);
  const [showReferences, setShowReferences] = useState(() =>
    typeof prefs?.showReferences === 'boolean' ? prefs.showReferences : true
  );
  const [qualityFloorEnabled, setQualityFloorEnabled] = useState(() =>
    typeof prefs?.qualityFloorEnabled === 'boolean' ? prefs.qualityFloorEnabled : true
  );
  const [visibleColumns, setVisibleColumns] = useState(() =>
    {
      const requiredLongCtx = [
        'robustness_long_ctx_score',
        'robustness_long_ctx_scaling_score',
        'robustness_long_ctx_assoc_score',
        'robustness_long_ctx_multi_hop_score',
        'robustness_long_ctx_passkey_score',
        'robustness_long_ctx_retrieval_aggregate',
        'max_viable_seq_len',
      ];
      const validKeys = new Set(COLUMNS.map(c => c.key));
      const saved = Array.isArray(prefs?.visibleColumns)
        ? prefs.visibleColumns.filter((key) => validKeys.has(key))
        : null;

      // Respect saved user choices exactly. Only inject long-context defaults
      // for first-time users with no saved column preferences.
      if (saved && saved.length > 0) {
        return saved;
      }

      const defaults = [...COLUMNS.map(c => c.key)];
      for (const key of requiredLongCtx) {
        if (!defaults.includes(key) && validKeys.has(key)) defaults.push(key);
      }
      return defaults;
    }
  );
  const [showColumnPicker, setShowColumnPicker] = useState(false);
  const queuedSet = useMemo(() => new Set(queuedResultIds || []), [queuedResultIds]);
  const highlightRef = useRef(null);

  // Persist preferences
  useEffect(() => {
    try {
      if (typeof window === 'undefined') return;
      window.localStorage.setItem(DISCOVERIES_PREFS_KEY, JSON.stringify({
        activeTier, sortKey, sortDesc, searchQuery, showChart, showReferences,
        qualityFloorEnabled, visibleColumns,
      }));
    } catch {}
  }, [activeTier, sortKey, sortDesc, searchQuery, showChart, showReferences, qualityFloorEnabled, visibleColumns]);

  // Handle external highlight
  useEffect(() => {
    if (highlightResultId) {
      setHighlightId(highlightResultId);
      const timer = setTimeout(() => {
        setHighlightId(null);
        onHighlightClear?.();
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [highlightResultId, onHighlightClear]);

  // Scroll to highlighted row
  useEffect(() => {
    if (highlightId && highlightRef.current) {
      highlightRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [highlightId]);

  const lastDataRef = useRef(null);

  const fetchData = useCallback(async (isBackground = false) => {
    if (!isBackground) {
      setLoading(true);
      setError(null);
    }
    try {
      const params = new URLSearchParams({ sort: 'composite_score', limit: '200', view: 'ranked' });
      if (activeTier !== 'all') params.set('tier', activeTier);
      const res = await apiCall(`/api/discoveries?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      // Only update state if data actually changed — prevents scroll reset
      const jsonStr = JSON.stringify(json);
      if (jsonStr !== lastDataRef.current) {
        lastDataRef.current = jsonStr;
        setData(json);
        setLastUpdated(new Date());
      }
      setError(null);
    } catch (e) {
      if (!isBackground) setError('Failed to load discoveries: ' + e.message);
    }
    if (!isBackground) setLoading(false);
  }, [activeTier]);

  useEffect(() => {
    fetchData(false);
    const interval = setInterval(() => fetchData(true), 60000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleSort = (key) => {
    if (key === '_actions') return;
    if (sortKey === key) setSortDesc(!sortDesc);
    else { setSortKey(key); setSortDesc(true); }
  };

  const handleStatusDraftChange = useCallback((rowId, tier) => {
    setStatusDrafts(prev => ({ ...prev, [rowId]: tier }));
  }, []);

  const handleSaveStatus = useCallback(async (entry) => {
    const rowId = entry.entry_id || entry.result_id;
    if (!rowId) return;
    const nextTier = statusDrafts[rowId] || entry.tier;
    if (!nextTier || nextTier === entry.tier) return;

    setSavingStatusRowId(rowId);
    setStatusError(null);
    try {
      const res = await apiCall(`/api/leaderboard/status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          entry_id: entry.entry_id,
          result_id: entry.result_id,
          tier: nextTier,
        }),
      });
      if (!res.ok) {
        let payload = null;
        try { payload = await res.json(); } catch {}
        throw new Error(payload?.error || `HTTP ${res.status}`);
      }

      setData(prev => {
        if (!prev?.entries) return prev;
        return {
          ...prev,
          entries: prev.entries.map(item => (
            (item.entry_id && item.entry_id === entry.entry_id)
              || (item.result_id && item.result_id === entry.result_id)
              ? { ...item, tier: nextTier }
              : item
          )),
        };
      });
    } catch (e) {
      setStatusError(`Failed to save status: ${e.message}`);
    } finally {
      setSavingStatusRowId(null);
    }
  }, [statusDrafts]);

  const handleDelete = useCallback(async (entryId) => {
    try {
      const res = await apiCall(`/api/leaderboard/${entryId}`, { method: 'DELETE' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setStatusError(`Delete failed: ${err.error || res.statusText}`);
        return;
      }
      fetchData();
    } catch (e) {
      setStatusError('Delete failed: ' + e.message);
    }
  }, [fetchData]);

  // Sort & augment entries
  const sorted = useMemo(() => {
    const entries = data?.entries || [];
    const refs = entries.filter(e => e.is_reference);
    
    const augmented = entries.map(e => {
      const entryBestLoss = bestLoss(e);
      
      // Find best reference in same family or paradigm for comparison
      let vsRef = null;
      if (entryBestLoss != null && !e.is_reference) {
        // 1. Try same family
        let bestRefLoss = null;
        const familyRefs = refs.filter(r => r.architecture_family === e.architecture_family && bestLoss(r) != null);
        if (familyRefs.length > 0) {
          bestRefLoss = Math.min(...familyRefs.map(r => bestLoss(r)));
        } else {
          // 2. Fallback to GPT-2 Small as the "universal" baseline
          const gpt2 = refs.find(r => r.reference_name === 'GPT-2 Small' || r.reference_name === 'GPT-2');
          bestRefLoss = bestLoss(gpt2);
        }
        
        if (bestRefLoss != null) {
          vsRef = percentOfReference(entryBestLoss, bestRefLoss);
        }
      }

      return {
        ...e,
        // Keep Discoveries score aligned with backend leaderboard composite when present.
        _score: (e.composite_score != null ? Number(e.composite_score) : candidateScore(e, TIER_ORDER)),
        _best_loss: entryBestLoss,
        _vs_ref: vsRef,
        _novelty: e.screening_novelty ?? e.novelty_score ?? null,
      };
    });
    augmented.sort((a, b) => {
      // ONLY prioritize references at the top if we are sorting by score (the default)
      if (sortKey === '_score') {
        const aRef = Number(Boolean(a?.is_reference));
        const bRef = Number(Boolean(b?.is_reference));
        if (aRef !== bRef) return bRef - aRef;
      }

      let va, vb;
      if (sortKey === 'tier') {
        va = TIER_ORDER[a.tier] || 0;
        vb = TIER_ORDER[b.tier] || 0;
      } else if (sortKey === 'display_name' || sortKey === 'architecture_family') {
        va = a[sortKey] || '';
        vb = b[sortKey] || '';
        return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      } else {
        va = a[sortKey]; vb = b[sortKey];
      }
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return sortDesc ? vb - va : va - vb;
    });
    return augmented;
  }, [data?.entries, sortKey, sortDesc]);

  const visibilityFiltered = useMemo(() => {
    if (showReferences) return sorted;
    return sorted.filter(e => !e?.is_reference);
  }, [sorted, showReferences]);

  const qualityFiltered = useMemo(() => {
    if (!qualityFloorEnabled) return visibilityFiltered;
    return visibilityFiltered.filter((e) => {
      if (e?.is_reference) return true;
      const best = bestLoss(e);
      return best != null && Number(best) <= QUALITY_FLOOR_BEST_LOSS_MAX;
    });
  }, [visibilityFiltered, qualityFloorEnabled]);

  const qualityHiddenCount = useMemo(() => {
    if (!qualityFloorEnabled) return 0;
    return Math.max(0, (visibilityFiltered?.length || 0) - (qualityFiltered?.length || 0));
  }, [qualityFloorEnabled, visibilityFiltered, qualityFiltered]);

  // Search filter
  const filtered = useMemo(() => {
    if (!searchQuery.trim()) return qualityFiltered;
    const q = searchQuery.trim().toLowerCase();
    return qualityFiltered.filter(e =>
      (e.display_name && e.display_name.toLowerCase().includes(q)) ||
      (e.architecture_family && e.architecture_family.toLowerCase().includes(q)) ||
      (e.graph_fingerprint && e.graph_fingerprint.toLowerCase().includes(q)) ||
      (e.result_id && e.result_id.toLowerCase().includes(q)) ||
      (e.architecture_desc && e.architecture_desc.toLowerCase().includes(q))
    );
  }, [qualityFiltered, searchQuery]);

  const tierCounts = data?.tier_counts || {};
  const tiers = ['all', 'screening', 'investigation', 'validation', 'breakthrough'];

  return (
    <div className="card" style={{ padding: 16 }}>
      <div className="card-title" style={{ marginBottom: 8 }}>
        Discoveries
        <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 8 }}>
          {lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString()}` : 'Loading...'}
        </span>
      </div>

      {/* Summary bar */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start', marginBottom: 12 }}>
        <div style={{ flex: 1 }}>
          <SummaryBar tierCounts={tierCounts} />
        </div>
        <button
          className={`refresh-btn ${showChart ? 'active' : ''}`}
          style={{ padding: '8px 12px', fontSize: 12 }}
          onClick={() => setShowChart(!showChart)}
          title={showChart ? 'Hide performance chart' : 'Show performance chart'}
        >
          {showChart ? 'Hide Chart' : 'Show Chart'}
        </button>
      </div>

      {showChart && filtered.length > 0 && (
        <FingerprintLeaderboardChart entries={filtered} />
      )}

      {/* Tier filter tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 12, flexWrap: 'wrap' }}>
        {tiers.map(tier => {
          const count = tier === 'all'
            ? (data?.total || 0)
            : (tierCounts[tier] || 0);
          return (
            <button
              key={tier}
              onClick={() => setActiveTier(tier)}
              aria-label={`Filter by ${tier === 'all' ? 'all tiers' : `${TIER_LABELS[tier]} tier`}`}
              style={{
                padding: '5px 14px', borderRadius: 4,
                border: `1px solid ${activeTier === tier ? 'var(--accent-blue)' : 'var(--border)'}`,
                background: activeTier === tier ? 'rgba(88, 166, 255, 0.15)' : 'transparent',
                color: activeTier === tier ? 'var(--accent-blue)' : 'var(--text-secondary)',
                cursor: 'pointer', fontSize: 12, fontWeight: activeTier === tier ? 600 : 400,
              }}
            >
              {tier === 'all' ? 'All' : TIER_LABELS[tier]}
              {count > 0 && (
                <span style={{
                  marginLeft: 5, fontSize: 10,
                  color: tier === 'all' ? 'var(--text-muted)' : (TIER_COLORS[tier] || 'var(--text-muted)'),
                }}>
                  ({count})
                </span>
              )}
            </button>
          );
        })}
        <button
          onClick={() => setShowReferences(v => !v)}
          aria-label={showReferences ? 'Hide references' : 'Show references'}
          style={{
            fontSize: 11, padding: '5px 12px', cursor: 'pointer',
            background: showReferences ? 'rgba(188, 140, 255, 0.12)' : 'transparent',
            border: `1px solid ${showReferences ? 'var(--accent-purple)' : 'var(--border)'}`,
            borderRadius: 4, color: showReferences ? 'var(--accent-purple)' : 'var(--text-secondary)',
          }}
        >
          {showReferences ? 'Hide references' : 'Show references'}
        </button>
        <button
          onClick={() => setQualityFloorEnabled(v => !v)}
          aria-label={qualityFloorEnabled ? 'Disable quality floor' : 'Enable quality floor'}
          style={{
            fontSize: 11, padding: '5px 12px', cursor: 'pointer',
            background: qualityFloorEnabled ? 'rgba(63, 185, 80, 0.14)' : 'transparent',
            border: `1px solid ${qualityFloorEnabled ? 'var(--accent-green)' : 'var(--border)'}`,
            borderRadius: 4, color: qualityFloorEnabled ? 'var(--accent-green)' : 'var(--text-secondary)',
          }}
          title={`Hide low-quality entries where best loss > ${QUALITY_FLOOR_BEST_LOSS_MAX.toFixed(1)}`}
        >
          {qualityFloorEnabled ? `Quality floor ≤ ${QUALITY_FLOOR_BEST_LOSS_MAX.toFixed(1)}` : 'Show all quality'}
        </button>
        <button
          onClick={() => setShowColumnPicker(!showColumnPicker)}
          style={{
            fontSize: 11, padding: '5px 12px', cursor: 'pointer',
            border: `1px solid ${showColumnPicker ? 'var(--accent-blue)' : 'var(--border)'}`, 
            borderRadius: 4,
            background: showColumnPicker ? 'rgba(88, 166, 255, 0.12)' : 'transparent', 
            color: showColumnPicker ? 'var(--accent-blue)' : 'var(--text-secondary)',
          }}
        >
          Columns
        </button>
        <button
          onClick={fetchData}
          disabled={loading}
          aria-label="Refresh discoveries"
          style={{ marginLeft: 'auto', fontSize: 11, padding: '5px 12px', cursor: loading ? 'not-allowed' : 'pointer', background: 'transparent', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-secondary)', opacity: loading ? 0.6 : 1 }}
        >
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {showColumnPicker && (
        <div style={{
          marginBottom: 12, padding: 12, background: 'var(--bg-secondary)', 
          border: '1px solid var(--border)', borderRadius: 6,
          display: 'flex', gap: 12, flexWrap: 'wrap'
        }}>
          {COLUMNS.map(col => (
            <label key={col.key} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-primary)', cursor: 'pointer' }}>
              <input 
                type="checkbox" 
                checked={visibleColumns.includes(col.key)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setVisibleColumns([...visibleColumns, col.key]);
                  } else {
                    setVisibleColumns(visibleColumns.filter(k => k !== col.key));
                  }
                }}
              />
              {col.label}
            </label>
          ))}
        </div>
      )}

      {/* Search */}
      <div style={{ marginBottom: 12 }}>
        <input
          type="text"
          placeholder="Search by name, family, fingerprint, or ID..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          aria-label="Search discoveries"
          style={{
            width: '100%', maxWidth: 400, padding: '6px 10px', fontSize: 12,
            border: '1px solid var(--border)', borderRadius: 4,
            background: 'var(--bg-secondary)', color: 'var(--text-primary)',
          }}
        />
        {searchQuery && (
          <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--text-muted)' }}>
            {filtered.length} of {qualityFiltered.length} entries
          </span>
        )}
        {qualityFloorEnabled && qualityHiddenCount > 0 && (
          <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--accent-yellow)' }}>
            {qualityHiddenCount} low-quality hidden
          </span>
        )}
      </div>

      {/* Reference Baselines Banner */}
      {showReferences && sorted.filter(e => e.is_reference).length > 0 && (
        <div style={{
          marginBottom: 14, padding: '10px 14px',
          background: 'rgba(188, 140, 255, 0.06)',
          border: '1px solid rgba(188, 140, 255, 0.25)',
          borderRadius: 6,
        }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent-purple)', marginBottom: 8 }}>
            Reference Baselines
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {sorted.filter(e => e.is_reference).map(ref => (
              <div key={ref.entry_id || ref.result_id} style={{
                padding: '6px 12px', borderRadius: 5,
                background: 'rgba(188, 140, 255, 0.10)',
                border: '1px solid rgba(188, 140, 255, 0.18)',
                fontSize: 11, lineHeight: 1.5, minWidth: 130,
              }}>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                  {ref.reference_name || ref.display_name || ref.architecture_desc || 'Reference'}
                </div>
                <div style={{ color: 'var(--text-muted)' }}>
                  {ref.architecture_family || '--'}
                  {ref._best_loss != null && <span style={{ marginLeft: 8 }}>Loss: {ref._best_loss.toFixed(4)}</span>}
                </div>
                {ref.param_count != null && (
                  <div style={{ color: 'var(--text-muted)' }}>
                    {(ref.param_count / 1e6).toFixed(1)}M params
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {error && <p style={{ color: 'var(--accent-red)', fontSize: 13, marginBottom: 8 }}>{error}</p>}
      {statusError && <p style={{ color: 'var(--accent-red)', fontSize: 12, marginBottom: 8 }}>{statusError}</p>}

      {loading ? (
        <p style={{ color: 'var(--text-muted)' }}>Loading discoveries...</p>
      ) : filtered.length === 0 && !error ? (
        <div style={{ color: 'var(--text-muted)', fontSize: 13, lineHeight: 1.6 }}>
          {searchQuery.trim() ? (
            <p>No entries match "{searchQuery}".</p>
          ) : activeTier === 'all' ? (
            <p>No discoveries yet. Run experiments to generate candidates.</p>
          ) : (
            <p>No entries in {TIER_LABELS[activeTier]} tier yet.</p>
          )}
        </div>
      ) : (
        <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: 'calc(100vh - 280px)' }}>
          <table className="data-table table-wide">
            <thead style={{ position: 'sticky', top: 0, zIndex: 2, background: 'var(--bg-card, #1a1a2e)' }}>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                <th style={{ ...thStyle, width: 26, position: 'sticky', top: 0, background: 'inherit' }} aria-label="Pinned marker" />
                <th style={{ ...thStyle, position: 'sticky', top: 0, background: 'inherit' }}>#</th>
                {COLUMNS.filter(col => visibleColumns.includes(col.key)).map(col => (
                  <th
                    key={col.key}
                    onClick={() => handleSort(col.key)}
                    title={col.title}
                    style={{
                      ...thStyle,
                      position: 'sticky',
                      top: 0,
                      background: 'inherit',
                      width: col.width ? `${col.width}px` : undefined,
                      cursor: col.key === '_actions' ? 'default' : 'pointer',
                      userSelect: 'none',
                    }}
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
              {filtered.map((entry, i) => {
                const rowId = entry.entry_id || entry.result_id || i;
                const isExpanded = expandedRowId === rowId;
                const isHighlighted = highlightId && entry.result_id === highlightId;
                const isQueued = !!entry.result_id && queuedSet.has(entry.result_id);
                const isPinnedReference = isPinnedReferenceRow(entry);
                const eligibility = eligibilityByResultId?.[entry.result_id] || null;
                const displayName = entry.display_name || entry.architecture_desc || entry.graph_fingerprint?.slice(0, 10) || '--';
                return (
                  <DiscoveryRow 
                    key={rowId}
                    entry={entry}
                    i={i}
                    rowId={rowId}
                    isExpanded={isExpanded}
                    isHighlighted={isHighlighted}
                    isQueued={isQueued}
                    isPinnedReference={isPinnedReference}
                    eligibility={eligibility}
                    displayName={displayName}
                    highlightRef={highlightRef}
                    onSelectProgram={onSelectProgram}
                    tdStyle={tdStyle}
                    COLUMNS={COLUMNS}
                    visibleColumns={visibleColumns}
                    onOpenInDesigner={onOpenInDesigner}
                    setExpandedRowId={setExpandedRowId}
                    actionBtnStyle={actionBtnStyle}
                    handleDelete={handleDelete}
                    onInvestigate={onInvestigate}
                    onValidate={onValidate}
                    onQueueAdd={onQueueAdd}
                    onQueueRemove={onQueueRemove}
                    statusDrafts={statusDrafts}
                    handleStatusDraftChange={handleStatusDraftChange}
                    handleSaveStatus={handleSaveStatus}
                    savingStatusRowId={savingStatusRowId}
                  />
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const thStyle = {
  padding: '6px 8px', textAlign: 'left', fontSize: 11,
  color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', whiteSpace: 'nowrap',
};

const tdStyle = {
  padding: '6px 8px', whiteSpace: 'nowrap',
};

const actionBtnStyle = {
  padding: '4px 10px', fontSize: 11,
  border: '1px solid rgba(88, 166, 255, 0.4)', borderRadius: 4,
  background: 'rgba(88, 166, 255, 0.12)', color: 'var(--accent-blue)', cursor: 'pointer',
};


const DiscoveryRow = React.memo(function DiscoveryRow({
  entry, 
  i, 
  rowId, 
  isExpanded, 
  isHighlighted, 
  isQueued, 
  isPinnedReference, 
  eligibility, 
  displayName,
  highlightRef,
  onSelectProgram,
  tdStyle,
  COLUMNS,
  visibleColumns,
  onOpenInDesigner,
  setExpandedRowId,
  actionBtnStyle,
  handleDelete,
  onInvestigate,
  onValidate,
  onQueueAdd,
  onQueueRemove,
  statusDrafts,
  handleStatusDraftChange,
  handleSaveStatus,
  savingStatusRowId
}) {
  const canDelete = !entry.is_reference && (entry.tier === 'screening' || entry.tier === 'failed' || entry.tier === 'rejected' || entry.screening_passed === false || entry.investigation_passed === false || entry.validation_passed === false);

  return (
    <React.Fragment>
      <tr
        ref={isHighlighted ? highlightRef : undefined}
        style={{
          borderBottom: '1px solid var(--border)',
          cursor: 'pointer',
          background: isHighlighted
            ? 'rgba(88, 166, 255, 0.2)'
            : isPinnedReference
              ? 'rgba(188, 140, 255, 0.14)'
              : entry.tier === 'breakthrough' ? 'rgba(63, 185, 80, 0.08)' : undefined,
          animation: isHighlighted ? 'leaderboard-pulse 1.5s ease-in-out 2' : undefined,
        }}
        onClick={() => onSelectProgram?.(entry.result_id)}
      >
        <td style={{ ...tdStyle, width: 26, textAlign: 'center', paddingLeft: 4, paddingRight: 4 }}>
          {isPinnedReference ? (
            <span title="Pinned reference" style={{ color: 'var(--accent-purple)', fontSize: 12, fontWeight: 700 }}>
              ★
            </span>
          ) : null}
        </td>
        <td style={tdStyle}>{i + 1}</td>
        {COLUMNS.filter(col => visibleColumns.includes(col.key)).map(col => {
          switch (col.key) {
            case '_score':
              return <td key={col.key} style={tdStyle}><ScoreCell entry={entry} /></td>;
            case 'display_name':
              return (
                <td key={col.key} style={{ ...tdStyle, maxWidth: 200 }}>
                  <div style={{ fontWeight: 500 }}>{displayName}</div>
                  {entry.graph_fingerprint && (
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                      {entry.graph_fingerprint.slice(0, 12)}
                    </div>
                  )}
                </td>
              );
            case 'architecture_family':
              return (
                <td key={col.key} style={tdStyle}>
                  <span style={{
                    fontSize: 11, padding: '1px 6px', borderRadius: 3,
                    background: 'var(--bg-tertiary)', color: 'var(--text-secondary)',
                  }}>
                    {entry.architecture_family || '--'}
                  </span>
                </td>
              );
            case 'discovery_loss_ratio':
              const discoveryDisplay = discoveryLossDisplay(entry);
              return (
                <td key={col.key} style={{ ...tdStyle, color: lossColor(discoveryDisplay), fontFamily: 'monospace' }}>
                  {discoveryDisplay != null ? Number(discoveryDisplay).toFixed(4) : '--'}
                </td>
              );
            case 'validation_loss_ratio':
              const validationDisplay = validationLossDisplay(entry);
              return (
                <td key={col.key} style={{ ...tdStyle, color: lossColor(validationDisplay), fontFamily: 'monospace' }}>
                  {validationDisplay != null ? Number(validationDisplay).toFixed(4) : '--'}
                </td>
              );
            case '_best_loss':
              return (
                <td key={col.key} style={{ ...tdStyle, color: lossColor(entry._best_loss), fontFamily: 'monospace' }}>
                  {entry._best_loss != null ? (Number(entry._best_loss) !== 0 && Math.abs(Number(entry._best_loss)) < 0.0001 ? Number(entry._best_loss).toExponential(2) : Number(entry._best_loss).toFixed(4)) : '--'}
                </td>
              );
            case '_vs_ref':
              return (
                <td key={col.key} style={{ ...tdStyle, fontFamily: 'monospace', color: entry._vs_ref != null ? (entry._vs_ref <= 100 ? 'var(--accent-green)' : 'var(--accent-red)') : 'var(--text-muted)' }}>
                  {entry._vs_ref != null ? `${entry._vs_ref.toFixed(1)}%` : '--'}
                </td>
              );
            case '_novelty':
              return (
                <td key={col.key} style={{ ...tdStyle, color: noveltyColor(entry._novelty), fontFamily: 'monospace' }}>
                  {entry._novelty != null ? Number(entry._novelty).toFixed(3) : '--'}
                </td>
              );
            case 'param_efficiency':
              return (
                <td key={col.key} style={tdStyle}>{entry.param_efficiency != null ? Number(entry.param_efficiency).toFixed(3) : '--'}</td>
              );
            case 'robustness_long_ctx_best_score':
              return <td key={col.key} style={tdStyle}>{entry.robustness_long_ctx_best_score != null ? Number(entry.robustness_long_ctx_best_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_multi_hop_score':
              return <td key={col.key} style={tdStyle}>{entry.robustness_long_ctx_multi_hop_score != null ? Number(entry.robustness_long_ctx_multi_hop_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_passkey_score':
              return <td key={col.key} style={tdStyle}>{entry.robustness_long_ctx_passkey_score != null ? Number(entry.robustness_long_ctx_passkey_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_retrieval_aggregate':
              return <td key={col.key} style={tdStyle}>{entry.robustness_long_ctx_retrieval_aggregate != null ? Number(entry.robustness_long_ctx_retrieval_aggregate).toFixed(3) : '--'}</td>;
            case 'max_viable_seq_len':
              return <td key={col.key} style={tdStyle}>{entry.max_viable_seq_len != null ? Number(entry.max_viable_seq_len).toFixed(0) : '--'}</td>;
            case 'jacobian_spectral_norm':
              const specVal = finitePositiveOrNull(entry.jacobian_spectral_norm ?? entry.fp_jacobian_spectral_norm);
              return <td key={col.key} style={tdStyle}>{specVal != null ? Number(specVal).toFixed(4) : '--'}</td>;
            case 'init_sensitivity_std':
              return <td key={col.key} style={tdStyle}>{entry.init_sensitivity_std != null ? Number(entry.init_sensitivity_std).toFixed(4) : '--'}</td>;
            case 'tier':
              return (
                <td key={col.key} style={tdStyle}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <StatusBadge entry={entry} />
                  </div>
                </td>
              );
            case '_actions':
              return (
                <td key={col.key} style={tdStyle} onClick={e => e.stopPropagation()}>
                  <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                    <button
                      onClick={() => setExpandedRowId(isExpanded ? null : rowId)}
                      style={{
                        ...actionBtnStyle,
                        borderColor: 'var(--accent-blue)',
                        color: 'var(--accent-blue)',
                        background: isExpanded ? 'rgba(88, 166, 255, 0.12)' : 'transparent',
                      }}
                    >
                      {isExpanded ? 'Collapse' : 'Details'}
                    </button>
                    {onOpenInDesigner && (
                      <button
                        onClick={() => {
                          if (entry.result_id) onOpenInDesigner(entry.result_id)
                        }}
                        disabled={!entry.result_id}
                        style={{
                          ...actionBtnStyle,
                          borderColor: 'var(--accent-purple)',
                          color: 'var(--accent-purple)',
                          opacity: entry.result_id ? 1 : 0.5,
                          cursor: entry.result_id ? 'pointer' : 'not-allowed',
                        }}
                        title={entry.result_id ? 'Open architecture in visual designer' : 'Designer unavailable: missing result ID'}
                      >
                        Designer
                      </button>
                    )}
                    {canDelete && (
                      <button
                        onClick={() => {
                          if (window.confirm(`Delete ${entry.entry_id?.slice(0, 11) || entry.result_id?.slice(0, 11)} and all associated data?`)) {
                            handleDelete(entry.entry_id || entry.result_id);
                          }
                        }}
                        style={{
                          ...actionBtnStyle,
                          borderColor: 'rgba(248, 81, 73, 0.4)',
                          background: 'rgba(248, 81, 73, 0.12)',
                          color: 'var(--accent-red, #f85149)',
                        }}
                        title="Delete entry and all associated data"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </td>
              );
            default:
              return null;
          }
        })}
      </tr>
      {isExpanded && (
        <ExpandedDetail
          entry={entry}
          onInvestigate={onInvestigate}
          onValidate={onValidate}
          onQueueAdd={onQueueAdd}
          onQueueRemove={onQueueRemove}
          onDelete={handleDelete}
          isQueued={isQueued}
          eligibility={eligibility}
          statusDraft={statusDrafts[rowId] || entry.tier}
          onStatusDraftChange={(tier) => handleStatusDraftChange(rowId, tier)}
          onSaveStatus={() => handleSaveStatus(entry)}
          savingStatus={savingStatusRowId === rowId}
        />
      )}
    </React.Fragment>
  );
});

export default Discoveries;
