/**
 * Column renderer map for LeaderboardRow.
 *
 * Each key maps a column ID to a function: (entry, ctx) => ReactNode
 * where ctx provides shared state (compression, chips, eligibility, etc).
 *
 * Eliminates the 47-case switch statement in LeaderboardRow.
 */
import React from 'react';
import { reliabilityColor } from '../../utils/colors';
import TierBadge from '../shared/TierBadge';
import StatusBadge from '../shared/StatusBadge';
import Sparkline from '../shared/Sparkline';
import ScoreBreakdown from './ScoreBreakdown';

const fmt = (v, d = 4) => {
  if (v == null) return '--';
  const num = Number(v);
  if (num !== 0 && Math.abs(num) < 0.0001) return num.toExponential(2);
  return num.toFixed(d);
};

// Helper: simple threshold-colored metric
const coloredMetric = (value, thresholds, decimals = 3, pct = false) => {
  if (value == null) return '--';
  const v = Number(value);
  let color = 'var(--text-primary)';
  for (const [lo, hi, c] of thresholds) {
    if ((lo === null || v >= lo) && (hi === null || v < hi)) { color = c; break; }
  }
  const display = pct ? `${(v * 100).toFixed(1)}%` : v.toFixed(decimals);
  return <span style={{ color, fontWeight: 600 }}>{display}</span>;
};

/**
 * All column renderers. Each entry: (entry, ctx) => ReactNode.
 * ctx shape: { compression, chips, reproPacket, eligibility, isExpanded,
 *              hasBeenInvestigated, hasBeenValidated, canDelete,
 *              onInvestigate, onValidate, onToggleExpand, onDelete, rowId, handleActionClick, actionBtnStyle }
 */
const RENDERERS = {
  _score: (entry) => <ScoreBreakdown entry={entry} />,
  tier: (entry) => <TierBadge tier={entry.tier} entry={entry} />,

  _verified: (entry) => {
    const tags = (entry.tags || '').toLowerCase();
    const isRef = entry.is_reference;
    const hasTiktoken = tags.includes('tiktoken_native') || isRef;
    const hasWikitext = tags.includes('wikitext103') || isRef;
    let vLabel, vColor;
    if (hasTiktoken && hasWikitext) { vLabel = '\u2713'; vColor = 'var(--accent-green)'; }
    else if (hasTiktoken) { vLabel = '\u26A0'; vColor = 'var(--accent-yellow)'; }
    else { vLabel = '\u2717'; vColor = 'var(--accent-red)'; }
    return <span style={{ fontSize: 12, color: vColor, fontWeight: 700 }}>{vLabel}</span>;
  },

  _rate: (entry) => {
    const rate = entry.loss_improvement_rate;
    if (rate == null) return <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>?</span>;
    const pct = (rate * 100).toFixed(1);
    const rColor = rate > 0.10 ? 'var(--accent-green)' : rate > 0.05 ? 'var(--accent-yellow)' : 'var(--accent-red)';
    return <span style={{ fontSize: 11, color: rColor, fontWeight: 600 }}>{pct}%</span>;
  },

  _gap: (entry) => {
    const gap = entry.gap_vs_gpt2;
    if (gap == null) return <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>--</span>;
    const gColor = gap < 0 ? 'var(--accent-green)' : gap < 0.1 ? 'var(--accent-yellow)' : 'var(--accent-red)';
    return <span style={{ fontSize: 11, color: gColor, fontWeight: 600 }}>{gap > 0 ? '+' : ''}{gap.toFixed(2)}</span>;
  },

  _stability: (entry) => {
    const s = entry.cross_run_stability || {};
    const trend = s.trend || 'unknown';
    const sColor = trend === 'up' ? 'var(--accent-green)' : trend === 'down' ? 'var(--accent-red)' : trend === 'stable' ? 'var(--accent-yellow)' : 'var(--text-muted)';
    return <span style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', padding: '2px 6px', borderRadius: 4, color: sColor, background: `${sColor}22`, border: `1px solid ${sColor}55` }}>{trend}</span>;
  },

  model_source: (entry) => (
    <>
      {entry.is_reference && <span style={{ fontSize: 10, color: 'var(--accent-purple)', border: '1px solid var(--accent-purple)', borderRadius: 4, padding: '1px 6px', marginRight: 6 }}>REF</span>}
      <span style={{ fontSize: 10, color: entry.model_source === 'morphological_box' ? 'var(--accent-purple)' : 'var(--accent-blue)' }}>
        {entry.model_source === 'reference' ? 'REF' : entry.model_source === 'morphological_box' ? 'MORPH' : 'GRAPH'}
      </span>
    </>
  ),

  architecture_family: (entry) => entry.architecture_family || '--',
  architecture_desc: (entry) => entry.reference_name || entry.architecture_desc || entry.result_id?.slice(0, 12),

  _composition: (entry) => {
    const templates = entry.applied_templates || [];
    if (templates.length === 0) return <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>--</span>;
    return (
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
        {templates.slice(0, 3).map((t, i) => (
          <span key={i} title={t.name} style={{ fontSize: 9, padding: '1px 4px', borderRadius: 3, background: 'rgba(88, 166, 255, 0.15)', color: 'var(--accent-blue)', border: '1px solid rgba(88, 166, 255, 0.3)' }}>
            {t.name?.replace(/_template$/, '').replace(/apply_/, '')}
          </span>
        ))}
        {templates.length > 3 && <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>+{templates.length - 3}</span>}
      </div>
    );
  },

  _vs_reference: (entry) => entry._vs_reference != null ? `${entry._vs_reference.toFixed(0)}%` : '--',

  composite_score: (entry) => {
    const tags = (entry.tags || '').toLowerCase();
    const hasTiktoken = tags.includes('tiktoken_native') || entry.is_reference;
    const tokIcon = hasTiktoken ? '\u2713' : '\u26A0';
    const tokColor = hasTiktoken ? 'var(--accent-green)' : 'var(--accent-yellow)';
    return (
      <span style={{ color: 'var(--accent-green)' }}>
        {fmt(entry.composite_score, 3)}
        <span style={{ fontSize: 10, marginLeft: 4, color: tokColor }} title={hasTiktoken ? 'tiktoken-native' : 'byte-era'}>{tokIcon}</span>
      </span>
    );
  },

  // Simple fmt columns
  discovery_loss_ratio: (e) => fmt(e.discovery_loss_ratio),
  validation_loss_ratio: (e) => fmt(e.validation_loss_ratio),
  moe_routing_efficiency: (e) => fmt(e.moe_routing_efficiency, 3),
  screening_loss_ratio: (e) => fmt(e.screening_loss_ratio),
  screening_novelty: (e) => fmt(e.screening_novelty, 3),
  investigation_loss_ratio: (e) => fmt(e.investigation_loss_ratio),
  robustness_noise_score: (e) => fmt(e.robustness_noise_score, 3),
  robustness_long_ctx_score: (e) => fmt(e.robustness_long_ctx_score, 3),
  robustness_long_ctx_scaling_score: (e) => fmt(e.robustness_long_ctx_scaling_score, 3),
  robustness_long_ctx_assoc_score: (e) => fmt(e.robustness_long_ctx_assoc_score, 3),
  robustness_long_ctx_multi_hop_score: (e) => fmt(e.robustness_long_ctx_multi_hop_score, 3),
  robustness_long_ctx_passkey_score: (e) => fmt(e.robustness_long_ctx_passkey_score, 3),
  robustness_long_ctx_retrieval_aggregate: (e) => fmt(e.robustness_long_ctx_retrieval_aggregate, 3),
  robustness_long_ctx_combined_score: (e) => fmt(e.robustness_long_ctx_combined_score, 3),
  init_sensitivity_std: (e) => fmt(e.init_sensitivity_std, 4),
  jacobian_spectral_norm: (e) => fmt(e.jacobian_spectral_norm ?? e.fp_jacobian_spectral_norm, 4),
  max_viable_seq_len: (e) => e.max_viable_seq_len != null ? Number(e.max_viable_seq_len).toFixed(0) : '--',
  quant_int8_retention: (e) => e._quant_retention_pct != null ? `${e._quant_retention_pct.toFixed(1)}%` : '--',
  divergence_step: (e) => e.divergence_step || '--',

  // Colored metrics
  arch_quality_score: (e) => (
    <span style={{ color: e.arch_quality_score > 0.7 ? 'var(--accent-green)' : (e.arch_quality_score < 0.4 ? 'var(--accent-red)' : 'var(--text-primary)') }}>
      {fmt(e.arch_quality_score, 3)}
    </span>
  ),
  investigation_robustness: (e) => (
    <span style={{ color: e.investigation_robustness >= 0.5 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
      {fmt(e.investigation_robustness, 2)}
    </span>
  ),
  wikitext_ppl: (e) => <span style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>{fmt(e.wikitext_ppl ?? e.wikitext_perplexity, 2)}</span>,
  peak_ppl: (e) => <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{fmt(e.peak_ppl, 2)}</span>,
  validation_baseline_ratio: (e) => <span style={{ color: e.validation_baseline_ratio < 1 ? 'var(--accent-green)' : 'var(--accent-red)' }}>{fmt(e.validation_baseline_ratio)}</span>,

  hellaswag_acc: (e) => coloredMetric(e.hellaswag_acc, [
    [null, 0.281, 'var(--accent-red)'], [0.31, null, 'var(--accent-green)'], [0.281, 0.31, 'var(--accent-yellow)'],
  ], 1, true),
  induction_auc: (e) => coloredMetric(e.induction_auc, [
    [null, 0.10, 'var(--accent-red)'], [0.35, null, 'var(--accent-green)'], [0.10, 0.35, 'var(--accent-yellow)'],
  ]),
  induction_v2_investigation_auc: (e) => coloredMetric(e.induction_v2_investigation_auc, [
    [null, 0.30, 'var(--accent-red)'], [0.70, null, 'var(--accent-green)'], [0.30, 0.70, 'var(--accent-yellow)'],
  ]),
  ar_auc: (e) => coloredMetric(e.ar_auc, [
    [null, 0.05, 'var(--accent-red)'], [0.20, null, 'var(--accent-green)'], [0.05, 0.20, 'var(--accent-yellow)'],
  ]),
  binding_auc: (e) => coloredMetric(e.binding_auc, [
    [null, 0.10, 'var(--accent-red)'], [0.30, null, 'var(--accent-green)'], [0.10, 0.30, 'var(--accent-yellow)'],
  ]),
  binding_v2_investigation_auc: (e) => coloredMetric(e.binding_v2_investigation_auc, [
    [null, 0.30, 'var(--accent-red)'], [0.70, null, 'var(--accent-green)'], [0.30, 0.70, 'var(--accent-yellow)'],
  ]),
  blimp_overall_accuracy: (e) => coloredMetric(e.blimp_overall_accuracy, [
    [null, 0.501, 'var(--accent-red)'], [0.60, null, 'var(--accent-green)'], [0.501, 0.60, 'var(--accent-yellow)'],
  ], 1, true),

  pre_inv_score: (e) => {
    const pis = Number(e.pre_inv_score);
    const pisColor = pis >= 50 ? 'var(--accent-green)' : pis >= 20 ? 'var(--accent-yellow)' : 'var(--accent-red)';
    return <span style={{ color: pisColor, fontWeight: 600 }}>{fmt(pis, 1)}</span>;
  },

  wikitext_ppl_trajectory: (e) => {
    const data = Array.isArray(e.wikitext_ppl_trajectory) ? e.wikitext_ppl_trajectory :
                 (typeof e.wikitext_ppl_trajectory === 'string' ? e.wikitext_ppl_trajectory.split(',').map(v => parseFloat(v.trim())) : null);
    return <Sparkline data={data} />;
  },

  evaluation_stage: (e) => (
    <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
      <span style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-secondary)' }}>{e.evaluation_stage || '--'}</span>
      {e.is_frontier_signal && <StatusBadge type="FRONTIER_SIGNAL" label="FRONTIER" title="Model beats reference PPL at equal budget" />}
      {(e.improvement_ratio > 2.0 || e.is_slow_burn) && <StatusBadge type="SLOW_BURN" label="SLOW-BURN" title="Sharp trajectory: PPL improved >2x" />}
      {e.divergence_step && <StatusBadge type="DIVERGED" label={`DIVERGED @ ${e.divergence_step}`} title="PPL diverged >2x peak" />}
      {!e.divergence_step && e.wikitext_ppl_trajectory?.length >= 4 && <StatusBadge type="STABLE_GENERALIZER" label="STABLE" title="No divergence seen in 4000 steps" />}
    </div>
  ),

  robustness_grade: (e) => {
    const grade = e.robustness_grade || (e.investigation_robustness >= 0.8 ? 'A' : e.investigation_robustness >= 0.5 ? 'B' : e.investigation_robustness != null ? 'C' : null);
    const statusType = grade === 'A' ? 'ROBUST' : grade === 'B' ? 'STABLE' : grade === 'C' ? 'FRAGILE' : null;
    return statusType ? <StatusBadge type={statusType} label={grade} title={`Robustness Grade ${grade}`} /> : null;
  },

  _compression_ratio: (e, ctx) => (
    <>
      <div style={{ fontSize: 11 }}>{ctx.compression?.ratio != null ? `${(ctx.compression.ratio * 100).toFixed(0)}%` : '--'}</div>
      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{ctx.compression?.memoryMb?.toFixed(2)} MB · {ctx.compression?.label}</div>
    </>
  ),

  _metric_quality: (e, ctx) => (
    <>
      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Q: {ctx.reproPacket.label} · R: {ctx.reproPacket.label}</div>
      {ctx.isExpanded && (
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 4 }}>
          {ctx.chips.map(c => <span key={c.label} style={{ fontSize: 10, padding: '1px 5px', borderRadius: 4, background: `${reliabilityColor(c.reliability)}22`, color: reliabilityColor(c.reliability) }}>{c.label}</span>)}
        </div>
      )}
    </>
  ),
};

// Columns that need special td styling (not just tdStyle)
const TD_STYLE_OVERRIDES = {
  architecture_desc: { maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
};

export { RENDERERS, TD_STYLE_OVERRIDES, fmt };
