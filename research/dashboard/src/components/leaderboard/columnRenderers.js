/**
 * Column renderer map for LeaderboardRow.
 *
 * Each key maps a column ID to a function: (entry, ctx) => ReactNode
 * where ctx provides shared state (compression, chips, eligibility, etc).
 *
 * Eliminates the 47-case switch statement in LeaderboardRow.
 */
import React from 'react';
import { blimpColor, hellaswagColor, pplColor, probeAucColor, reliabilityColor } from '../../utils/colors';
import { scoreColor } from '../../utils/format';
import { evalMetricQuality } from '../../utils/backendScore';
import TierBadge from '../shared/TierBadge';
import StatusBadge from '../shared/StatusBadge';
import Sparkline from '../shared/Sparkline';
import ScoreBreakdown from './ScoreBreakdown';
import { capabilityQualityLabel, capabilityQualityStatus } from '../../utils/discoveryStatus';

const fmt = (v, d = 4) => {
  if (v == null) return '--';
  const num = Number(v);
  if (num !== 0 && Math.abs(num) < 0.0001) return num.toExponential(2);
  return num.toFixed(d);
};

const fmtInt = (v) => {
  if (v == null) return '--';
  const num = Number(v);
  return Number.isFinite(num) ? num.toFixed(0) : '--';
};

const textMetric = (v) => {
  if (v == null || v === '') return '--';
  const text = String(v);
  return <span title={text} style={{ fontSize: 10, color: 'var(--text-secondary)' }}>{text}</span>;
};

const jsonMetric = (v) => {
  if (v == null || v === '') return '--';
  return <span title={String(v)} style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'monospace' }}>json</span>;
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

const languageControlTiers = [
  {
    key: 's05',
    label: 'S05',
    maxPoints: 5,
    sa: 'language_control_s05_sentence_assoc_score',
    order: 'language_control_s05_binding_order_acc',
    nb: 'language_control_s05_binding_score',
    pointKeys: ['cl_s05_sa', 'cl_s05_order'],
  },
  {
    key: 's10',
    label: 'S10',
    maxPoints: 15,
    sa: 'language_control_s10_sentence_assoc_score',
    order: 'language_control_s10_binding_order_acc',
    nb: 'language_control_s10_binding_score',
    pointKeys: ['cl_s10_sa', 'cl_s10_order'],
  },
  {
    key: 'inv',
    label: 'INV',
    maxPoints: 25,
    sa: 'language_control_investigation_sentence_assoc_score',
    order: 'language_control_investigation_binding_order_acc',
    nb: 'language_control_investigation_binding_score',
    pointKeys: ['cl_investigation_sa', 'cl_investigation_order'],
  },
];

const finiteMetric = (value) => {
  if (value == null) return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const languageControlColor = (value) => {
  if (value == null) return 'var(--text-muted)';
  if (value >= 0.85) return 'var(--accent-green)';
  if (value >= 0.65) return 'var(--accent-yellow)';
  return 'var(--accent-red)';
};

function languageControlTierView(entry, tier) {
  const breakdown = entry?.score_breakdown || {};
  const sa = finiteMetric(entry?.[tier.sa]);
  const order = finiteMetric(entry?.[tier.order]);
  const nb = finiteMetric(entry?.[tier.nb]);
  const hasRawData = sa != null || order != null || nb != null;
  const rawValues = [sa, nb ?? order].filter((value) => value != null);
  const displayScore = rawValues.length
    ? rawValues.reduce((acc, value) => acc + value, 0) / rawValues.length
    : null;
  const points = tier.pointKeys.reduce((acc, key) => acc + (finiteMetric(breakdown[key]) || 0), 0);
  const hasPoints = hasRawData && (points > 0 || tier.pointKeys.some((key) => breakdown[key] != null));
  const fillRatio = hasPoints
    ? points / tier.maxPoints
    : displayScore;

  return {
    ...tier,
    sa,
    order,
    nb,
    hasRawData,
    displayScore,
    points: hasPoints ? points : null,
    fillRatio: fillRatio == null ? 0 : Math.max(0, Math.min(1, fillRatio)),
  };
}

function LanguageControlLadder({ entry }) {
  const rows = languageControlTiers.map((tier) => languageControlTierView(entry, tier));
  const hasAnyData = rows.some((row) => row.hasRawData);
  const invSa = finiteMetric(entry?.language_control_investigation_sentence_assoc_score);
  const invSaFlag = invSa != null && invSa < 0.85;
  const total = finiteMetric(entry?.score_breakdown?._v14_language_control_total);

  if (!hasAnyData) {
    return <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>--</span>;
  }

  return (
    <div
      title={[
        entry?.language_control_metric_version ? `version ${entry.language_control_metric_version}` : null,
        total != null ? `v14 controlled-lang points ${total.toFixed(1)}` : null,
        ...rows.map((row) => `${row.label}: SA ${fmt(row.sa, 2)}, NB order ${fmt(row.order, 2)}, NB score ${fmt(row.nb, 2)}${row.points != null ? `, pts ${row.points.toFixed(1)}/${row.maxPoints}` : ''}`),
        invSaFlag ? 'Flag: INV SA score is below 0.85' : null,
      ].filter(Boolean).join('\n')}
      style={{ minWidth: 86 }}
    >
      {rows.map((row) => {
        const color = languageControlColor(row.displayScore);
        const muted = row.displayScore == null && row.points == null;
        return (
          <div key={row.key} style={{ display: 'grid', gridTemplateColumns: '24px 48px 30px', alignItems: 'center', gap: 4, marginBottom: 2 }}>
            <span style={{ fontSize: 9, color: row.key === 'inv' && invSaFlag ? 'var(--accent-yellow)' : 'var(--text-muted)', fontWeight: 700 }}>
              {row.label}
            </span>
            <span style={{ height: 5, borderRadius: 2, overflow: 'hidden', background: 'var(--bg-tertiary)' }}>
              <span style={{ display: 'block', height: '100%', width: `${Math.max(3, row.fillRatio * 100)}%`, background: muted ? 'var(--text-muted)' : color }} />
            </span>
            <span style={{ fontSize: 10, color, fontVariantNumeric: 'tabular-nums', fontWeight: 700 }}>
              {row.displayScore == null ? '--' : row.displayScore.toFixed(2)}
              {row.key === 'inv' && invSaFlag && (
                <span style={{ marginLeft: 3, color: 'var(--accent-yellow)' }} aria-label="INV SA below 0.85">!</span>
              )}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/**
 * All column renderers. Each entry: (entry, ctx) => ReactNode.
 * ctx shape: { compression, chips, reproPacket, eligibility, isExpanded,
 *              hasBeenInvestigated, hasBeenValidated, canDelete,
 *              onInvestigate, onValidate, onToggleExpand, onDelete, rowId, handleActionClick, actionBtnStyle }
 */
const RENDERERS = {
  _score: (entry) => <ScoreBreakdown entry={entry} />,
  _capability_quality: (entry) => {
    const status = capabilityQualityStatus(entry);
    const label = capabilityQualityLabel(entry) || '--';
    const color = status === 'qualified' || status === 'breakthrough'
      ? 'var(--accent-green)'
      : status === 'pending'
        ? 'var(--accent-purple)'
        : status === 'training_only'
          ? 'var(--accent-yellow)'
          : 'var(--text-muted)';
    return <span style={{ color, fontWeight: 600 }}>{label}</span>;
  },
  tier: (entry) => <TierBadge tier={entry.tier} entry={entry} />,

  _verified: (entry) => {
    const quality = evalMetricQuality(entry);
    const icon = quality.key === 'trusted_bpe' ? '\u2713'
      : quality.key === 'partial_bpe' ? '\u25D0'
        : quality.key === 'legacy_eval' ? '\u26A0' : '\u2717';
    const title = quality.missing?.length
      ? `${quality.label}: missing ${quality.missing.join(', ')}`
      : `${quality.label}: ${quality.version}`;
    return <span style={{ fontSize: 12, color: quality.color, fontWeight: 700 }} title={title}>{icon}</span>;
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
    const quality = evalMetricQuality(entry);
    const tokIcon = quality.key === 'trusted_bpe' ? '\u2713'
      : quality.key === 'partial_bpe' ? '\u25D0'
        : quality.key === 'legacy_eval' ? '\u26A0' : '\u2717';
    const score = Number(entry.composite_score);
    return (
      <span style={{ color: Number.isFinite(score) ? scoreColor(score) : 'var(--text-muted)', fontWeight: 700 }}>
        {fmt(entry.composite_score, 3)}
        <span style={{ fontSize: 10, marginLeft: 4, color: quality.color }} title={`${quality.label}: ${quality.version}`}>{tokIcon}</span>
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
  champion_tiny_model_score: (e) => coloredMetric(e.champion_tiny_model_score, [
    [null, 20, 'var(--accent-red)'], [35, null, 'var(--accent-green)'], [20, 35, 'var(--accent-yellow)'],
  ], 1),
  champion_tiny_model_protocol_version: (e) => textMetric(e.champion_tiny_model_protocol_version),
  champion_hard_failure_reason: (e) => textMetric(e.champion_hard_failure_reason),
  champion_steps_to_floor: (e) => fmtInt(e.champion_steps_to_floor),
  champion_floor_loss: (e) => fmt(e.champion_floor_loss, 3),
  champion_floor_ppl: (e) => fmt(e.champion_floor_ppl, 2),
  champion_floor_loss_std: (e) => fmt(e.champion_floor_loss_std, 4),
  champion_plateau_detected_step: (e) => fmtInt(e.champion_plateau_detected_step),
  champion_plateau_window: (e) => fmtInt(e.champion_plateau_window),
  champion_floor_protocol_version: (e) => textMetric(e.champion_floor_protocol_version),
  champion_baseline_result_id: (e) => textMetric(e.champion_baseline_result_id),
  champion_baseline_layers: (e) => fmtInt(e.champion_baseline_layers),
  champion_baseline_protocol_version: (e) => textMetric(e.champion_baseline_protocol_version),
  champion_steps_to_floor_score: (e) => fmt(e.champion_steps_to_floor_score, 1),
  champion_floor_quality_score: (e) => fmt(e.champion_floor_quality_score, 1),
  champion_floor_stability_score: (e) => fmt(e.champion_floor_stability_score, 1),
  champion_induction_validation_score: (e) => fmt(e.champion_induction_validation_score, 1),
  champion_binding_long_context_score: (e) => fmt(e.champion_binding_long_context_score, 1),
  champion_ar_validation_score: (e) => fmt(e.champion_ar_validation_score, 1),
  init_sensitivity_std: (e) => fmt(e.init_sensitivity_std, 4),
  jacobian_spectral_norm: (e) => fmt(e.jacobian_spectral_norm ?? e.fp_jacobian_spectral_norm, 4),
  fp_jacobian_effective_rank: (e) => fmt(e.fp_jacobian_effective_rank, 3),
  fp_sensitivity_uniformity: (e) => fmt(e.fp_sensitivity_uniformity, 3),
  fp_jacobian_erf_density: (e) => coloredMetric(e.fp_jacobian_erf_density, [
    [null, 0.08, 'var(--text-muted)'], [0.18, null, 'var(--accent-green)'], [0.08, 0.18, 'var(--accent-yellow)'],
  ], 3),
  fp_id_collapse_rate: (e) => coloredMetric(e.fp_id_collapse_rate, [
    [null, 0.02, 'var(--text-muted)'], [0.10, null, 'var(--accent-green)'], [0.02, 0.10, 'var(--accent-yellow)'],
  ], 3),
  fp_id_collapse_rate_normalized: (e) => coloredMetric(e.fp_id_collapse_rate_normalized, [
    [null, 0.02, 'var(--text-muted)'], [0.10, null, 'var(--accent-green)'], [0.02, 0.10, 'var(--accent-yellow)'],
  ], 3),
  fp_jacobian_erf_decay_slope: (e) => coloredMetric(e.fp_jacobian_erf_decay_slope, [
    [null, 0.00, 'var(--text-muted)'], [0.20, null, 'var(--accent-green)'], [0.00, 0.20, 'var(--accent-yellow)'],
  ], 3),
  fp_jacobian_erf_first_norm: (e) => fmt(e.fp_jacobian_erf_first_norm, 3),
  fp_jacobian_erf_last_norm: (e) => fmt(e.fp_jacobian_erf_last_norm, 3),
  fp_logit_margin_velocity: (e) => fmt(e.fp_logit_margin_velocity, 3),
  fp_logit_margin_delta: (e) => fmt(e.fp_logit_margin_delta, 3),
  fp_jacobian_erf_variance_log: (e) => fmt(e.fp_jacobian_erf_variance_log, 3),
  fp_jacobian_spectral_norm_log: (e) => fmt(e.fp_jacobian_spectral_norm_log, 3),
  fp_icld_velocity: (e) => <span style={{ color: 'var(--text-muted)' }}>{fmt(e.fp_icld_velocity, 3)}</span>,
  fp_icld_delta_loss: (e) => <span style={{ color: 'var(--text-muted)' }}>{fmt(e.fp_icld_delta_loss, 3)}</span>,
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
  wikitext_ppl: (e) => {
    const value = e.wikitext_ppl ?? e.wikitext_perplexity;
    return <span style={{ color: pplColor(value), fontWeight: 600 }}>{fmt(value, 2)}</span>;
  },
  peak_ppl: (e) => <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{fmt(e.peak_ppl, 2)}</span>,
  validation_baseline_ratio: (e) => <span style={{ color: e.validation_baseline_ratio < 1 ? 'var(--accent-green)' : 'var(--accent-red)' }}>{fmt(e.validation_baseline_ratio)}</span>,

  hellaswag_acc: (e) => {
    if (e.hellaswag_acc == null) return '--';
    return <span style={{ color: hellaswagColor(e.hellaswag_acc), fontWeight: 600 }}>{(Number(e.hellaswag_acc) * 100).toFixed(1)}%</span>;
  },
  induction_screening_auc: (e) => <span style={{ color: probeAucColor(e.induction_screening_auc), fontWeight: 600 }}>{fmt(e.induction_screening_auc, 3)}</span>,
  induction_intermediate_auc: (e) => <span style={{ color: probeAucColor(e.induction_intermediate_auc), fontWeight: 600 }}>{fmt(e.induction_intermediate_auc, 3)}</span>,
  induction_validation_auc: (e) => <span style={{ color: probeAucColor(e.induction_validation_auc), fontWeight: 600 }}>{fmt(e.induction_validation_auc, 3)}</span>,
  induction_validation_max_gap_acc: (e) => fmt(e.induction_validation_max_gap_acc, 3),
  induction_validation_gap_accuracy_cv: (e) => fmt(e.induction_validation_gap_accuracy_cv, 3),
  induction_validation_gap_accuracies_json: (e) => jsonMetric(e.induction_validation_gap_accuracies_json),
  induction_validation_steps_trained: (e) => fmtInt(e.induction_validation_steps_trained),
  induction_validation_status: (e) => textMetric(e.induction_validation_status),
  induction_validation_elapsed_ms: (e) => fmtInt(e.induction_validation_elapsed_ms),
  induction_validation_protocol_version: (e) => textMetric(e.induction_validation_protocol_version),
  ar_legacy_auc: (e) => <span style={{ color: probeAucColor(e.ar_legacy_auc), fontWeight: 600 }}>{fmt(e.ar_legacy_auc, 3)}</span>,
  ar_validation_rank_score: (e) => fmt(e.ar_validation_rank_score, 3),
  ar_validation_final_acc: (e) => fmt(e.ar_validation_final_acc, 3),
  ar_validation_held_pair_acc: (e) => fmt(e.ar_validation_held_pair_acc, 3),
  ar_validation_held_class_acc: (e) => fmt(e.ar_validation_held_class_acc, 3),
  ar_validation_learning_curve_json: (e) => jsonMetric(e.ar_validation_learning_curve_json),
  ar_validation_steps_to_floor: (e) => fmtInt(e.ar_validation_steps_to_floor),
  ar_validation_status: (e) => textMetric(e.ar_validation_status),
  ar_validation_elapsed_ms: (e) => fmtInt(e.ar_validation_elapsed_ms),
  ar_validation_metric_version: (e) => textMetric(e.ar_validation_metric_version),
  ar_curriculum_auc_pair_final: (e) => <span style={{ color: probeAucColor(e.ar_curriculum_auc_pair_final), fontWeight: 600 }}>{fmt(e.ar_curriculum_auc_pair_final, 3)}</span>,
  ar_curriculum_s0_retention: (e) => {
    if (e.ar_curriculum_s0_retention == null) return '--';
    const v = Number(e.ar_curriculum_s0_retention);
    const color = v < 0.30 ? 'var(--accent-red)' : v < 0.70 ? 'var(--accent-yellow)' : 'var(--accent-green)';
    return <span style={{ color, fontWeight: 600 }} title={v < 0.30 ? 'catastrophic forgetting' : v < 0.70 ? 'partial retention' : 'retention preserved'}>{fmt(v, 2)}</span>;
  },
  ar_curriculum_max_passing_stage: (e) => {
    const v = e.ar_curriculum_max_passing_stage;
    if (v == null) return '--';
    const n = Number(v);
    const color = n < 0 ? 'var(--accent-red)' : n < 3 ? 'var(--accent-yellow)' : 'var(--accent-green)';
    return <span style={{ color, fontWeight: 600 }}>{n}</span>;
  },
  ar_curriculum_per_stage_held_pair_acc: (e) => jsonMetric(e.ar_curriculum_per_stage_held_pair_acc),
  ar_curriculum_status: (e) => textMetric(e.ar_curriculum_status),
  ar_curriculum_elapsed_ms: (e) => fmtInt(e.ar_curriculum_elapsed_ms),
  ar_curriculum_metric_version: (e) => textMetric(e.ar_curriculum_metric_version),
  binding_screening_auc: (e) => <span style={{ color: probeAucColor(e.binding_screening_auc), fontWeight: 600 }}>{fmt(e.binding_screening_auc, 3)}</span>,
  binding_intermediate_auc: (e) => <span style={{ color: probeAucColor(e.binding_intermediate_auc), fontWeight: 600 }}>{fmt(e.binding_intermediate_auc, 3)}</span>,
  _language_control_ladder: (e) => <LanguageControlLadder entry={e} />,
  blimp_overall_accuracy: (e) => {
    if (e.blimp_overall_accuracy == null) return '--';
    return <span style={{ color: blimpColor(e.blimp_overall_accuracy), fontWeight: 600 }}>{(Number(e.blimp_overall_accuracy) * 100).toFixed(1)}%</span>;
  },

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
  _language_control_ladder: { minWidth: 110 },
};

export { RENDERERS, TD_STYLE_OVERRIDES, fmt };
