import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { apiCall } from '../services/apiService';
import MiniChart from './charts/MiniChart';
import { fmtPct as _fmtPct, fmtLoss } from '../utils/format';
import { TIER_COLORS } from '../utils/scoringEngine';
import { useAriaData } from '../hooks/useAriaData';
import useInteractiveTable from './shared/useInteractiveTable';
import SortIndicator from './shared/SortIndicator';

const fmtPct = (v) => _fmtPct(v, 1);

const STATUS_COLORS = {
  healthy: '#22c55e',
  structural: '#6b7280',
  degraded: '#eab308',
  broken: '#ef4444',
};

const SOURCE_COLORS = {
  search: '#3b82f6',
  'search+profiling': '#8b5cf6',
  profiling_only: '#6b7280',
};

const TIME_WINDOWS = [
  { value: '1h', label: '1h' },
  { value: '6h', label: '6h' },
  { value: '24h', label: '24h' },
  { value: '7d', label: '7d' },
  { value: 'all', label: 'All' },
];

// ─── Health Summary ───
function HealthSummary({ health }) {
  if (!health) return null;
  const items = [
    { label: 'Total Ops', value: health.total, color: 'var(--text-primary)' },
    { label: 'Healthy', value: health.healthy, color: STATUS_COLORS.healthy },
    { label: 'Degraded', value: health.degraded, color: STATUS_COLORS.degraded },
    { label: 'Broken', value: health.broken, color: STATUS_COLORS.broken },
  ];
  return (
    <div style={{ display: 'flex', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
      {items.map(item => (
        <div key={item.label} style={{
          flex: '1 1 100px', padding: '10px 14px', borderRadius: 8,
          background: 'var(--bg-secondary)',
          border: `1px solid ${item.value > 0 && item.label !== 'Total Ops' && item.label !== 'Healthy' ? item.color + '44' : 'var(--border-color)'}`,
        }}>
          <div style={{ fontSize: 22, fontWeight: 700, color: item.color }}>{item.value}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{item.label}</div>
        </div>
      ))}
    </div>
  );
}

// ─── Sortable Column Header ───
const SORT_COLUMNS = [
  { key: 'status', label: 'Status', tooltip: 'Current component health classification.', sticky: true, left: 0, width: 58, group: 'identity', always: true },
  { key: 'op', label: 'Op Name', tooltip: 'Component/operator name.', sticky: true, left: 58, width: 260, group: 'identity', always: true },
  { key: 'n_used', label: 'Used', tooltip: 'Number of observed programs containing this op.', group: 'health' },
  { key: 's0_rate', label: 'S0', tooltip: 'Share of observed programs containing this op that passed Stage 0.', group: 'health' },
  { key: 's05_rate', label: 'S0.5', tooltip: 'Share of observed programs containing this op that passed the stability band.', group: 'health' },
  { key: 's1_rate', label: 'S1', tooltip: 'Share of Stage 0 passes that reached Stage 1.', group: 'health' },
  { key: 'avg_composite_score', label: 'Score', tooltip: 'Average leaderboard composite score for programs containing this op.', group: 'learning' },
  { key: 'avg_loss_ratio', label: 'Train LR', tooltip: 'Average training loss ratio for programs containing this op.', group: 'learning' },
  { key: 'avg_validation_loss_ratio', label: 'Val LR', tooltip: 'Average validation loss ratio for programs containing this op.', group: 'learning' },
  { key: 'avg_induction_auc', label: 'Ind', tooltip: 'Average induction-task AUC for programs containing this op.', group: 'benchmarks' },
  { key: 'avg_induction_v2_auc', label: 'Ind v2', tooltip: 'Average induction v2 investigation AUC for programs containing this op.', group: 'benchmarks' },
  { key: 'avg_binding_auc', label: 'Bind', tooltip: 'Average binding/copy-task AUC for programs containing this op.', group: 'benchmarks' },
  { key: 'avg_binding_v2_auc', label: 'Bind v2', tooltip: 'Average binding v2 investigation AUC for programs containing this op.', group: 'benchmarks' },
  { key: 'avg_controlled_lang_s05_score', label: 'CL S05', tooltip: 'Average controlled-language S0.5 tier score: SA plus nano-BLiMP score/order.', group: 'benchmarks' },
  { key: 'avg_controlled_lang_s10_score', label: 'CL S10', tooltip: 'Average controlled-language S1.0 tier score: SA plus nano-BLiMP score/order.', group: 'benchmarks' },
  { key: 'avg_controlled_lang_inv_score', label: 'CL Inv', tooltip: 'Average controlled-language investigation tier score. Shows a yellow flag when investigation SA is below 0.850.', group: 'benchmarks' },
  { key: 'avg_hellaswag_acc', label: 'Hella', tooltip: 'Average HellaSwag accuracy signal for programs containing this op.', group: 'benchmarks' },
  { key: 'avg_blimp_overall_accuracy', label: 'BLiMP', tooltip: 'Average BLiMP grammatical reasoning accuracy for programs containing this op.', group: 'benchmarks' },
  { key: 'avg_erf_density', label: 'ERF Dens', tooltip: 'Average ERF density for programs containing this op. Strong binding v2 predictor.', group: 'architecture' },
  { key: 'avg_id_collapse_rate', label: 'ID Coll', tooltip: 'Average intrinsic-dimension collapse rate. Strong binding v2 signal, sparse but meaningful.', group: 'architecture' },
  { key: 'avg_id_collapse_rate_normalized', label: 'ID CollN', tooltip: 'Average normalized intrinsic-dimension collapse rate.', group: 'architecture' },
  { key: 'avg_erf_decay_slope', label: 'ERF Decay', tooltip: 'Average ERF decay slope. Moderate binding and induction v2 signal.', group: 'architecture' },
  { key: 'avg_erf_first_norm', label: 'ERF First', tooltip: 'Average ERF first-position norm.', group: 'architecture' },
  { key: 'avg_erf_last_norm', label: 'ERF Last', tooltip: 'Average ERF last-position norm.', group: 'architecture' },
  { key: 'avg_logit_margin_velocity', label: 'Margin Vel', tooltip: 'Average logit-margin velocity. Weak positive capability signal.', group: 'architecture' },
  { key: 'avg_logit_margin_delta', label: 'Margin Δ', tooltip: 'Average logit-margin delta.', group: 'architecture' },
  { key: 'avg_erf_variance_log', label: 'ERF VarLog', tooltip: 'Average log-scaled ERF variance. Mild negative capability correlation.', group: 'architecture' },
  { key: 'avg_spec_norm_log', label: 'SpecLog', tooltip: 'Average log-scaled spectral norm. Mostly stability/loss signal.', group: 'architecture' },
  { key: 'avg_icld_velocity', label: 'ICLD Vel', tooltip: 'Average ICLD velocity. Empirically near-noise; audit only.', group: 'architecture' },
  { key: 'avg_icld_delta_loss', label: 'ICLD ΔLoss', tooltip: 'Average ICLD early-to-late loss delta.', group: 'architecture' },
  { key: 'avg_jacobian_effective_rank', label: 'JRank', tooltip: 'Average Jacobian effective rank.', group: 'architecture' },
  { key: 'avg_sensitivity_uniformity', label: 'SensUnif', tooltip: 'Average sensitivity uniformity.', group: 'architecture' },
  { key: 'grad_norm', label: 'Grad Norm', tooltip: 'Gradient norm from component profiling.', group: 'runtime' },
  { key: 'fwd_us', label: 'Fwd (us)', tooltip: 'Forward-pass runtime in microseconds from component profiling.', group: 'runtime' },
  { key: 'reasons', label: 'Issues', tooltip: 'Dominant failure reason or health-grid diagnostic.', group: 'diagnosis' },
];

const COLUMN_GROUPS = [
  { key: 'health', label: 'Health' },
  { key: 'learning', label: 'Learning' },
  { key: 'benchmarks', label: 'Benchmarks' },
  { key: 'architecture', label: 'Architecture' },
  { key: 'runtime', label: 'Runtime' },
  { key: 'diagnosis', label: 'Diagnosis' },
];

const COMPONENT_VIEW_PRESETS = [
  { key: 'triage', label: 'Triage', columns: ['avg_composite_score', 'n_used', 's1_rate', 'avg_loss_ratio', 'avg_validation_loss_ratio', 'avg_induction_v2_auc', 'avg_binding_v2_auc', 'avg_controlled_lang_s05_score', 'avg_controlled_lang_s10_score', 'avg_controlled_lang_inv_score', 'avg_hellaswag_acc', 'avg_blimp_overall_accuracy', 'avg_erf_density', 'avg_id_collapse_rate', 'reasons'] },
  { key: 'learning', label: 'Learning', columns: ['avg_composite_score', 'n_used', 's0_rate', 's05_rate', 's1_rate', 'avg_loss_ratio', 'avg_validation_loss_ratio', 'avg_induction_v2_auc', 'avg_binding_v2_auc', 'avg_controlled_lang_s05_score', 'avg_controlled_lang_inv_score', 'reasons'] },
  { key: 'benchmarks', label: 'Benchmarks', columns: ['avg_composite_score', 'n_used', 'avg_induction_auc', 'avg_induction_v2_auc', 'avg_binding_auc', 'avg_binding_v2_auc', 'avg_controlled_lang_s05_score', 'avg_controlled_lang_s10_score', 'avg_controlled_lang_inv_score', 'avg_hellaswag_acc', 'avg_blimp_overall_accuracy', 'reasons'] },
  { key: 'architecture', label: 'Architecture', columns: ['n_used', 'avg_binding_v2_auc', 'avg_erf_density', 'avg_id_collapse_rate', 'avg_id_collapse_rate_normalized', 'avg_erf_decay_slope', 'avg_logit_margin_velocity', 'avg_jacobian_effective_rank', 'reasons'] },
  { key: 'runtime', label: 'Runtime', columns: ['n_used', 'grad_norm', 'fwd_us', 'reasons'] },
  { key: 'all', label: 'All Columns', columns: SORT_COLUMNS.map(col => col.key) },
];

const REASON_FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'low_s1', label: 'Low S1' },
  { key: 'poor_val', label: 'Poor Val' },
  { key: 'low_ind', label: 'Low Ind' },
  { key: 'low_bind', label: 'Low Bind' },
  { key: 'high_grad', label: 'High Grad' },
  { key: 'slow_fwd', label: 'Slow Fwd' },
  { key: 'runtime_excluded', label: 'Runtime Excluded' },
];

const SORT_PRESETS = [
  { key: 'most_used', label: 'Most Used', sortKey: 'n_used', desc: true },
  { key: 'worst_s1', label: 'Worst S1', sortKey: 's1_rate', desc: false },
  { key: 'best_val', label: 'Best Val', sortKey: 'avg_validation_loss_ratio', desc: false },
  { key: 'worst_gap', label: 'Worst Val Gap', sortKey: 'val_gap', desc: true },
  { key: 'best_ind', label: 'Best Ind', sortKey: 'avg_induction_auc', desc: true },
  { key: 'best_cl_inv', label: 'Best CL Inv', sortKey: 'avg_controlled_lang_inv_score', desc: true },
  { key: 'slowest', label: 'Slowest Runtime', sortKey: 'fwd_us', desc: true },
];

const STATUS_ORDER = { broken: 0, degraded: 1, structural: 2, healthy: 3 };

function usePersistentState(key, initialValue) {
  const [value, setValue] = useState(() => {
    if (typeof window === 'undefined') return initialValue;
    try {
      const stored = window.localStorage.getItem(key);
      return stored === null ? initialValue : JSON.parse(stored);
    } catch {
      return initialValue;
    }
  });
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Ignore localStorage failures.
    }
  }, [key, value]);
  return [value, setValue];
}

function getComponentSortValue(row, key) {
  if (key === 'status') return STATUS_ORDER[row.status] ?? 3;
  if (key === 'reasons') return Array.isArray(row.reasons) ? row.reasons.length : 0;
  if (key === 'val_gap') {
    const val = Number(row.avg_validation_loss_ratio);
    const train = Number(row.avg_loss_ratio);
    if (!Number.isFinite(val) || !Number.isFinite(train)) return null;
    return val - train;
  }
  return row[key];
}

function getComponentInitialSortDesc(key) {
  return key !== 'op';
}

function getPairInitialSortDesc(key) {
  return key !== 'op_a' && key !== 'op_b';
}

function metricText(value, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '-';
  return Number(value).toFixed(digits);
}

function controlledLangTone(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 'var(--text-muted)';
  if (num >= 0.85) return 'var(--accent-green)';
  if (num >= 0.65) return STATUS_COLORS.degraded;
  return STATUS_COLORS.broken;
}

function controlledLangFlag(row) {
  const invSa = Number(row.avg_controlled_lang_inv_sa_score);
  return Number.isFinite(invSa) && invSa < 0.85;
}

function controlledLangCell(row, key) {
  return (
    <td
      title={controlledLangFlag(row) ? `INV SA ${metricText(row.avg_controlled_lang_inv_sa_score)} below 0.850` : undefined}
      style={{ textAlign: 'right', color: controlledLangTone(row[key]), fontWeight: 600, whiteSpace: 'nowrap' }}
    >
      {controlledLangFlag(row) && key === 'avg_controlled_lang_inv_score' ? <span style={{ color: STATUS_COLORS.degraded, marginRight: 4 }}>!</span> : null}
      {metricText(row[key])}
    </td>
  );
}

function componentMatchesReason(component, reasonFilter) {
  if (reasonFilter === 'all') return true;
  if (reasonFilter === 'low_s1') return component.s1_rate != null && component.s1_rate < 0.15;
  if (reasonFilter === 'poor_val') {
    const val = Number(component.avg_validation_loss_ratio);
    const train = Number(component.avg_loss_ratio);
    return Number.isFinite(val) && (
      val >= 0.65 || (Number.isFinite(train) && val > train * 1.15)
    );
  }
  if (reasonFilter === 'low_ind') return component.avg_induction_auc != null && component.avg_induction_auc < 0.02;
  if (reasonFilter === 'low_bind') return component.avg_binding_auc != null && component.avg_binding_auc < 0.05;
  if (reasonFilter === 'high_grad') return component.grad_norm != null && component.grad_norm > 3000;
  if (reasonFilter === 'slow_fwd') return component.fwd_us != null && component.fwd_us > 1000;
  if (reasonFilter === 'runtime_excluded') return Number(component.n_excluded || 0) > 0;
  return true;
}

function componentValLossTone(component) {
  const val = Number(component.avg_validation_loss_ratio);
  const train = Number(component.avg_loss_ratio);
  if (!Number.isFinite(val)) return 'var(--text-muted)';
  if (val >= 0.68 || (Number.isFinite(train) && val > train * 1.15)) return STATUS_COLORS.degraded;
  return 'var(--text-primary)';
}

function stickyCellStyle(col, extra = {}) {
  if (!col?.sticky) return extra;
  return {
    ...extra,
    position: 'sticky',
    left: col.left,
    minWidth: col.width,
    maxWidth: col.width,
    width: col.width,
    background: extra.background || 'var(--table-row-bg)',
    zIndex: extra.zIndex || 2,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    boxShadow: col.key === 'op' ? '1px 0 0 var(--border)' : undefined,
  };
}

function presetComponentColumns(columnView) {
  const preset = COMPONENT_VIEW_PRESETS.find(item => item.key === columnView) || COMPONENT_VIEW_PRESETS[0];
  const allowed = new Set(['status', 'op', ...preset.columns]);
  return SORT_COLUMNS.filter(col => col.always || allowed.has(col.key));
}

function normalizeColumnKeys(columns, keys) {
  const valid = new Set(columns.map(col => col.key));
  const normalized = Array.isArray(keys) ? keys.filter(key => valid.has(key)) : [];
  const always = columns.filter(col => col.always).map(col => col.key);
  return Array.from(new Set([...always, ...normalized]));
}

function requiredComponentColumns() {
  return [
    'avg_controlled_lang_s05_score',
    'avg_controlled_lang_s10_score',
    'avg_controlled_lang_inv_score',
  ];
}

function visibleComponentColumns(columnView, customColumnKeys) {
  if (Array.isArray(customColumnKeys) && customColumnKeys.length > 0) {
    const allowed = new Set(normalizeColumnKeys(SORT_COLUMNS, [...customColumnKeys, ...requiredComponentColumns()]));
    return SORT_COLUMNS.filter(col => allowed.has(col.key));
  }
  return presetComponentColumns(columnView);
}

function componentTableMinWidth(columns) {
  return columns.reduce((total, col) => total + (col.width || 88), 0);
}

function ColumnPickerPanel({ columns, selectedKeys, onChange, onReset }) {
  const selected = new Set(normalizeColumnKeys(columns, selectedKeys));
  return (
    <div style={{
      display: 'flex',
      gap: 10,
      flexWrap: 'wrap',
      padding: 10,
      marginBottom: 10,
      border: '1px solid var(--border-color)',
      borderRadius: 6,
      background: 'var(--bg-secondary)',
    }}>
      {columns.filter(col => !col.always).map(col => (
        <label key={col.key} title={col.tooltip} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-primary)', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={selected.has(col.key)}
            onChange={(event) => {
              const next = new Set(selected);
              if (event.target.checked) next.add(col.key);
              else next.delete(col.key);
              onChange(normalizeColumnKeys(columns, Array.from(next)));
            }}
          />
          {col.label}
        </label>
      ))}
      <button onClick={onReset} style={{ padding: '2px 8px', fontSize: 11, borderRadius: 10, border: '1px solid var(--border-color)', background: 'var(--bg-tertiary)', color: 'var(--text-muted)', cursor: 'pointer' }}>
        Preset
      </button>
    </div>
  );
}

function renderComponentCell(c, col) {
  if (col.key === 'status') {
    return (
      <td style={stickyCellStyle(col, {
        background: c.status === 'broken' ? '#18161a' : c.status === 'degraded' ? '#1b1a15' : 'var(--bg-secondary)',
        zIndex: 3,
      })}>
        <span style={{
          display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
          background: STATUS_COLORS[c.status],
          boxShadow: c.status !== 'healthy' ? `0 0 4px ${STATUS_COLORS[c.status]}66` : 'none',
        }} />
      </td>
    );
  }
  if (col.key === 'op') {
    return (
      <td style={stickyCellStyle(col, {
        fontFamily: 'monospace',
        fontWeight: c.status !== 'healthy' ? 600 : 400,
        background: c.status === 'broken' ? '#18161a' : c.status === 'degraded' ? '#1b1a15' : 'var(--bg-secondary)',
        zIndex: 3,
        whiteSpace: 'nowrap',
      })}>
        {c.op}
        {c.data_source && (
          <span style={{
            marginLeft: 6, padding: '1px 5px', borderRadius: 4, fontSize: 9,
            background: (SOURCE_COLORS[c.data_source] || '#666') + '22',
            color: SOURCE_COLORS[c.data_source] || '#666',
          }}>
            {c.data_source === 'profiling_only' ? 'prof' : c.data_source === 'search+profiling' ? 's+p' : 'src'}
          </span>
        )}
      </td>
    );
  }
  if (col.key === 'n_used') return <td style={{ textAlign: 'right' }}>{c.n_used || 0}</td>;
  if (col.key === 's0_rate') {
    return <td style={{ textAlign: 'right', color: c.s0_rate !== null ? (c.s0_rate < 0.3 ? STATUS_COLORS.broken : c.s0_rate < 0.6 ? STATUS_COLORS.degraded : 'var(--text-primary)') : 'var(--text-muted)' }}>{c.s0_rate !== null ? `${(c.s0_rate * 100).toFixed(0)}%` : '-'}</td>;
  }
  if (col.key === 's05_rate') {
    return <td style={{ textAlign: 'right', color: c.s05_rate !== null && c.s05_rate !== undefined ? (c.s05_rate < 0.3 ? STATUS_COLORS.broken : c.s05_rate < 0.6 ? STATUS_COLORS.degraded : 'var(--text-primary)') : 'var(--text-muted)' }}>{c.s05_rate !== null && c.s05_rate !== undefined ? `${(c.s05_rate * 100).toFixed(0)}%` : '-'}</td>;
  }
  if (col.key === 's1_rate') {
    return <td style={{ textAlign: 'right', color: c.s1_rate !== null ? (c.s1_rate < 0.05 ? STATUS_COLORS.broken : c.s1_rate < 0.15 ? STATUS_COLORS.degraded : 'var(--text-primary)') : 'var(--text-muted)' }}>{c.s1_rate !== null ? `${(c.s1_rate * 100).toFixed(1)}%` : '-'}</td>;
  }
  if (col.key === 'avg_composite_score') return <td style={{ textAlign: 'right', color: c.avg_composite_score != null && c.avg_composite_score >= 80 ? 'var(--accent-green)' : 'var(--text-primary)' }}>{metricText(c.avg_composite_score)}</td>;
  if (col.key === 'avg_loss_ratio') return <td style={{ textAlign: 'right' }}>{fmtLoss(c.avg_loss_ratio)}</td>;
  if (col.key === 'avg_validation_loss_ratio') {
    return (
      <td
        title={Number.isFinite(Number(c.avg_validation_loss_ratio)) && Number.isFinite(Number(c.avg_loss_ratio)) ? `Gap ${(Number(c.avg_validation_loss_ratio) - Number(c.avg_loss_ratio)).toFixed(3)}` : undefined}
        style={{ textAlign: 'right', color: componentValLossTone(c) }}
      >
        {fmtLoss(c.avg_validation_loss_ratio)}
      </td>
    );
  }
  if (col.key === 'avg_induction_auc') return <td style={{ textAlign: 'right' }}>{metricText(c.avg_induction_auc)}</td>;
  if (col.key === 'avg_induction_v2_auc') return <td style={{ textAlign: 'right' }}>{metricText(c.avg_induction_v2_auc)}</td>;
  if (col.key === 'avg_binding_auc') return <td style={{ textAlign: 'right' }}>{metricText(c.avg_binding_auc)}</td>;
  if (col.key === 'avg_binding_v2_auc') return <td style={{ textAlign: 'right', color: c.avg_binding_v2_auc != null && c.avg_binding_v2_auc >= 0.30 ? 'var(--accent-green)' : 'var(--text-primary)' }}>{metricText(c.avg_binding_v2_auc)}</td>;
  if (col.key === 'avg_controlled_lang_s05_score') return controlledLangCell(c, col.key);
  if (col.key === 'avg_controlled_lang_s10_score') return controlledLangCell(c, col.key);
  if (col.key === 'avg_controlled_lang_inv_score') return controlledLangCell(c, col.key);
  if (col.key === 'avg_hellaswag_acc') return <td style={{ textAlign: 'right' }}>{metricText(c.avg_hellaswag_acc)}</td>;
  if (col.key === 'avg_blimp_overall_accuracy') return <td style={{ textAlign: 'right' }}>{metricText(c.avg_blimp_overall_accuracy)}</td>;
  if (col.key === 'avg_erf_density') return <td style={{ textAlign: 'right', color: c.avg_erf_density != null && c.avg_erf_density >= 0.18 ? 'var(--accent-green)' : 'var(--text-primary)' }}>{metricText(c.avg_erf_density)}</td>;
  if (col.key === 'avg_id_collapse_rate') return <td style={{ textAlign: 'right', color: c.avg_id_collapse_rate != null && c.avg_id_collapse_rate >= 0.10 ? 'var(--accent-green)' : 'var(--text-primary)' }}>{metricText(c.avg_id_collapse_rate)}</td>;
  if (col.key === 'avg_id_collapse_rate_normalized') return <td style={{ textAlign: 'right', color: c.avg_id_collapse_rate_normalized != null && c.avg_id_collapse_rate_normalized >= 0.10 ? 'var(--accent-green)' : 'var(--text-primary)' }}>{metricText(c.avg_id_collapse_rate_normalized)}</td>;
  if (col.key === 'avg_erf_decay_slope') return <td style={{ textAlign: 'right', color: c.avg_erf_decay_slope != null && c.avg_erf_decay_slope >= 0.20 ? 'var(--accent-green)' : 'var(--text-primary)' }}>{metricText(c.avg_erf_decay_slope)}</td>;
  if (col.key === 'avg_erf_first_norm') return <td style={{ textAlign: 'right' }}>{metricText(c.avg_erf_first_norm)}</td>;
  if (col.key === 'avg_erf_last_norm') return <td style={{ textAlign: 'right' }}>{metricText(c.avg_erf_last_norm)}</td>;
  if (col.key === 'avg_logit_margin_velocity') return <td style={{ textAlign: 'right' }}>{metricText(c.avg_logit_margin_velocity)}</td>;
  if (col.key === 'avg_logit_margin_delta') return <td style={{ textAlign: 'right' }}>{metricText(c.avg_logit_margin_delta)}</td>;
  if (col.key === 'avg_erf_variance_log') return <td style={{ textAlign: 'right' }}>{metricText(c.avg_erf_variance_log)}</td>;
  if (col.key === 'avg_spec_norm_log') return <td style={{ textAlign: 'right' }}>{metricText(c.avg_spec_norm_log)}</td>;
  if (col.key === 'avg_icld_velocity') return <td style={{ textAlign: 'right', color: 'var(--text-muted)' }}>{metricText(c.avg_icld_velocity)}</td>;
  if (col.key === 'avg_icld_delta_loss') return <td style={{ textAlign: 'right', color: 'var(--text-muted)' }}>{metricText(c.avg_icld_delta_loss)}</td>;
  if (col.key === 'avg_jacobian_effective_rank') return <td style={{ textAlign: 'right' }}>{metricText(c.avg_jacobian_effective_rank)}</td>;
  if (col.key === 'avg_sensitivity_uniformity') return <td style={{ textAlign: 'right' }}>{metricText(c.avg_sensitivity_uniformity)}</td>;
  if (col.key === 'grad_norm') {
    return (
      <td style={{ textAlign: 'right', fontFamily: 'monospace', fontSize: 11, color: c.grad_norm !== null ? (c.grad_norm > 50000 ? STATUS_COLORS.broken : c.grad_norm > 3000 ? STATUS_COLORS.degraded : 'var(--text-muted)') : 'var(--text-muted)' }}>
        {c.grad_norm !== null ? (c.grad_norm > 1e6 ? `${(c.grad_norm / 1e6).toFixed(1)}M` : c.grad_norm > 1e3 ? `${(c.grad_norm / 1e3).toFixed(1)}K` : c.grad_norm.toFixed(0)) : '-'}
      </td>
    );
  }
  if (col.key === 'fwd_us') return <td style={{ textAlign: 'right', fontFamily: 'monospace', fontSize: 11, color: 'var(--text-muted)' }}>{c.fwd_us != null ? c.fwd_us.toFixed(1) : '-'}</td>;
  if (col.key === 'reasons') {
    return <td style={{ fontSize: 11, color: 'var(--text-muted)', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.top_failure_reason || (c.reasons && c.reasons.length > 0 ? c.reasons.join('; ') : '')}</td>;
  }
  return <td>{c[col.key]}</td>;
}

// ─── Component Health Grid ───
function ComponentGrid({ components, filter, searchTerm, sourceFilter, reasonFilter, sortPreset, columnView, customColumnKeys }) {
  const tableScrollRef = useRef(null);
  const visibleColumns = useMemo(() => visibleComponentColumns(columnView, customColumnKeys), [columnView, customColumnKeys]);
  const groupSpans = useMemo(() => COLUMN_GROUPS.map(group => ({
    ...group,
    span: visibleColumns.filter(col => col.group === group.key).length,
  })).filter(group => group.span > 0), [visibleColumns]);
  const minWidth = Math.max(720, componentTableMinWidth(visibleColumns));

  // Pre-filter rows before passing to the hook (custom filtering not suited to filterRowsByQuery)
  const preFiltered = useMemo(() => {
    let list = components || [];
    const seen = new Set();
    list = list.filter(c => {
      if (seen.has(c.op)) return false;
      seen.add(c.op);
      return true;
    });
    if (filter !== 'all') list = list.filter(c => c.status === filter);
    if (sourceFilter !== 'all') list = list.filter(c => c.data_source === sourceFilter);
    if (reasonFilter !== 'all') list = list.filter(c => componentMatchesReason(c, reasonFilter));
    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      list = list.filter(c => c.op.toLowerCase().includes(q));
    }
    return list;
  }, [components, filter, reasonFilter, searchTerm, sourceFilter]);

  const { sortKey, sortDesc, setSortKey, setSortDesc, sortedRows: filtered, handleSort } = useInteractiveTable({
    rows: preFiltered,
    filterFields: [],
    initialSortKey: 'n_used',
    initialSortDesc: true,
    getSortValue: getComponentSortValue,
    getInitialSortDesc: getComponentInitialSortDesc,
  });

  useEffect(() => {
    const preset = SORT_PRESETS.find(item => item.key === sortPreset);
    if (!preset) return;
    setSortKey(preset.sortKey);
    setSortDesc(preset.desc);
  }, [setSortDesc, setSortKey, sortPreset]);

  useEffect(() => {
    if (!tableScrollRef.current || typeof window === 'undefined') return;
    const saved = Number(window.localStorage.getItem('aria.componentTable.scrollLeft') || 0);
    if (Number.isFinite(saved)) tableScrollRef.current.scrollLeft = saved;
  }, [columnView]);

  const uniqueCount = useMemo(() => {
    const seen = new Set();
    for (const component of components || []) {
      if (component?.op) seen.add(component.op);
    }
    return seen.size;
  }, [components]);

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, marginBottom: 8, fontSize: 11, color: 'var(--text-muted)' }}>
        <span>{filtered.length === 0 ? 'No matching ops' : `${filtered.length} of ${uniqueCount} ops`}</span>
        <span>Sorted by {SORT_PRESETS.find(item => item.key === sortPreset)?.label || sortKey}</span>
      </div>
      {filtered.length === 0 ? (
        <p className="ux-state ux-state-empty">No components match the current filter.</p>
      ) : (
    <div
      ref={tableScrollRef}
      onScroll={(event) => {
        if (typeof window !== 'undefined') window.localStorage.setItem('aria.componentTable.scrollLeft', String(event.currentTarget.scrollLeft));
      }}
      style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}
    >
      <table className="data-table" style={{ fontSize: 12, borderCollapse: 'separate', borderSpacing: 0, minWidth }}>
        <thead style={{ position: 'sticky', top: 0, zIndex: 1, background: 'var(--bg-primary)' }}>
          <tr>
            <th style={stickyCellStyle(SORT_COLUMNS[0], {
              textAlign: 'center',
              fontSize: 10,
              color: 'var(--text-muted)',
              textTransform: 'uppercase',
              letterSpacing: 0,
              background: 'var(--bg-primary)',
              top: 0,
              zIndex: 6,
            })} />
            <th style={stickyCellStyle(SORT_COLUMNS[1], {
              textAlign: 'center',
              fontSize: 10,
              color: 'var(--text-muted)',
              textTransform: 'uppercase',
              letterSpacing: 0,
              background: 'var(--bg-primary)',
              top: 0,
              zIndex: 6,
            })}>
              Identity
            </th>
            {groupSpans.map(group => (
              <th
                key={group.label}
                colSpan={group.span}
                style={{
                  textAlign: 'center',
                  fontSize: 10,
                  color: 'var(--text-muted)',
                  textTransform: 'uppercase',
                  letterSpacing: 0,
                  background: 'var(--bg-primary)',
                  position: 'sticky',
                  top: 0,
                  zIndex: 3,
                }}
              >
                {group.label}
              </th>
            ))}
          </tr>
          <tr>
            {visibleColumns.map(col => (
              <th
                key={col.key}
                title={col.tooltip}
                onClick={() => handleSort(col.key)}
                style={stickyCellStyle(col, {
                  cursor: 'pointer',
                  userSelect: 'none',
                  whiteSpace: 'nowrap',
                  background: 'var(--bg-primary)',
                  top: 34,
                  zIndex: col.sticky ? 5 : 1,
                })}
              >
                {col.label}
                <SortIndicator active={sortKey === col.key} desc={sortDesc} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {filtered.map(c => (
            <tr key={c.op} style={{
              background: c.status === 'broken' ? '#ef444408' : c.status === 'degraded' ? '#eab30808' : undefined,
            }}>
              {visibleColumns.map(col => <React.Fragment key={col.key}>{renderComponentCell(c, col)}</React.Fragment>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
      )}
    </>
  );
}

// ─── Failure Blocklist ───
function FailureBlocklist({ blocklist }) {
  if (!blocklist || Object.keys(blocklist).length === 0) return null;
  const entries = Object.entries(blocklist).sort((a, b) => a[1] - b[1]).slice(0, 20);
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Failure Penalties (auto-deweighted op pairs)</div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table" style={{ fontSize: 12 }}>
          <thead><tr><th>Op Pair Signature</th><th>Penalty</th></tr></thead>
          <tbody>
            {entries.map(([sig, penalty]) => (
              <tr key={sig}>
                <td style={{ fontFamily: 'monospace' }}>{sig}</td>
                <td style={{
                  textAlign: 'right', fontWeight: 600,
                  color: penalty <= 0.05 ? STATUS_COLORS.broken : STATUS_COLORS.degraded,
                }}>
                  {`${(penalty * 100).toFixed(0)}%`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Op Pair Heatmap ───
const PAIR_COLUMNS = [
  { key: 'op_a', label: 'Op A' },
  { key: 'op_b', label: 'Op B' },
  { key: 'n', label: 'Count' },
  { key: 's0_rate', label: 'S0 Rate' },
  { key: 's1_rate', label: 'S1 Rate' },
];

function OpPairHeatmap({ pairs }) {
  const { sortKey, sortDesc, sortedRows: sorted, handleSort } = useInteractiveTable({
    rows: pairs || [],
    filterFields: [],
    initialSortKey: 'n',
    initialSortDesc: true,
    getInitialSortDesc: getPairInitialSortDesc,
  });

  if (!sorted || sorted.length === 0) return null;

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Top Op Pairs (co-occurrence)</div>
      <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
        <table className="data-table" style={{ fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0, zIndex: 1, background: 'var(--bg-primary)' }}>
            <tr>
              {PAIR_COLUMNS.map(col => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap', background: 'var(--bg-primary)' }}
                >
                  {col.label}
                  <SortIndicator active={sortKey === col.key} desc={sortDesc} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map(p => {
              const s0Color = p.s0_rate < 0.3 ? STATUS_COLORS.broken : p.s0_rate < 0.6 ? STATUS_COLORS.degraded : '#22c55e';
              return (
                <tr key={`${p.op_a}-${p.op_b}`}>
                  <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{p.op_a}</td>
                  <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{p.op_b}</td>
                  <td style={{ textAlign: 'right' }}>{p.n}</td>
                  <td style={{ textAlign: 'right', color: s0Color }}>{(p.s0_rate * 100).toFixed(0)}%</td>
                  <td style={{ textAlign: 'right', color: p.s1_rate < 0.05 ? STATUS_COLORS.broken : 'var(--text-primary)' }}>
                    {(p.s1_rate * 100).toFixed(1)}%
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

// ─── Loss Distribution Panel (CSS box plots) ───
function LossDistributionPanel({ distributions }) {
  if (!distributions || distributions.length === 0) return null;
  const globalMax = Math.max(...distributions.map(d => d.max), 1.5);

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Loss Distribution by Op</div>
      <div style={{ maxHeight: 400, overflowY: 'auto' }}>
        {distributions.slice(0, 30).map(d => {
          const scale = (v) => `${Math.min((v / globalMax) * 100, 100)}%`;
          return (
            <div key={d.op} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ width: 120, fontSize: 11, fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{d.op}</span>
              <div style={{ flex: 1, position: 'relative', height: 16, background: 'var(--bg-tertiary)', borderRadius: 4 }}>
                {/* Whiskers min-max */}
                <div style={{
                  position: 'absolute', top: 7, height: 2, background: 'var(--text-muted)',
                  left: scale(d.min), width: `calc(${scale(d.max)} - ${scale(d.min)})`,
                  opacity: 0.4,
                }} />
                {/* Box q1-q3 */}
                <div style={{
                  position: 'absolute', top: 2, height: 12, borderRadius: 2,
                  background: 'var(--accent-blue)', opacity: 0.5,
                  left: scale(d.q1), width: `calc(${scale(d.q3)} - ${scale(d.q1)})`,
                }} />
                {/* Median line */}
                <div style={{
                  position: 'absolute', top: 1, height: 14, width: 2,
                  background: '#fff', borderRadius: 1, left: scale(d.median),
                }} />
              </div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)', minWidth: 50, textAlign: 'right' }}>
                n={d.n}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Grammar Evolution Panel ───
function GrammarEvolutionPanel({ events }) {
  if (!events || events.length === 0) return null;
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Grammar Evolution</div>
      <div style={{ maxHeight: 300, overflowY: 'auto' }}>
        {events.map(e => (
          <div key={e.id} style={{ padding: '6px 0', borderBottom: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              {new Date(e.timestamp * 1000).toLocaleString()}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-primary)', marginTop: 2 }}>
              {e.description?.slice(0, 100)}
            </div>
            {e.changes && Object.keys(e.changes).length > 0 && (
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 4 }}>
                {Object.entries(e.changes).slice(0, 8).map(([op, ch]) => (
                  <span key={op} style={{
                    padding: '1px 6px', borderRadius: 4, fontSize: 10,
                    background: ch.new > ch.old ? '#22c55e22' : '#ef444422',
                    color: ch.new > ch.old ? '#22c55e' : '#ef4444',
                  }}>
                    {op}: {Number(ch.old).toFixed(2)} &rarr; {Number(ch.new).toFixed(2)}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Failure Pattern Panel ───
function FailurePatternPanel({ patterns }) {
  if (!patterns || patterns.length === 0) return null;
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Failure Patterns</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {patterns.slice(0, 10).map(p => (
          <div key={p.error_type} style={{
            padding: '8px 12px', borderRadius: 6, background: '#ef444408',
            borderLeft: '3px solid #ef4444',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>{p.error_type}</span>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{p.count} occurrences</span>
            </div>
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 4 }}>
              {p.top_ops.map(op => (
                <span key={op.op} style={{
                  padding: '1px 6px', borderRadius: 4, fontSize: 10,
                  background: 'var(--bg-tertiary)', color: 'var(--text-muted)',
                }}>
                  {op.op} ({op.occurrences})
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Leaderboard Dynamics Panel ───
function LeaderboardDynamicsPanel({ daily, recentPromotions }) {
  if ((!daily || Object.keys(daily).length === 0) && (!recentPromotions || recentPromotions.length === 0)) return null;

  const tierColors = TIER_COLORS;

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Leaderboard Dynamics</div>
      {daily && Object.keys(daily).length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Daily tier counts</div>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table" style={{ fontSize: 11 }}>
              <thead>
                <tr><th>Date</th><th>Screening</th><th>Investigation</th><th>Validation</th><th>Breakthrough</th></tr>
              </thead>
              <tbody>
                {Object.entries(daily).slice(-10).map(([day, tiers]) => (
                  <tr key={day}>
                    <td>{day}</td>
                    {['screening', 'investigation', 'validation', 'breakthrough'].map(t => (
                      <td key={t} style={{ textAlign: 'right', color: tierColors[t] || 'var(--text-primary)' }}>{tiers[t] || 0}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {recentPromotions && recentPromotions.length > 0 && (
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Recent entries</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {recentPromotions.slice(0, 8).map(p => (
              <div key={p.entry_id} style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 12 }}>
                <span style={{
                  padding: '1px 6px', borderRadius: 4, fontSize: 10, fontWeight: 600,
                  background: (tierColors[p.tier] || '#888') + '22',
                  color: tierColors[p.tier] || '#888',
                }}>{p.tier}</span>
                <span style={{ fontFamily: 'monospace', fontSize: 11, color: 'var(--text-muted)' }}>{p.result_id?.slice(0, 12)}</span>
                {p.composite_score != null && (
                  <span style={{ fontSize: 11 }}>score: {p.composite_score.toFixed(3)}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Insight Effectiveness Panel ───
function InsightEffectivenessPanel({ insights }) {
  if (!insights || insights.length === 0) return null;
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Insight Effectiveness</div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table" style={{ fontSize: 12 }}>
          <thead>
            <tr><th>Type</th><th>Subject</th><th>Predictions</th><th>Accuracy</th><th>Bayesian Mean</th><th>Status</th></tr>
          </thead>
          <tbody>
            {insights.slice(0, 20).map(ins => (
              <tr key={ins.insight_id}>
                <td style={{ fontSize: 11 }}>{ins.insight_type || ins.category}</td>
                <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{ins.subject_key?.slice(0, 20) || '-'}</td>
                <td style={{ textAlign: 'right' }}>{ins.n_predictions}</td>
                <td style={{
                  textAlign: 'right', fontWeight: 600,
                  color: ins.accuracy > 0.6 ? '#22c55e' : ins.accuracy > 0.3 ? '#eab308' : '#ef4444',
                }}>
                  {(ins.accuracy * 100).toFixed(0)}%
                </td>
                <td style={{ textAlign: 'right' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4, justifyContent: 'flex-end' }}>
                    <div style={{ width: 40, height: 6, borderRadius: 3, background: 'var(--bg-tertiary)', overflow: 'hidden' }}>
                      <div style={{
                        width: `${(ins.bayesian_mean * 100).toFixed(0)}%`, height: '100%',
                        background: ins.bayesian_mean > 0.5 ? '#22c55e' : '#eab308', borderRadius: 3,
                      }} />
                    </div>
                    <span style={{ fontSize: 11 }}>{ins.bayesian_mean.toFixed(2)}</span>
                  </div>
                </td>
                <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{ins.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StructuralDiagnosticsPanel({ data }) {
  if (!data) return null;
  const topTemplates = Array.isArray(data.top_templates) ? data.top_templates.slice(0, 5) : [];
  const strugglingTemplates = Array.isArray(data.struggling_templates) ? data.struggling_templates.slice(0, 5) : [];
  const weakSlots = Array.isArray(data.slot_observability) ? data.slot_observability.slice(0, 6) : [];
  const templateTrends = Array.isArray(data.template_trends) ? data.template_trends.slice(0, 3) : [];
  const slotTrends = Array.isArray(data.slot_trends) ? data.slot_trends.slice(0, 3) : [];
  const lossTrends = Array.isArray(data.loss_trends) ? data.loss_trends : [];
  const recommendations = Array.isArray(data.recommendations) ? data.recommendations : [];
  const loss = data.loss_distribution || {};
  const summary = data.summary || {};

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Structural Diagnostics</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Template and slot-level observability for the search grammar. This highlights which structural recipes survive, which slots are collapsing quality, and whether validation loss is drifting above training loss.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 10, marginBottom: 16 }}>
        <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }}>{Number(summary.templates_tracked || 0)}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Templates tracked</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>{Number(summary.avg_templates_per_graph || 0).toFixed(2)} / graph</div>
        </div>
        <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }}>{Number(summary.avg_motifs_per_graph || 0).toFixed(2)}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Avg motifs / graph</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>{Number(summary.motifs_tracked || 0)} motifs tracked</div>
        </div>
        <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent-blue)' }}>{fmtLoss(loss.training?.median)}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Median train LR</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>P75 {fmtLoss(loss.training?.p75)}</div>
        </div>
        <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent-yellow)' }}>{fmtLoss(loss.validation?.median)}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Median val LR</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>P75 {fmtLoss(loss.validation?.p75)}</div>
        </div>
        <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent-blue)' }}>{Number(summary.routing_fast_lane_templates || 0)}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Routing fast-lane templates</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>
            {Number(summary.routing_fast_lane_positive_templates || 0)} positive slow starters
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>Highest Success Templates</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {topTemplates.length > 0 ? topTemplates.map((row) => (
              <div key={row.name} style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>{row.name}</span>
                  <span style={{ fontSize: 12, color: '#22c55e', fontWeight: 700 }}>{fmtPct(row.s1_rate)}</span>
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>
                  {row.n_used} runs · val LR {fmtLoss(row.avg_validation_loss_ratio ?? row.avg_loss_ratio)} · best {fmtLoss(row.best_loss_ratio)}
                  {row.routing_fast_lane_runs ? ` · fast lane ${fmtPct(row.routing_fast_lane_positive_rate)}` : ''}
                </div>
              </div>
            )) : <div className="ux-state ux-state-empty">No template diagnostics yet.</div>}
          </div>
        </div>

        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>Templates To Fix</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {strugglingTemplates.length > 0 ? strugglingTemplates.map((row) => (
              <div key={row.name} style={{ padding: '8px 10px', borderRadius: 6, background: '#ef444408', border: '1px solid #ef444422' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>{row.name}</span>
                  <span style={{ fontSize: 12, color: '#ef4444', fontWeight: 700 }}>{fmtPct(row.s1_rate)}</span>
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>
                  {row.n_used} runs · slots {row.slot_count || 0} · fail {row.top_failure_reason || 'unknown'} · avg LR {fmtLoss(row.avg_validation_loss_ratio ?? row.avg_loss_ratio)}
                  {row.routing_fast_lane_runs ? ` · fast lane ${fmtPct(row.routing_fast_lane_positive_rate)}` : ''}
                </div>
              </div>
            )) : <div className="ux-state ux-state-empty">No struggling templates identified.</div>}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.15fr 0.85fr', gap: 16 }}>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>Weakest Slots</div>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table" style={{ fontSize: 11 }}>
              <thead>
                <tr><th>Slot</th><th>Template</th><th>S1</th><th>Avg LR</th><th>Motif</th><th>Failure</th></tr>
              </thead>
              <tbody>
                {weakSlots.map((row) => (
                  <tr key={row.slot_key}>
                    <td style={{ fontFamily: 'monospace' }}>{row.slot_key}</td>
                    <td>{row.template_name}</td>
                    <td style={{ textAlign: 'right', color: (row.s1_rate || 0) < 0.15 ? '#ef4444' : '#eab308' }}>{fmtPct(row.s1_rate)}</td>
                    <td style={{ textAlign: 'right' }}>{fmtLoss(row.avg_loss_ratio)}</td>
                    <td style={{ fontFamily: 'monospace' }}>{row.top_selected_motif || '-'}</td>
                    <td>{row.top_failure_reason || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {Array.isArray(summary.zero_slot_templates) && summary.zero_slot_templates.length > 0 && (
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 8 }}>
              Zero-slot templates: {summary.zero_slot_templates.slice(0, 6).join(', ')}
              {summary.zero_slot_templates.length > 6 ? ` +${summary.zero_slot_templates.length - 6}` : ''}
            </div>
          )}
        </div>

        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>Recommended Fixes</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {recommendations.length > 0 ? recommendations.map((item, idx) => (
              <div key={idx} style={{ padding: '9px 10px', borderRadius: 6, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', fontSize: 12, color: 'var(--text-primary)', lineHeight: 1.5 }}>
                {item}
              </div>
            )) : <div className="ux-state ux-state-empty">No structural recommendations yet.</div>}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 18 }}>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>Template Success Trends</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {templateTrends.length > 0 ? templateTrends.map((series) => (
              <div key={series.name} style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
                <MiniChart
                  data={series.points}
                  valueKey="s1_rate"
                  label={`${series.name} S1`}
                  color="#22c55e"
                  formatValue={(v) => `${(Number(v) * 100).toFixed(1)}%`}
                  scaleKey="pass_rate"
                  windowSize={12}
                />
              </div>
            )) : <div className="ux-state ux-state-empty">Need more experiments for template trends.</div>}
          </div>
        </div>

        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>Weak Slot Trends</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {slotTrends.length > 0 ? slotTrends.map((series) => (
              <div key={series.slot_key} style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
                <MiniChart
                  data={series.points}
                  valueKey="s1_rate"
                  label={`${series.slot_key} S1`}
                  color="#ef4444"
                  formatValue={(v) => `${(Number(v) * 100).toFixed(1)}%`}
                  scaleKey="pass_rate"
                  windowSize={12}
                />
              </div>
            )) : <div className="ux-state ux-state-empty">Need more experiments for slot trends.</div>}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 18 }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>Loss Drift Trends</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
          <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
            <MiniChart
              data={lossTrends}
              valueKey="training_median"
              label="Median Train LR"
              color="#58a6ff"
              formatValue={(v) => Number(v).toFixed(3)}
              scaleKey="loss_ratio"
              windowSize={12}
            />
          </div>
          <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
            <MiniChart
              data={lossTrends}
              valueKey="validation_median"
              label="Median Val LR"
              color="#d29922"
              formatValue={(v) => Number(v).toFixed(3)}
              scaleKey="loss_ratio"
              windowSize={12}
            />
          </div>
          <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
            <MiniChart
              data={lossTrends}
              valueKey="discovery_median"
              label="Median Discovery LR"
              color="#a855f7"
              formatValue={(v) => Number(v).toFixed(3)}
              scaleKey="loss_ratio"
              windowSize={12}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Main Component Analytics Dashboard ───
export default function ComponentAnalyticsDashboard() {
  const [health, setHealth] = useState(null);
  const [blocklist, setBlocklist] = useState({});
  const [opPairs, setOpPairs] = useState([]);
  const [lossDist, setLossDist] = useState([]);
  const [grammarEvents, setGrammarEvents] = useState([]);
  const [failurePatterns, setFailurePatterns] = useState([]);
  const [leaderboardData, setLeaderboardData] = useState({ daily: {}, recent_promotions: [] });
  const [insightData, setInsightData] = useState([]);
  const [filter, setFilter] = usePersistentState('aria.componentTable.filter', 'all');
  const [reasonFilter, setReasonFilter] = usePersistentState('aria.componentTable.reasonFilter', 'all');
  const [sourceFilter, setSourceFilter] = usePersistentState('aria.componentTable.sourceFilter', 'all');
  const [timeWindow, setTimeWindow] = usePersistentState('aria.componentTable.timeWindow', 'all');
  const [sortPreset, setSortPreset] = usePersistentState('aria.componentTable.sortPreset', 'most_used');
  const [columnView, setColumnView] = usePersistentState('aria.componentTable.columnView', 'triage');
  const [customColumnKeys, setCustomColumnKeys] = usePersistentState('aria.componentTable.customColumns', null);
  const [showColumnPicker, setShowColumnPicker] = useState(false);
  const [searchTerm, setSearchTerm] = usePersistentState('aria.componentTable.searchTerm', '');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const { slowPollTick } = useAriaData();
  const hasComponentFilters = filter !== 'all' || reasonFilter !== 'all' || sourceFilter !== 'all' || timeWindow !== 'all' || sortPreset !== 'most_used' || columnView !== 'triage' || Boolean(customColumnKeys) || searchTerm.trim() !== '';

  const fetchHealthOnly = useCallback(async () => {
    const windowParam = timeWindow !== 'all' ? `?window=${timeWindow}` : '';
    const healthRes = await apiCall(`/api/observability/health${windowParam}`);
    if (healthRes.ok) {
      setHealth(await healthRes.json());
    }
  }, [timeWindow]);

  const fetchData = useCallback(async ({ includeHeavy = true } = {}) => {
    try {
      if (!includeHeavy) {
        await fetchHealthOnly();
        return;
      }
      const windowParam = timeWindow !== 'all' ? `?window=${timeWindow}` : '';
      const [healthRes, blockRes, pairRes, lossRes, gramRes, failRes, lbRes, insRes] = await Promise.all([
        apiCall(`/api/observability/health${windowParam}`),
        apiCall('/api/observability/failure-blocklist'),
        apiCall('/api/observability/op-pairs'),
        apiCall('/api/observability/loss-distribution'),
        apiCall('/api/observability/grammar-evolution'),
        apiCall('/api/observability/failure-patterns'),
        apiCall('/api/observability/leaderboard-dynamics?trusted_only=0'),
        apiCall('/api/observability/insight-effectiveness'),
      ]);
      if (healthRes.ok) setHealth(await healthRes.json());
      if (blockRes.ok) { const d = await blockRes.json(); setBlocklist(d.blocklist || {}); }
      if (pairRes.ok) { const d = await pairRes.json(); setOpPairs(d.pairs || []); }
      if (lossRes.ok) { const d = await lossRes.json(); setLossDist(d.distributions || []); }
      if (gramRes.ok) { const d = await gramRes.json(); setGrammarEvents(d.events || []); }
      if (failRes.ok) { const d = await failRes.json(); setFailurePatterns(d.patterns || []); }
      if (lbRes.ok) setLeaderboardData(await lbRes.json());
      if (insRes.ok) { const d = await insRes.json(); setInsightData(d.insights || []); }
    } catch (err) {
      console.error('ComponentAnalytics fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [fetchHealthOnly, timeWindow]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await apiCall('/api/observability/health/refresh', { method: 'POST' });
      await fetchData({ includeHeavy: true });
    } finally {
      setRefreshing(false);
    }
  }, [fetchData]);

  useEffect(() => {
    fetchData({ includeHeavy: true });
  }, [fetchData, timeWindow]);

  useEffect(() => {
    if (loading) {
      return;
    }
    fetchData({ includeHeavy: false });
  }, [fetchData, loading, slowPollTick]);

  if (loading) {
    return <div className="card"><p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Loading component analytics...</p></div>;
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>Component Analytics</h2>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <select value={timeWindow} onChange={e => setTimeWindow(e.target.value)} style={{
            padding: '4px 8px', fontSize: 11, borderRadius: 6,
            background: 'var(--bg-tertiary)', color: 'var(--text-primary)',
            border: '1px solid var(--border-color)', cursor: 'pointer',
          }}>
            {TIME_WINDOWS.map(w => (
              <option key={w.value} value={w.value}>{w.label}</option>
            ))}
          </select>
          <select value={sourceFilter} onChange={e => setSourceFilter(e.target.value)} style={{
            padding: '4px 8px', fontSize: 11, borderRadius: 6,
            background: 'var(--bg-tertiary)', color: 'var(--text-primary)',
            border: '1px solid var(--border-color)', cursor: 'pointer',
          }}>
            <option value="all">All sources</option>
            <option value="search">Search only</option>
            <option value="search+profiling">Search+Profiling</option>
            <option value="profiling_only">Profiling only</option>
          </select>
          <select value={sortPreset} onChange={e => setSortPreset(e.target.value)} style={{
            padding: '4px 8px', fontSize: 11, borderRadius: 6,
            background: 'var(--bg-tertiary)', color: 'var(--text-primary)',
            border: '1px solid var(--border-color)', cursor: 'pointer',
          }}>
            {SORT_PRESETS.map(preset => (
              <option key={preset.key} value={preset.key}>{preset.label}</option>
            ))}
          </select>
          <select value={columnView} onChange={e => { setColumnView(e.target.value); setCustomColumnKeys(null); }} style={{
            padding: '4px 8px', fontSize: 11, borderRadius: 6,
            background: 'var(--bg-tertiary)', color: 'var(--text-primary)',
            border: '1px solid var(--border-color)', cursor: 'pointer',
          }}>
            {COMPONENT_VIEW_PRESETS.map(preset => (
              <option key={preset.key} value={preset.key}>{preset.label}</option>
            ))}
          </select>
          <button onClick={() => {
            if (!customColumnKeys) setCustomColumnKeys(presetComponentColumns(columnView).map(col => col.key));
            setShowColumnPicker(value => !value);
          }} style={{
            padding: '4px 10px', fontSize: 11, borderRadius: 6,
            background: showColumnPicker ? 'var(--accent-blue)22' : 'var(--bg-secondary)',
            color: showColumnPicker ? 'var(--accent-blue)' : 'var(--text-primary)',
            border: `1px solid ${showColumnPicker ? 'var(--accent-blue)' : 'var(--border-color)'}`,
            cursor: 'pointer',
          }}>
            Columns
          </button>
          <button onClick={handleRefresh} disabled={refreshing} style={{
            padding: '4px 12px', fontSize: 12, borderRadius: 6,
            background: 'var(--bg-secondary)', color: 'var(--text-primary)',
            border: '1px solid var(--border-color)', cursor: 'pointer',
            opacity: refreshing ? 0.5 : 1,
          }}>
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </div>

      <HealthSummary health={health} />

      {/* Component Grid with filters */}
      <div className="card" style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
          <div className="card-title" style={{ margin: 0 }}>Component Health Grid</div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <input type="text" placeholder="Search ops..." value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              style={{
                padding: '4px 10px', fontSize: 12, borderRadius: 6, width: 160,
                background: 'var(--bg-tertiary)', color: 'var(--text-primary)',
                border: '1px solid var(--border-color)', outline: 'none',
              }}
            />
            {['all', 'broken', 'degraded', 'healthy'].map(f => (
              <button key={f} onClick={() => setFilter(f)} style={{
                padding: '3px 10px', fontSize: 11, borderRadius: 12, cursor: 'pointer',
                background: filter === f ? (STATUS_COLORS[f] || 'var(--accent-blue)') + '22' : 'var(--bg-tertiary)',
                color: filter === f ? (STATUS_COLORS[f] || 'var(--accent-blue)') : 'var(--text-muted)',
                border: `1px solid ${filter === f ? (STATUS_COLORS[f] || 'var(--accent-blue)') + '44' : 'var(--border-color)'}`,
                fontWeight: filter === f ? 600 : 400, textTransform: 'capitalize',
              }}>
                {f}{f !== 'all' && health ? ` (${health[f] || 0})` : ''}
              </button>
            ))}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
          {REASON_FILTERS.map(f => (
            <button key={f.key} onClick={() => setReasonFilter(f.key)} style={{
              padding: '3px 10px', fontSize: 11, borderRadius: 12, cursor: 'pointer',
              background: reasonFilter === f.key ? 'var(--accent-blue)22' : 'var(--bg-tertiary)',
              color: reasonFilter === f.key ? 'var(--accent-blue)' : 'var(--text-muted)',
              border: `1px solid ${reasonFilter === f.key ? 'var(--accent-blue)' : 'var(--border-color)'}`,
              fontWeight: reasonFilter === f.key ? 600 : 400,
            }}>
              {f.label}
            </button>
          ))}
          <button
            onClick={() => {
              setFilter('all');
              setReasonFilter('all');
              setSourceFilter('all');
              setTimeWindow('all');
              setSortPreset('most_used');
              setColumnView('triage');
              setCustomColumnKeys(null);
              setSearchTerm('');
            }}
            disabled={!hasComponentFilters}
            style={{
              padding: '3px 10px', fontSize: 11, borderRadius: 12,
              background: 'var(--bg-secondary)', color: hasComponentFilters ? 'var(--text-primary)' : 'var(--text-muted)',
              border: '1px solid var(--border-color)', cursor: hasComponentFilters ? 'pointer' : 'not-allowed',
            }}
          >
            Reset
          </button>
        </div>
        {showColumnPicker && (
          <ColumnPickerPanel
            columns={SORT_COLUMNS}
            selectedKeys={customColumnKeys || presetComponentColumns(columnView).map(col => col.key)}
            onChange={setCustomColumnKeys}
            onReset={() => setCustomColumnKeys(null)}
          />
        )}
        <ComponentGrid
          components={health?.components}
          filter={filter}
          searchTerm={searchTerm}
          sourceFilter={sourceFilter}
          reasonFilter={reasonFilter}
          sortPreset={sortPreset}
          columnView={columnView}
          customColumnKeys={customColumnKeys}
        />
      </div>

      <OpPairHeatmap pairs={opPairs} />
      <LossDistributionPanel distributions={lossDist} />
      <FailureBlocklist blocklist={blocklist} />
      <FailurePatternPanel patterns={failurePatterns} />
      <GrammarEvolutionPanel events={grammarEvents} />
      <LeaderboardDynamicsPanel daily={leaderboardData.daily} recentPromotions={leaderboardData.recent_promotions} />
      <InsightEffectivenessPanel insights={insightData} />
    </div>
  );
}
