import { apiCall, postJson } from "../services/apiService";
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { SCORE_MAX, scoreColor, scoreGradient, scoreToneLabel } from '../utils/format';
import { blimpColor, hellaswagColor, lossColor, noveltyColor, pplColor, probeAucColor } from '../utils/colors';
import { useAriaData } from '../hooks/useAriaData';
import { TIER_COLORS, TIER_LABELS, TIER_ORDER, bestLoss, percentOfReference } from '../utils/scoringEngine';
import { DISCOVERY_TIER_FILTERS, capabilityQualityRank, capabilityQualityStatus, getDiscoveryDisplayStatus, matchesActiveTier } from '../utils/discoveryStatus';
import {
  ExpandedDetailPanel,
  FingerprintLeaderboardChart,
  ScoreCell,
  StatusBadge,
  SummaryBar,
} from './discoveries/DiscoveryUiBits';
import SortIndicator from './shared/SortIndicator';

const DISCOVERIES_PREFS_KEY = 'aria_discoveries_prefs_v1';
const QUALITY_FLOOR_THRESHOLD = 0.8;
const DEFAULT_ACTIVE_TIER = 'all';
const DEFAULT_SHOW_REFERENCES = true;
const DEFAULT_HIDE_FAILED = true;
const DEFAULT_QUALITY_FLOOR_ENABLED = true;
const DEFAULT_SOURCE_FILTER = 'trusted';
const DEFAULT_CAPABILITY_FILTER = 'all';

const SOURCE_FILTER_LABELS = {
  trusted: 'Trusted ranked',
  backlog: 'Backlog',
  all_graphs: 'All graphs',
  all: 'Mixed trust ranked',
  untrusted: 'Untrusted',
  backfill: 'Backfill',
  replay: 'Replay',
};

const CAPABILITY_FILTER_LABELS = {
  all: 'All quality states',
  qualified: 'Capability-Qualified',
  training_only: 'Training-Only',
  pending: 'Validation Pending',
};

const FILTER_PANEL_STYLE = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  flexWrap: 'wrap',
  padding: '8px 10px',
  border: '1px solid var(--border)',
  borderRadius: 8,
  background: 'var(--bg-secondary)',
  minHeight: 44,
};

const FILTER_PANEL_TITLE_STYLE = {
  fontSize: 11,
  fontWeight: 600,
  color: 'var(--text-muted)',
  marginRight: 2,
};
const PIN_COLUMN_WIDTH = 26;
const RANK_COLUMN_WIDTH = 44;
const STATUS_COLUMN_WIDTH = 152;
const ACTION_COLUMN_WIDTH = 84;

function toggleButtonStyle(active, activeColor, activeBackground) {
  return {
    fontSize: 11,
    padding: '5px 12px',
    cursor: 'pointer',
    background: active ? activeBackground : 'transparent',
    border: `1px solid ${active ? activeColor : 'var(--border)'}`,
    borderRadius: 4,
    color: active ? activeColor : 'var(--text-secondary)',
  };
}

function provenanceBucket(entry) {
  const cohort = String(entry?.result_cohort || '').trim().toLowerCase();
  const experimentType = String(entry?.experiment_type || '').trim().toLowerCase();
  const trustLabel = String(entry?.trust_label || '').trim().toLowerCase();
  const comparabilityLabel = String(entry?.comparability_label || '').trim().toLowerCase();

  if (experimentType === 'exact_graph_replay' || cohort === 'exact_graph_replay') {
    return 'replay';
  }
  if (
    cohort === 'backfill'
    || trustLabel === 'backfill_observation'
    || comparabilityLabel === 'reconstructed_init_variant'
  ) {
    return 'backfill';
  }
  if (['candidate_screening', 'candidate_grade', 'reference'].includes(trustLabel)) {
    return 'trusted';
  }
  return 'untrusted';
}

function rowBackgrounds({ index, isHighlighted, isPinnedReference, isExpanded, tier, score }) {
  const numericScore = Number(score);
  if (isPinnedReference) {
    return {
      base: 'color-mix(in srgb, var(--table-row-bg) 88%, var(--score-reference))',
      hover: 'color-mix(in srgb, var(--table-row-bg) 83%, var(--score-reference))',
    };
  }
  if (isHighlighted) {
    return {
      base: 'color-mix(in srgb, var(--table-row-bg) 80%, var(--accent-blue))',
      hover: 'color-mix(in srgb, var(--table-row-bg) 74%, var(--accent-blue))',
    };
  }
  if (isExpanded) {
    return {
      base: 'color-mix(in srgb, var(--table-row-bg) 88%, var(--accent-blue))',
      hover: 'color-mix(in srgb, var(--table-row-bg) 84%, var(--accent-blue))',
    };
  }
  if (Number.isFinite(numericScore) && numericScore >= 245) {
    return {
      base: 'color-mix(in srgb, var(--table-row-bg) 91%, var(--score-champion))',
      hover: 'color-mix(in srgb, var(--table-row-bg) 86%, var(--score-champion))',
    };
  }
  if (Number.isFinite(numericScore) && numericScore >= 205) {
    return {
      base: 'color-mix(in srgb, var(--table-row-bg) 93%, var(--score-elite))',
      hover: 'color-mix(in srgb, var(--table-row-bg) 88%, var(--score-elite))',
    };
  }
  if (Number.isFinite(numericScore) && numericScore >= 150) {
    return {
      base: 'color-mix(in srgb, var(--table-row-bg) 94%, var(--score-reference))',
      hover: 'color-mix(in srgb, var(--table-row-bg) 89%, var(--score-reference))',
    };
  }
  if (tier === 'breakthrough') {
    return {
      base: 'color-mix(in srgb, var(--table-row-bg) 92%, var(--score-elite))',
      hover: 'color-mix(in srgb, var(--table-row-bg) 87%, var(--score-elite))',
    };
  }
  return {
    base: index % 2 === 1 ? 'var(--table-row-alt-bg)' : 'var(--table-row-bg)',
    hover: 'var(--table-row-hover-bg)',
  };
}

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

function finiteOrNull(value) {
  if (value == null) return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function metricDisplay(value, decimals = 3) {
  const num = finiteOrNull(value);
  if (num == null) return '--';
  if (num !== 0 && Math.abs(num) < 0.0001) return num.toExponential(2);
  return num.toFixed(decimals);
}

function fingerprintTone(key, value) {
  const num = finiteOrNull(value);
  if (num == null) return 'var(--text-muted)';
  if (key === 'fp_jacobian_erf_density' && num >= 0.18) return 'var(--accent-green)';
  if (key === 'fp_id_collapse_rate' && num >= 0.10) return 'var(--accent-green)';
  if (key === 'fp_id_collapse_rate_normalized' && num >= 0.10) return 'var(--accent-green)';
  if (key === 'fp_jacobian_erf_decay_slope' && num >= 0.20) return 'var(--accent-green)';
  if (key === 'fp_icld_velocity' || key === 'fp_icld_delta_loss') return 'var(--text-muted)';
  return 'var(--text-primary)';
}

function scoreCellTone(key, value) {
  if (key === 'wikitext_perplexity') return pplColor(value);
  if (key === 'hellaswag_acc') return hellaswagColor(value);
  if (key === 'blimp_overall_accuracy') return blimpColor(value);
  if (
    key === 'induction_auc'
    || key === 'induction_v2_investigation_auc'
    || key === 'binding_auc'
    || key === 'binding_v2_investigation_auc'
    || key === 'binding_composite'
    || key === 'ar_auc'
  ) {
    return probeAucColor(value);
  }
  return 'var(--text-primary)';
}

// ── Main Component ─────────────────────────────────────────────────

const COLUMNS = [
  { key: '_score', label: 'Composite Score', width: 124, title: `Canonical post-BPE + understanding composite used for ranking discoveries. Color bands are fixed percentages of the ${SCORE_MAX}-point v10 rubric ceiling.` },
  { key: '_capability_quality', label: 'Capability', width: 134, title: 'Quality state separate from workflow stage: Capability-Qualified, Training-Only, Validation Pending, etc.' },
  { key: 'display_name', label: 'Architecture', width: 240, title: 'Human-readable name or fingerprint of the model topology.' },
  { key: 'architecture_family', label: 'Family', width: 120, title: 'The architectural category (e.g., Attention, SSM, Hybrid).' },
  { key: 'discovery_loss_ratio', label: 'Disc Loss', width: 92, title: 'Loss ratio on random tokens (fast triage).' },
  { key: 'validation_loss_ratio', label: 'Val Loss', width: 92, title: 'Loss ratio on real micro-corpus (true causal performance).' },
  { key: '_best_loss', label: 'Best', width: 84, title: 'The lowest loss ratio achieved by this architecture across all tests.' },
  { key: '_vs_ref', label: 'vs Ref', width: 84, title: 'How this model compares to the GPT-2 baseline (lower % is better).' },
  { key: '_novelty', label: 'Novelty', width: 78, title: 'Measures how unique this model is compared to existing designs.' },
  { key: 'param_efficiency', label: 'P Eff', width: 78, title: 'Parameter efficiency: FLOPs per parameter (higher = parameters are used more efficiently).' },
  { key: 'sample_efficiency', label: 'S Eff', width: 78, title: 'How quickly the model converges to 25% of initial loss (1.0 = instant, 0.0 = never).' },
  { key: 'investigation_robustness', label: 'Robust', width: 82, title: 'Consistency across different training recipes (higher is more stable).' },
  { key: 'robustness_long_ctx_score', label: 'LongCtx', width: 82, title: 'Combined long-context score used in final evaluation.' },
  { key: 'robustness_long_ctx_scaling_score', label: 'LC-Scale', width: 82, title: 'Long-context scaling component score.' },
  { key: 'robustness_long_ctx_assoc_score', label: 'LC-Assoc', width: 82, title: 'Associative retrieval benchmark score.' },
  { key: 'robustness_long_ctx_multi_hop_score', label: 'LC-MHop', width: 82, title: 'Multi-hop retrieval benchmark score.' },
  { key: 'robustness_long_ctx_passkey_score', label: 'LC-Key', width: 82, title: 'Zero-shot passkey retrieval benchmark score.' },
  { key: 'robustness_long_ctx_retrieval_aggregate', label: 'LC-Retr', width: 82, title: 'Aggregate retrieval score across long-context benchmarks.' },
  { key: 'max_viable_seq_len', label: 'MaxLen', width: 86, title: 'Maximum viable sequence length from long-context scaling sweep.' },
  { key: 'jacobian_spectral_norm', label: 'Spectral', width: 82, title: 'Jacobian Spectral Norm: stability of gradient propagation (lower is better).' },
  { key: 'fp_jacobian_effective_rank', label: 'JRank', width: 78, title: 'Jacobian effective rank from the fingerprint pass.' },
  { key: 'fp_sensitivity_uniformity', label: 'SensUnif', width: 86, title: 'Sensitivity uniformity from fingerprinting.' },
  { key: 'fp_jacobian_erf_density', label: 'ERF Dens', width: 88, title: 'Effective receptive-field density. Strongest observed binding v2 predictor.' },
  { key: 'fp_id_collapse_rate', label: 'ID Coll', width: 82, title: 'Intrinsic-dimension collapse rate. Strong binding v2 signal but sparse.' },
  { key: 'fp_id_collapse_rate_normalized', label: 'ID CollN', width: 84, title: 'Normalized intrinsic-dimension collapse rate.' },
  { key: 'fp_jacobian_erf_decay_slope', label: 'ERF Decay', width: 88, title: 'ERF influence decay slope. Moderate binding and induction v2 signal.' },
  { key: 'fp_jacobian_erf_first_norm', label: 'ERF First', width: 84, title: 'ERF norm at the first input position.' },
  { key: 'fp_jacobian_erf_last_norm', label: 'ERF Last', width: 84, title: 'ERF norm at the last input position.' },
  { key: 'fp_logit_margin_velocity', label: 'Margin Vel', width: 88, title: 'Logit-margin velocity. Weak positive capability signal.' },
  { key: 'fp_logit_margin_delta', label: 'Margin Δ', width: 84, title: 'Total logit-margin change during fingerprinting.' },
  { key: 'fp_jacobian_erf_variance_log', label: 'ERF VarLog', width: 92, title: 'Log-scaled ERF variance.' },
  { key: 'fp_jacobian_spectral_norm_log', label: 'SpecLog', width: 82, title: 'Log-scaled Jacobian spectral norm.' },
  { key: 'fp_icld_velocity', label: 'ICLD Vel', width: 82, title: 'ICLD velocity. Empirically near-noise for capability.' },
  { key: 'fp_icld_delta_loss', label: 'ICLD ΔLoss', width: 92, title: 'ICLD early-to-late loss delta.' },
  { key: 'init_sensitivity_std', label: 'InitStd', width: 82, title: 'Sensitivity to weight initialization (lower means more predictable).' },
  { key: 'wikitext_perplexity', label: 'WikiPPL', width: 90, title: 'WikiText-103 validation perplexity (lower is better).' },
  { key: 'hellaswag_acc', label: 'HellaSwag', width: 90, title: 'HellaSwag accuracy (commonsense reasoning).' },
  { key: 'induction_auc', label: 'Ind v1', width: 78, title: 'Induction-head probe AUC (v1 protocol).' },
  { key: 'induction_v2_investigation_auc', label: 'Ind v2', width: 78, title: 'Induction-head probe AUC (v2 investigation protocol).' },
  { key: 'binding_auc', label: 'Bind v1', width: 78, title: 'Variable-binding probe AUC (v1 protocol).' },
  { key: 'binding_v2_investigation_auc', label: 'Bind v2', width: 78, title: 'Variable-binding probe AUC (v2 investigation protocol).' },
  { key: 'binding_composite', label: 'Bind Cmp', width: 84, title: 'Composite binding score (3-signal AND).' },
  { key: 'ar_auc', label: 'AR AUC', width: 78, title: 'Associative recall probe AUC.' },
  { key: 'blimp_overall_accuracy', label: 'BLiMP', width: 78, title: 'BLiMP overall grammatical-acceptability accuracy.' },
  { key: 'ncd_score', label: 'NCD', width: 78, title: 'Normalized compression distance score.' },
  { key: 'rapid_screening_passed', label: 'Rapid', width: 70, title: 'Whether the rapid-screening pre-gate passed.' },
  { key: 'stage_at_death', label: 'Died At', width: 90, title: 'Stage where the program failed (blank if it survived).' },
  { key: 'error_type', label: 'Error', width: 130, title: 'Error class if the program failed (e.g. unstable_dynamics).' },
  { key: 'completeness_ratio', label: 'Compl', width: 78, title: 'Fraction of promotion-relevant fields populated (backlog/all_graphs only).' },
  { key: 'missing_metrics_count', label: 'Missing', width: 80, title: 'Count of promotion-relevant metric fields still NULL.' },
  { key: 'tier', label: 'Status', width: STATUS_COLUMN_WIDTH, title: 'Current research phase of this architecture.' },
  { key: '_details', label: 'View', width: ACTION_COLUMN_WIDTH, title: 'Open the detailed fingerprint side panel for this architecture.' },
  { key: '_compare', label: 'Cmp', width: ACTION_COLUMN_WIDTH, title: 'Add architecture to side-by-side comparison.' },
  { key: '_designer', label: 'UI', width: ACTION_COLUMN_WIDTH, title: 'Open architecture in the visual designer.' },
];

const CORE_VISIBLE_COLUMNS = [
  '_score',
  'display_name',
  'architecture_family',
  'discovery_loss_ratio',
  'validation_loss_ratio',
  '_best_loss',
  'induction_v2_investigation_auc',
  'binding_v2_investigation_auc',
  'hellaswag_acc',
  'blimp_overall_accuracy',
  'ar_auc',
  'fp_jacobian_erf_density',
  'fp_id_collapse_rate',
  'tier',
  '_details',
  '_compare',
  '_designer',
];

const RESEARCH_VISIBLE_COLUMNS = [
  ...CORE_VISIBLE_COLUMNS,
  '_vs_ref',
  'param_efficiency',
  'sample_efficiency',
  'investigation_robustness',
  'robustness_long_ctx_score',
  'max_viable_seq_len',
  'init_sensitivity_std',
];

const DEFAULT_DISCOVERY_COLUMN_ADDONS = [
  'induction_v2_investigation_auc',
  'binding_v2_investigation_auc',
  'hellaswag_acc',
  'blimp_overall_accuracy',
  'ar_auc',
  'fp_jacobian_erf_density',
  'fp_id_collapse_rate',
];

const PROBES_VISIBLE_COLUMNS = [
  '_score',
  'display_name',
  'architecture_family',
  'wikitext_perplexity',
  'hellaswag_acc',
  'induction_auc',
  'induction_v2_investigation_auc',
  'binding_auc',
  'binding_v2_investigation_auc',
  'binding_composite',
  'ar_auc',
  'blimp_overall_accuracy',
  'ncd_score',
  'rapid_screening_passed',
  'stage_at_death',
  'error_type',
  'completeness_ratio',
  'missing_metrics_count',
  'tier',
  '_details',
  '_designer',
];

const ARCHITECTURE_VISIBLE_COLUMNS = [
  '_score',
  'display_name',
  'architecture_family',
  'induction_v2_investigation_auc',
  'binding_v2_investigation_auc',
  'fp_jacobian_erf_density',
  'fp_id_collapse_rate',
  'fp_id_collapse_rate_normalized',
  'fp_jacobian_erf_decay_slope',
  'fp_logit_margin_velocity',
  'fp_jacobian_effective_rank',
  'fp_sensitivity_uniformity',
  'fp_icld_velocity',
  'tier',
  '_details',
  '_designer',
];

function Discoveries({
  onSelectProgram,
  onAddToComparison,
  onRescreen,
  onPromoteScreening,
  onInvestigate,
  onValidate,
  onConfirm,
  highlightResultId,
  onHighlightClear,
  onQueueAdd,
  onQueueRemove,
  queuedResultIds,
  eligibilityByResultId,
  onOpenInDesigner,
}) {
  const { slowPollTick } = useAriaData();

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
    DISCOVERY_TIER_FILTERS.includes(prefs?.activeTier) ? prefs.activeTier : DEFAULT_ACTIVE_TIER
  );
  const [sortKey, setSortKey] = useState(() => typeof prefs?.sortKey === 'string' ? prefs.sortKey : '_score');
  const [sortDesc, setSortDesc] = useState(() => typeof prefs?.sortDesc === 'boolean' ? prefs.sortDesc : true);
  const [searchQuery, setSearchQuery] = useState(() => typeof prefs?.searchQuery === 'string' ? prefs.searchQuery : '');
  const [debouncedSearchQuery, setDebouncedSearchQuery] = useState(() => typeof prefs?.searchQuery === 'string' ? prefs.searchQuery : '');
  const [expandedRowId, setExpandedRowId] = useState(null);
  const [highlightId, setHighlightId] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [statusDrafts, setStatusDrafts] = useState({});
  const [savingStatusRowId, setSavingStatusRowId] = useState(null);
  const [statusError, setStatusError] = useState(null);
  const [showChart, setShowChart] = useState(true);
  const [showReferences, setShowReferences] = useState(() =>
    typeof prefs?.showReferences === 'boolean' ? prefs.showReferences : DEFAULT_SHOW_REFERENCES
  );
  const [hideFailed, setHideFailed] = useState(() =>
    typeof prefs?.hideFailed === 'boolean' ? prefs.hideFailed : DEFAULT_HIDE_FAILED
  );
  const [qualityFloorEnabled, setQualityFloorEnabled] = useState(() =>
    typeof prefs?.qualityFloorEnabled === 'boolean' ? prefs.qualityFloorEnabled : DEFAULT_QUALITY_FLOOR_ENABLED
  );
  const [sourceFilter, setSourceFilter] = useState(() =>
    ['trusted', 'all', 'all_graphs', 'untrusted', 'backfill', 'replay', 'backlog'].includes(prefs?.sourceFilter)
      ? prefs.sourceFilter
      : (typeof prefs?.trustedOnly === 'boolean' ? (prefs.trustedOnly ? 'trusted' : 'all') : DEFAULT_SOURCE_FILTER)
  );
  const [capabilityFilter, setCapabilityFilter] = useState(() =>
    ['all', 'qualified', 'training_only', 'pending'].includes(prefs?.capabilityFilter)
      ? prefs.capabilityFilter
      : DEFAULT_CAPABILITY_FILTER
  );
  const [showAdvancedSourceFilter, setShowAdvancedSourceFilter] = useState(() =>
    typeof prefs?.showAdvancedSourceFilter === 'boolean'
      ? prefs.showAdvancedSourceFilter
      : ['untrusted', 'backfill', 'replay'].includes(prefs?.sourceFilter)
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

      if (saved && saved.length > 0) {
        const next = [...saved];
        const insertAt = Math.max(
          next.findIndex((key) => key === 'tier'),
          0
        );
        for (const key of DEFAULT_DISCOVERY_COLUMN_ADDONS) {
          if (validKeys.has(key) && !next.includes(key)) {
            next.splice(insertAt, 0, key);
          }
        }
        return next;
      }

      const defaults = [...COLUMNS.map(c => c.key)];
      for (const key of requiredLongCtx) {
        if (!defaults.includes(key) && validKeys.has(key)) defaults.push(key);
      }
      return CORE_VISIBLE_COLUMNS.filter((key) => validKeys.has(key));
    }
  );
  const [showColumnPicker, setShowColumnPicker] = useState(false);
  const queuedSet = useMemo(() => new Set(queuedResultIds || []), [queuedResultIds]);
  const highlightRef = useRef(null);
  const visibleTableColumns = useMemo(
    () => COLUMNS.filter((col) => visibleColumns.includes(col.key)),
    [visibleColumns]
  );
  const displayNameColumnWidth = useMemo(
    () => visibleTableColumns.find((col) => col.key === 'display_name')?.width || 240,
    [visibleTableColumns]
  );
  const referenceLossIndex = useMemo(() => {
    const refs = Array.isArray(data?.references) ? data.references : [];
    const byFamily = new Map();
    let gpt2Loss = null;

    for (const ref of refs) {
      const loss = bestLoss(ref);
      if (loss == null) continue;

      const family = ref?.architecture_family;
      if (family) {
        const previous = byFamily.get(family);
        if (previous == null || loss < previous) {
          byFamily.set(family, loss);
        }
      }

      if (ref?.reference_name === 'GPT-2 Small' || ref?.reference_name === 'GPT-2') {
        gpt2Loss = gpt2Loss == null ? loss : Math.min(gpt2Loss, loss);
      }
    }

    return { byFamily, gpt2Loss };
  }, [data?.references]);
  const applyColumnPreset = useCallback((keys) => {
    const valid = new Set(COLUMNS.map((col) => col.key));
    setVisibleColumns(keys.filter((key) => valid.has(key)));
  }, []);

  // Persist preferences
  useEffect(() => {
    try {
      if (typeof window === 'undefined') return;
      window.localStorage.setItem(DISCOVERIES_PREFS_KEY, JSON.stringify({
        activeTier, sortKey, sortDesc, searchQuery, showChart, showReferences,
        qualityFloorEnabled, visibleColumns, hideFailed, sourceFilter,
        showAdvancedSourceFilter, capabilityFilter,
      }));
    } catch {}
  }, [activeTier, sortKey, sortDesc, searchQuery, showChart, showReferences, qualityFloorEnabled, visibleColumns, hideFailed, sourceFilter, showAdvancedSourceFilter, capabilityFilter]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearchQuery(searchQuery);
    }, 250);
    return () => clearTimeout(timer);
  }, [searchQuery]);

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
      const isBacklog = sourceFilter === 'backlog';
      const isAllGraphs = sourceFilter === 'all_graphs';
      const limit = sourceFilter === 'trusted' ? '200' : (isAllGraphs ? '5000' : '2500');
      const params = new URLSearchParams({
        sort: (isBacklog || isAllGraphs) ? 'loss_ratio' : 'composite_score',
        limit,
        view: isAllGraphs ? 'all_graphs' : (isBacklog ? 'backlog' : 'ranked'),
        trusted_only: sourceFilter === 'trusted' ? '1' : '0',
      });
      if (isBacklog || isAllGraphs) params.set('include_failed', '1');
      if (!isBacklog && !isAllGraphs && activeTier !== 'all') params.set('tier', activeTier);
      const q = debouncedSearchQuery.trim();
      if (q) {
        params.set('q', q);
        params.set('scope', 'all');
      }
      const res = await apiCall(`/api/discoveries?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      // Only update state if data actually changed — prevents scroll reset
      const entries = json?.entries || [];
      const fingerprint = `${entries.length}:${entries[0]?.entry_id || ''}:${entries[entries.length - 1]?.entry_id || ''}:${entries[0]?.composite_score ?? ''}`;
      if (fingerprint !== lastDataRef.current) {
        lastDataRef.current = fingerprint;
        setData(json);
        setLastUpdated(new Date());
      }
      setError(null);
    } catch (e) {
      if (!isBackground) setError('Failed to load discoveries: ' + e.message);
    } finally {
      if (!isBackground) setLoading(false);
    }
  }, [activeTier, debouncedSearchQuery, sourceFilter]);

  useEffect(() => {
    fetchData(slowPollTick > 0);
  }, [fetchData, slowPollTick]);

  const handleSort = (key) => {
    if (key === '_details' || key === '_compare' || key === '_designer') return;
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
      const res = await postJson('/api/leaderboard/status', {
        entry_id: entry.entry_id,
        result_id: entry.result_id,
        tier: nextTier,
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

  // Sort & augment discovery entries
  const sorted = useMemo(() => {
    const entries = data?.entries || [];
    
    const augmented = entries.map(e => {
      const entryBestLoss = bestLoss(e);
      
      // Find best reference in same family or paradigm for comparison
      let vsRef = null;
      if (entryBestLoss != null && !e.is_reference) {
        const bestRefLoss = referenceLossIndex.byFamily.get(e.architecture_family) ?? referenceLossIndex.gpt2Loss;
        if (bestRefLoss != null) {
          vsRef = percentOfReference(entryBestLoss, bestRefLoss);
        }
      }

      return {
        ...e,
        discovery_loss_ratio: discoveryLossDisplay(e),
        validation_loss_ratio: validationLossDisplay(e),
        _score: finiteOrNull(e.composite_score),
        _best_loss: entryBestLoss,
        _vs_ref: vsRef,
        _novelty: e.screening_novelty ?? e.novelty_score ?? null,
      };
    });
    augmented.sort((a, b) => {
      let va, vb;
      if (sortKey === 'tier') {
        va = TIER_ORDER[a.tier] || 0;
        vb = TIER_ORDER[b.tier] || 0;
      } else if (sortKey === 'display_name' || sortKey === 'architecture_family') {
        va = a[sortKey] || '';
        vb = b[sortKey] || '';
        return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      } else if (sortKey === '_capability_quality') {
        va = capabilityQualityRank(a);
        vb = capabilityQualityRank(b);
      } else {
        va = a[sortKey]; vb = b[sortKey];
      }
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return sortDesc ? vb - va : va - vb;
    });
    return augmented;
  }, [data?.entries, sortKey, sortDesc, referenceLossIndex]);

  const references = useMemo(() => {
    const refs = (data?.references || []).map((e) => ({
      ...e,
      _score: finiteOrNull(e.composite_score),
      _best_loss: bestLoss(e),
      _novelty: e.screening_novelty ?? e.novelty_score ?? null,
    }));
    refs.sort((a, b) => {
      const aScore = a._score;
      const bScore = b._score;
      if (aScore == null && bScore == null) return 0;
      if (aScore == null) return 1;
      if (bScore == null) return -1;
      return bScore - aScore;
    });
    return refs;
  }, [data?.references]);

  const sourceFiltered = useMemo(() => {
    return sorted.filter((entry) => {
      if (sourceFilter === 'backlog') return !entry?.entry_id;
      if (sourceFilter === 'all_graphs') return true;
      const bucket = provenanceBucket(entry);
      if (sourceFilter === 'all') return true;
      if (sourceFilter === 'trusted') return bucket === 'trusted';
      if (sourceFilter === 'untrusted') return bucket !== 'trusted';
      return bucket === sourceFilter;
    });
  }, [sorted, sourceFilter]);

  const tierFiltered = useMemo(() => {
    return sourceFiltered.filter((entry) => matchesActiveTier(entry, activeTier));
  }, [sourceFiltered, activeTier]);

  const effectiveQualityFloorEnabled = useMemo(() => {
    if (!qualityFloorEnabled) return false;
    return sourceFilter === 'trusted' || sourceFilter === 'all';
  }, [qualityFloorEnabled, sourceFilter]);

  const failedFiltered = useMemo(() => {
    if (!hideFailed) return tierFiltered;
    return tierFiltered.filter(e => {
      if (e.is_reference) return true;
      const tier = getDiscoveryDisplayStatus(e).tierKey;
      // Tier-based failures
      if (
        tier === 'screened_out'
        || tier === 'investigation_failed'
        || tier === 'validation_failed'
        || tier === 'failed'
        || tier === 'rejected'
      ) return false;
      if (e.screening_passed === false) return false;
      // Derived failures (mirrors DiscoveryUiBits logic)
      if (tier === 'investigation' && e.investigation_robustness != null && !e.investigation_passed) return false;
      if (tier === 'validation' && e.validation_baseline_ratio != null && !e.validation_passed) return false;
      return true;
    });
  }, [tierFiltered, hideFailed]);

  const capabilityFiltered = useMemo(() => {
    if (capabilityFilter === 'all') return failedFiltered;
    return failedFiltered.filter((entry) => {
      const status = capabilityQualityStatus(entry);
      if (capabilityFilter === 'qualified') return status === 'qualified' || status === 'breakthrough';
      if (capabilityFilter === 'training_only') return status === 'training_only';
      if (capabilityFilter === 'pending') return status === 'pending';
      return true;
    });
  }, [failedFiltered, capabilityFilter]);

  const failedHiddenCount = useMemo(() => {
    if (!hideFailed) return 0;
    return Math.max(0, (tierFiltered?.length || 0) - (failedFiltered?.length || 0));
  }, [hideFailed, tierFiltered, failedFiltered]);

  const qualityFiltered = useMemo(() => {
    if (!effectiveQualityFloorEnabled) return capabilityFiltered;
    return capabilityFiltered.filter((e) => {
      if (e?.is_reference) return true;
      const score = e?.composite_score;
      return score != null && (Number(score) / 100.0) >= QUALITY_FLOOR_THRESHOLD;
    });
  }, [capabilityFiltered, effectiveQualityFloorEnabled]);

  const qualityHiddenCount = useMemo(() => {
    if (!effectiveQualityFloorEnabled) return 0;
    return Math.max(0, (capabilityFiltered?.length || 0) - (qualityFiltered?.length || 0));
  }, [effectiveQualityFloorEnabled, capabilityFiltered, qualityFiltered]);

  const filtered = qualityFiltered;
  const expandedEntry = useMemo(
    () => filtered.find((entry, i) => (entry.entry_id || entry.result_id || i) === expandedRowId) || null,
    [filtered, expandedRowId]
  );

  const counts = data?.counts || data?.tier_counts || {};
  const summaryCounts = useMemo(() => {
    const base = { ...(counts || {}) };
    const entries = Array.isArray(data?.entries) ? data.entries : [];
    let backfill = 0;
    let replay = 0;
    for (const entry of entries) {
      const bucket = provenanceBucket(entry);
      if (bucket === 'replay') {
        replay += 1;
        continue;
      }
      if (bucket === 'backfill') {
        backfill += 1;
      }
    }
    base.backfill = backfill;
    base.replay = replay;
    return base;
  }, [counts, data?.entries]);
  const tiers = DISCOVERY_TIER_FILTERS;
  const hasLoadedData = Boolean(
    data && (Array.isArray(data.entries) || Array.isArray(data.references))
  );
  useEffect(() => {
    if (expandedRowId != null && !expandedEntry) {
      setExpandedRowId(null);
    }
  }, [expandedRowId, expandedEntry]);
  const filtersDirty = activeTier !== DEFAULT_ACTIVE_TIER
    || searchQuery.trim().length > 0
    || showReferences !== DEFAULT_SHOW_REFERENCES
    || hideFailed !== DEFAULT_HIDE_FAILED
    || qualityFloorEnabled !== DEFAULT_QUALITY_FLOOR_ENABLED
    || sourceFilter !== DEFAULT_SOURCE_FILTER
    || capabilityFilter !== DEFAULT_CAPABILITY_FILTER;
  const visibleTierLabel = activeTier === 'all' ? 'All statuses' : (TIER_LABELS[activeTier] || activeTier);
  const sourceFilterLabel = SOURCE_FILTER_LABELS[sourceFilter] || sourceFilter;
  const capabilityFilterLabel = CAPABILITY_FILTER_LABELS[capabilityFilter] || capabilityFilter;
  const filterSummaryParts = [
    `${filtered.length} ${filtered.length === 1 ? 'entry' : 'entries'}`,
    visibleTierLabel,
    sourceFilterLabel,
    showReferences ? 'references on' : 'references off',
    hideFailed ? 'failed hidden' : 'failed visible',
    effectiveQualityFloorEnabled
      ? `quality >= ${(QUALITY_FLOOR_THRESHOLD * 100).toFixed(0)}`
      : 'all quality',
  ];
  if (capabilityFilter !== 'all') {
    filterSummaryParts.push(capabilityFilterLabel);
  }
  if (searchQuery.trim()) {
    filterSummaryParts.push(`search "${searchQuery.trim()}"`);
  }

  const handleResetFilters = useCallback(() => {
    setActiveTier(DEFAULT_ACTIVE_TIER);
    setSearchQuery('');
    setDebouncedSearchQuery('');
    setShowReferences(DEFAULT_SHOW_REFERENCES);
    setHideFailed(DEFAULT_HIDE_FAILED);
    setQualityFloorEnabled(DEFAULT_QUALITY_FLOOR_ENABLED);
    setSourceFilter(DEFAULT_SOURCE_FILTER);
    setCapabilityFilter(DEFAULT_CAPABILITY_FILTER);
    setShowAdvancedSourceFilter(false);
  }, []);

  const presetButtonStyle = useCallback((isActive) => ({
    fontSize: 11,
    padding: '5px 10px',
    cursor: 'pointer',
    border: `1px solid ${isActive ? 'var(--accent-blue)' : 'var(--border)'}`,
    borderRadius: 4,
    background: isActive ? 'rgba(88, 166, 255, 0.12)' : 'transparent',
    color: isActive ? 'var(--accent-blue)' : 'var(--text-secondary)',
  }), []);
  const hasExactVisibleColumns = useCallback((keys) => {
    if (visibleColumns.length !== keys.length) return false;
    const keySet = new Set(keys);
    return visibleColumns.every((key) => keySet.has(key));
  }, [visibleColumns]);

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
          <SummaryBar tierCounts={summaryCounts} />
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
        <FingerprintLeaderboardChart entries={filtered} scoreScale={data?.score_scale} />
      )}

      {/* Tier filter tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 12, flexWrap: 'wrap' }}>
        {tiers.map(tier => {
          const count = tier === 'all'
            ? (counts.all || 0)
            : (counts[tier] || 0);
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
              {tier === 'all' ? 'All statuses' : TIER_LABELS[tier]}
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
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        gap: 8,
        marginBottom: 12,
      }}>
        <div style={FILTER_PANEL_STYLE}>
          <span style={FILTER_PANEL_TITLE_STYLE}>Scope</span>
          <button
            onClick={() => setShowReferences(v => !v)}
            aria-label={showReferences ? 'Hide references' : 'Show references'}
            style={toggleButtonStyle(showReferences, 'var(--accent-purple)', 'rgba(188, 140, 255, 0.12)')}
          >
            {showReferences ? 'References on' : 'References off'}
            {Number(counts.references || 0) > 0 && (
              <span style={{ marginLeft: 5, fontSize: 10, color: 'var(--accent-purple)' }}>
                ({counts.references})
              </span>
            )}
          </button>
          <label
            title="Filter by provenance source"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              fontSize: 11,
              color: 'var(--text-secondary)',
            }}
          >
            <span style={{ color: 'var(--text-muted)' }}>Source</span>
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              aria-label="Filter by source provenance"
              style={{
                fontSize: 11,
                border: '1px solid var(--border)',
                borderRadius: 4,
                background: 'var(--bg-card)',
                color: 'var(--text-primary)',
                outline: 'none',
                cursor: 'pointer',
                padding: '3px 22px 3px 8px',
                appearance: 'auto',
                minWidth: 132,
              }}
            >
              <option value="trusted">Trusted ranked</option>
              <option value="backlog">Backlog (unranked)</option>
              <option value="all_graphs">All graphs</option>
              {showAdvancedSourceFilter && (
                <>
                  <option value="all">Mixed trust ranked</option>
                  <option value="untrusted">Untrusted</option>
                  <option value="backfill">Backfill</option>
                  <option value="replay">Replay</option>
                </>
              )}
            </select>
          </label>
          <button
            type="button"
            onClick={() => setShowAdvancedSourceFilter((v) => !v)}
            aria-label={showAdvancedSourceFilter ? 'Hide advanced source filters' : 'Show advanced source filters'}
            title={showAdvancedSourceFilter ? 'Hide Untrusted/Backfill/Replay' : 'Show Untrusted/Backfill/Replay'}
            style={presetButtonStyle(showAdvancedSourceFilter)}
          >
            {showAdvancedSourceFilter ? 'Advanced on' : 'Advanced off'}
          </button>
          <label
            title="Filter by capability-quality state"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              fontSize: 11,
              color: 'var(--text-secondary)',
            }}
          >
            <span style={{ color: 'var(--text-muted)' }}>Capability</span>
            <select
              value={capabilityFilter}
              onChange={(e) => setCapabilityFilter(e.target.value)}
              aria-label="Filter by capability quality"
              style={{
                fontSize: 11,
                border: '1px solid var(--border)',
                borderRadius: 4,
                background: 'var(--bg-card)',
                color: 'var(--text-primary)',
                outline: 'none',
                cursor: 'pointer',
                padding: '3px 22px 3px 8px',
                appearance: 'auto',
                minWidth: 156,
              }}
            >
              <option value="all">All quality states</option>
              <option value="qualified">Capability-Qualified</option>
              <option value="training_only">Training-Only</option>
              <option value="pending">Validation Pending</option>
            </select>
          </label>
        </div>

        <div style={FILTER_PANEL_STYLE}>
          <span style={FILTER_PANEL_TITLE_STYLE}>Visibility</span>
          <button
            onClick={() => setHideFailed(v => !v)}
            aria-label={hideFailed ? 'Show failed' : 'Hide failed'}
            style={toggleButtonStyle(hideFailed, 'var(--accent-red)', 'rgba(248, 81, 73, 0.12)')}
          >
            {hideFailed ? 'Failed hidden' : 'Failed visible'}
          </button>
          <button
            onClick={() => setQualityFloorEnabled(v => !v)}
            aria-label={qualityFloorEnabled ? 'Disable quality floor' : 'Enable quality floor'}
            style={toggleButtonStyle(effectiveQualityFloorEnabled, 'var(--accent-green)', 'rgba(63, 185, 80, 0.14)')}
            title={effectiveQualityFloorEnabled
              ? `Hide entries with composite score < ${(QUALITY_FLOOR_THRESHOLD * 100).toFixed(0)}`
              : `Quality floor is bypassed for ${sourceFilterLabel.toLowerCase()}`
            }
          >
            {effectiveQualityFloorEnabled
              ? `Quality ≥ ${(QUALITY_FLOOR_THRESHOLD * 100).toFixed(0)}`
              : (qualityFloorEnabled ? 'Quality bypassed' : 'All quality')}
          </button>
          {filtersDirty && (
            <button
              onClick={handleResetFilters}
              style={{
                fontSize: 11,
                padding: '5px 12px',
                cursor: 'pointer',
                border: '1px solid var(--accent-orange)',
                borderRadius: 4,
                background: 'rgba(255, 166, 87, 0.10)',
                color: 'var(--accent-orange)',
              }}
              title="Reset tier, search, provenance, quality, and reference filters"
            >
              Reset filters
            </button>
          )}
        </div>

        <div style={FILTER_PANEL_STYLE}>
          <span style={FILTER_PANEL_TITLE_STYLE}>Columns</span>
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
            {showColumnPicker ? 'Picker open' : 'Picker closed'}
          </button>
          <button
            onClick={() => applyColumnPreset(CORE_VISIBLE_COLUMNS)}
            style={presetButtonStyle(hasExactVisibleColumns(CORE_VISIBLE_COLUMNS))}
            title="Show the core discovery columns"
          >
            Core
          </button>
          <button
            onClick={() => applyColumnPreset(RESEARCH_VISIBLE_COLUMNS)}
            style={presetButtonStyle(hasExactVisibleColumns(RESEARCH_VISIBLE_COLUMNS))}
            title="Show an expanded research-oriented column set"
          >
            Research
          </button>
          <button
            onClick={() => applyColumnPreset(PROBES_VISIBLE_COLUMNS)}
            style={presetButtonStyle(hasExactVisibleColumns(PROBES_VISIBLE_COLUMNS))}
            title="Show probe metrics: induction, binding, hellaswag, ar, blimp, ppl, completeness"
          >
            Probes
          </button>
          <button
            onClick={() => applyColumnPreset(ARCHITECTURE_VISIBLE_COLUMNS)}
            style={presetButtonStyle(hasExactVisibleColumns(ARCHITECTURE_VISIBLE_COLUMNS))}
            title="Show fingerprint architecture metrics from the recent backfill"
          >
            Architecture
          </button>
          <button
            onClick={() => applyColumnPreset(COLUMNS.map((col) => col.key))}
            style={presetButtonStyle(hasExactVisibleColumns(COLUMNS.map((col) => col.key)))}
            title="Show all available columns"
          >
            All
          </button>
        </div>

        <div style={{ ...FILTER_PANEL_STYLE, justifyContent: 'space-between' }}>
          <span style={FILTER_PANEL_TITLE_STYLE}>Actions</span>
          <button
            onClick={fetchData}
            disabled={loading}
            aria-label="Refresh discoveries"
            style={{
              fontSize: 11,
              padding: '5px 12px',
              cursor: loading ? 'not-allowed' : 'pointer',
              background: 'transparent',
              border: '1px solid var(--border)',
              borderRadius: 4,
              color: 'var(--text-secondary)',
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </div>

      <div style={{
        display: 'flex',
        gap: 8,
        alignItems: 'center',
        flexWrap: 'wrap',
        marginBottom: 12,
      }}>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          Showing:
        </span>
        {filterSummaryParts.map((part, idx) => (
          <span
            key={`${part}-${idx}`}
            style={{
              fontSize: 11,
              padding: '3px 8px',
              borderRadius: 999,
              border: '1px solid var(--border)',
              background: 'var(--bg-secondary)',
              color: 'var(--text-secondary)',
            }}
          >
            {part}
          </span>
        ))}
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
      <div style={{ marginBottom: 12, display: 'flex', gap: 8, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <input
          type="text"
          placeholder="Search by name, family, fingerprint, or ID..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          aria-label="Search discoveries"
          style={{
            flex: '1 1 320px',
            minWidth: 220,
            maxWidth: 520,
            padding: '6px 10px', fontSize: 12,
            border: '1px solid var(--border)', borderRadius: 4,
            background: 'var(--bg-secondary)', color: 'var(--text-primary)',
          }}
        />
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', flex: '1 1 240px' }}>
          {searchQuery && (
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              {loading ? 'Searching full DB...' : `${filtered.length} matches`}
            </span>
          )}
          {qualityFloorEnabled && qualityHiddenCount > 0 && (
            <span style={{ fontSize: 11, color: 'var(--accent-yellow)' }}>
              {qualityHiddenCount} low-quality hidden
            </span>
          )}
          {qualityFloorEnabled && !effectiveQualityFloorEnabled && (
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              quality floor bypassed for {sourceFilterLabel.toLowerCase()}
            </span>
          )}
          {hideFailed && failedHiddenCount > 0 && (
            <span style={{ fontSize: 11, color: 'var(--accent-red)' }}>
              {failedHiddenCount} failed hidden
            </span>
          )}
        </div>
      </div>

      {/* Reference Baselines Banner */}
      {showReferences && references.length > 0 && (
        <div style={{
          marginBottom: 14, padding: '10px 14px',
          background: 'linear-gradient(135deg, rgba(45, 212, 191, 0.08), rgba(255, 209, 102, 0.06))',
          border: '1px solid rgba(227, 179, 65, 0.28)',
          borderRadius: 8,
        }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--score-elite)', marginBottom: 8, textTransform: 'uppercase' }}>
            Reference Baselines
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
            Post-BPE anchors for the champion scale. These rows stay visible independently of discovery stage filters.
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {references.map(ref => (
              <div key={ref.entry_id || ref.result_id} style={{
                padding: '6px 12px', borderRadius: 5,
                background: 'rgba(13, 17, 23, 0.42)',
                border: `1px solid ${scoreColor(ref._score)}`,
                fontSize: 11, lineHeight: 1.5, minWidth: 150,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'baseline' }}>
                  <div style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
                    {ref.reference_name || ref.display_name || ref.architecture_desc || 'Reference'}
                  </div>
                  {ref._score != null && (
                    <div style={{ color: scoreColor(ref._score), fontWeight: 700, fontFamily: 'monospace' }}>
                      {Number(ref._score).toFixed(1)}
                    </div>
                  )}
                </div>
                {ref._score != null && (
                  <div className="champion-strip" title={scoreToneLabel(ref._score)} style={{ margin: '4px 0 5px' }}>
                    <div
                      className="champion-strip-fill"
                      style={{
                        width: `${Math.max(4, Math.min(100, (Number(ref._score) / 320) * 100))}%`,
                        background: scoreGradient(ref._score),
                      }}
                    />
                  </div>
                )}
                <div style={{ color: 'var(--text-muted)' }}>
                  {ref.architecture_family || '--'}
                  {ref._best_loss != null && <span style={{ marginLeft: 8 }}>Loss: {ref._best_loss.toFixed(4)}</span>}
                </div>
                {ref.param_count != null && (
                  <div style={{ color: 'var(--text-muted)' }}>
                    {(ref.param_count / 1e6).toFixed(1)}M params
                  </div>
                )}
                {ref.result_id && (
                  <div style={{ marginTop: 8, display: 'flex', gap: 6 }}>
                    <button
                      onClick={() => onSelectProgram?.(ref.result_id)}
                      style={{
                        ...actionBtnStyle,
                        borderColor: 'var(--accent-purple)',
                        color: 'var(--accent-purple)',
                        background: 'rgba(188, 140, 255, 0.12)',
                      }}
                      title="Open full reference detail"
                    >
                      Details
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {error && <p style={{ color: 'var(--accent-red)', fontSize: 13, marginBottom: 8 }}>{error}</p>}
      {statusError && <p style={{ color: 'var(--accent-red)', fontSize: 12, marginBottom: 8 }}>{statusError}</p>}

      {loading && !hasLoadedData ? (
        <p style={{ color: 'var(--text-muted)' }}>Loading discoveries...</p>
      ) : filtered.length === 0 && !error ? (
        <div style={{ color: 'var(--text-muted)', fontSize: 13, lineHeight: 1.6 }}>
          {searchQuery.trim() ? (
            <p>No discoveries match "{searchQuery}" in the full notebook.</p>
          ) : activeTier === 'all' ? (
            <p>No discoveries yet. Run experiments to generate candidates.</p>
          ) : (
            <p>No entries in {TIER_LABELS[activeTier]} tier yet.</p>
          )}
        </div>
      ) : (
        <div>
          {loading && hasLoadedData && (
            <p style={{ color: 'var(--text-muted)', fontSize: 12, marginBottom: 8 }}>
              Refreshing discoveries...
            </p>
          )}
          <div className="discoveries-table-shell">
            <div className="discoveries-table-meta">
              <span>Pinned while scrolling: rank and architecture</span>
              <span>{visibleTableColumns.length} visible columns</span>
            </div>
            <div className="discoveries-table-wrap">
          <table className="data-table table-wide table-compact discoveries-table" style={{ tableLayout: 'fixed' }}>
            <colgroup>
              <col style={{ width: PIN_COLUMN_WIDTH }} />
              <col style={{ width: RANK_COLUMN_WIDTH }} />
              {visibleTableColumns.map((col) => (
                <col key={col.key} style={{ width: col.width ? `${col.width}px` : '104px' }} />
              ))}
            </colgroup>
            <thead style={{ position: 'sticky', top: 0, zIndex: 2, background: 'var(--bg-card, #1a1a2e)' }}>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                <th className="sticky-cell sticky-header" style={{ ...thStyle, left: 0, width: PIN_COLUMN_WIDTH }} aria-label="Pinned marker" />
                <th className="sticky-cell sticky-header" style={{ ...thStyle, left: PIN_COLUMN_WIDTH, width: RANK_COLUMN_WIDTH }}>#</th>
                {visibleTableColumns.map(col => (
                  <th
                    key={col.key}
                    onClick={() => handleSort(col.key)}
                    title={col.title}
                    className={col.key === 'display_name' ? 'sticky-cell sticky-header sticky-divider' : undefined}
                    style={{
                      ...thStyle,
                      width: col.width ? `${col.width}px` : undefined,
                      left: col.key === 'display_name' ? (PIN_COLUMN_WIDTH + RANK_COLUMN_WIDTH) : undefined,
                      cursor: (col.key === '_details' || col.key === '_compare' || col.key === '_designer') ? 'default' : 'pointer',
                      userSelect: 'none',
                    }}
                  >
                    {col.label}
                    <SortIndicator active={sortKey === col.key} desc={sortDesc} />
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
                    onAddToComparison={onAddToComparison}
                    tdStyle={tdStyle}
                    COLUMNS={COLUMNS}
                    visibleColumns={visibleTableColumns.map((col) => col.key)}
                    stickyDisplayNameLeft={PIN_COLUMN_WIDTH + RANK_COLUMN_WIDTH}
                    stickyDisplayNameWidth={displayNameColumnWidth}
                    onOpenInDesigner={onOpenInDesigner}
                    setExpandedRowId={setExpandedRowId}
                    actionBtnStyle={actionBtnStyle}
                    handleDelete={handleDelete}
                    onRescreen={onRescreen}
                    onPromoteScreening={onPromoteScreening}
                    onInvestigate={onInvestigate}
                    onValidate={onValidate}
                    onConfirm={onConfirm}
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
        </div>
        {expandedEntry && (
          <ExpandedDetailPanel
            entry={expandedEntry}
            onClose={() => setExpandedRowId(null)}
            onRescreen={onRescreen}
            onPromoteScreening={onPromoteScreening}
            onInvestigate={onInvestigate}
            onValidate={onValidate}
            onConfirm={onConfirm}
            onQueueAdd={onQueueAdd}
            onQueueRemove={onQueueRemove}
            onDelete={handleDelete}
            isQueued={Boolean(expandedEntry.result_id) && queuedSet.has(expandedEntry.result_id)}
            eligibility={eligibilityByResultId?.[expandedEntry.result_id] || null}
            statusDraft={statusDrafts[expandedEntry.entry_id || expandedEntry.result_id] || expandedEntry.tier}
            onStatusDraftChange={(tier) => {
              const rowId = expandedEntry.entry_id || expandedEntry.result_id;
              handleStatusDraftChange(rowId, tier);
            }}
            onSaveStatus={() => handleSaveStatus(expandedEntry)}
            savingStatus={savingStatusRowId === (expandedEntry.entry_id || expandedEntry.result_id)}
            actionBtnStyle={actionBtnStyle}
          />
        )}
        </div>
      )}
    </div>
  );
}

const thStyle = {
  padding: '6px 10px', textAlign: 'left', fontSize: 10,
  color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', whiteSpace: 'nowrap',
};

const tdStyle = {
  padding: '6px 10px', whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
};

const statusCellStyle = {
  ...tdStyle,
  padding: '6px 10px',
  overflow: 'hidden',
};

const actionCellStyle = {
  ...tdStyle,
  padding: '6px',
  textAlign: 'center',
  overflow: 'hidden',
};

const actionBtnStyle = {
  padding: '4px 8px', fontSize: 10,
  border: '1px solid rgba(88, 166, 255, 0.4)', borderRadius: 4,
  background: 'rgba(88, 166, 255, 0.12)', color: 'var(--accent-blue)', cursor: 'pointer',
  width: '100%',
  minWidth: 0,
  whiteSpace: 'nowrap',
  lineHeight: 1.1,
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
  onAddToComparison,
  tdStyle,
  COLUMNS,
  visibleColumns,
  stickyDisplayNameLeft,
  stickyDisplayNameWidth,
  onOpenInDesigner,
  setExpandedRowId,
  actionBtnStyle,
  handleDelete,
  onRescreen,
  onPromoteScreening,
  onInvestigate,
  onValidate,
  onConfirm,
  onQueueAdd,
  onQueueRemove,
  statusDrafts,
  handleStatusDraftChange,
  handleSaveStatus,
  savingStatusRowId
}) {
  const canDelete = !entry.is_reference && (entry.tier === 'screening' || entry.tier === 'failed' || entry.tier === 'rejected' || entry.screening_passed === false || entry.investigation_passed === false || entry.validation_passed === false);
  const rowColors = rowBackgrounds({
    index: i,
    isHighlighted,
    isPinnedReference,
    isExpanded,
    tier: entry.tier,
    score: entry._score ?? entry.composite_score,
  });

  return (
    <React.Fragment>
      <tr
        ref={isHighlighted ? highlightRef : undefined}
        style={{
          '--discoveries-row-bg': rowColors.base,
          '--discoveries-row-hover-bg': rowColors.hover,
          borderBottom: '1px solid var(--border)',
          cursor: 'pointer',
          animation: isHighlighted ? 'leaderboard-pulse 1.5s ease-in-out 2' : undefined,
        }}
        onClick={() => setExpandedRowId(isExpanded ? null : rowId)}
      >
        <td className="sticky-cell sticky-body" style={{ ...tdStyle, left: 0, width: PIN_COLUMN_WIDTH, textAlign: 'center', paddingLeft: 4, paddingRight: 4 }}>
          {isPinnedReference ? (
            <span title="Pinned reference" style={{ color: 'var(--accent-purple)', fontSize: 12, fontWeight: 700 }}>
              ★
            </span>
          ) : null}
        </td>
        <td className="sticky-cell sticky-body" style={{ ...tdStyle, left: PIN_COLUMN_WIDTH, width: RANK_COLUMN_WIDTH, fontVariantNumeric: 'tabular-nums' }}>{i + 1}</td>
        {visibleColumns.map((colKey) => {
          const col = COLUMNS.find((item) => item.key === colKey);
          if (!col) return null;
          switch (col.key) {
            case '_score':
              return <td key={col.key} style={tdStyle}><ScoreCell entry={entry} /></td>;
            case '_capability_quality':
              return (
                <td key={col.key} style={tdStyle}>
                  <StatusBadge entry={entry} />
                </td>
              );
            case 'display_name':
              return (
                <td
                  key={col.key}
                  className="sticky-cell sticky-body sticky-divider"
                  style={{
                    ...tdStyle,
                    left: stickyDisplayNameLeft,
                    width: stickyDisplayNameWidth,
                    minWidth: stickyDisplayNameWidth,
                    maxWidth: stickyDisplayNameWidth,
                    whiteSpace: 'normal',
                    overflowWrap: 'anywhere',
                    lineHeight: 1.35,
                  }}
                >
                  <div style={{ fontWeight: 500 }}>{displayName}</div>
                  {entry.graph_fingerprint && (
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'monospace', whiteSpace: 'normal', overflowWrap: 'anywhere' }}>
                      {entry.graph_fingerprint}
                    </div>
                  )}
                </td>
              );
            case 'architecture_family':
              return (
                <td key={col.key} style={{ ...tdStyle, whiteSpace: 'normal', overflow: 'hidden', textOverflow: 'ellipsis' }}>
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
                <td key={col.key} style={{ ...tdStyle, textAlign: 'right', color: lossColor(discoveryDisplay), fontFamily: 'monospace' }}>
                  {discoveryDisplay != null ? Number(discoveryDisplay).toFixed(4) : '--'}
                </td>
              );
            case 'validation_loss_ratio':
              const validationDisplay = validationLossDisplay(entry);
              return (
                <td key={col.key} style={{ ...tdStyle, textAlign: 'right', color: lossColor(validationDisplay), fontFamily: 'monospace' }}>
                  {validationDisplay != null ? Number(validationDisplay).toFixed(4) : '--'}
                </td>
              );
            case '_best_loss':
              return (
                <td key={col.key} style={{ ...tdStyle, textAlign: 'right', color: lossColor(entry._best_loss), fontFamily: 'monospace' }}>
                  {entry._best_loss != null ? (Number(entry._best_loss) !== 0 && Math.abs(Number(entry._best_loss)) < 0.0001 ? Number(entry._best_loss).toExponential(2) : Number(entry._best_loss).toFixed(4)) : '--'}
                </td>
              );
            case '_vs_ref':
              return (
                <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace', color: entry._vs_ref != null ? (entry._vs_ref <= 100 ? 'var(--accent-green)' : 'var(--accent-red)') : 'var(--text-muted)' }}>
                  {entry._vs_ref != null ? `${entry._vs_ref.toFixed(1)}%` : '--'}
                </td>
              );
            case '_novelty':
              return (
                <td key={col.key} style={{ ...tdStyle, textAlign: 'right', color: noveltyColor(entry._novelty), fontFamily: 'monospace' }}>
                  {entry._novelty != null ? Number(entry._novelty).toFixed(3) : '--'}
                </td>
              );
            case 'param_efficiency':
              return (
                <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.param_efficiency != null ? Number(entry.param_efficiency).toFixed(3) : '--'}</td>
              );
            case 'sample_efficiency':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.sample_efficiency != null ? Number(entry.sample_efficiency).toFixed(3) : '--'}</td>;
            case 'investigation_robustness':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.investigation_robustness != null ? Number(entry.investigation_robustness).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_score':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.robustness_long_ctx_score != null ? Number(entry.robustness_long_ctx_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_scaling_score':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.robustness_long_ctx_scaling_score != null ? Number(entry.robustness_long_ctx_scaling_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_assoc_score':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.robustness_long_ctx_assoc_score != null ? Number(entry.robustness_long_ctx_assoc_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_multi_hop_score':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.robustness_long_ctx_multi_hop_score != null ? Number(entry.robustness_long_ctx_multi_hop_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_passkey_score':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.robustness_long_ctx_passkey_score != null ? Number(entry.robustness_long_ctx_passkey_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_retrieval_aggregate':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.robustness_long_ctx_retrieval_aggregate != null ? Number(entry.robustness_long_ctx_retrieval_aggregate).toFixed(3) : '--'}</td>;
            case 'max_viable_seq_len':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.max_viable_seq_len != null ? Number(entry.max_viable_seq_len).toFixed(0) : '--'}</td>;
            case 'jacobian_spectral_norm':
              const specVal = finitePositiveOrNull(entry.jacobian_spectral_norm ?? entry.fp_jacobian_spectral_norm);
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{specVal != null ? Number(specVal).toFixed(4) : '--'}</td>;
            case 'fp_jacobian_effective_rank':
            case 'fp_sensitivity_uniformity':
            case 'fp_jacobian_erf_density':
            case 'fp_id_collapse_rate':
            case 'fp_id_collapse_rate_normalized':
            case 'fp_jacobian_erf_decay_slope':
            case 'fp_jacobian_erf_first_norm':
            case 'fp_jacobian_erf_last_norm':
            case 'fp_logit_margin_velocity':
            case 'fp_logit_margin_delta':
            case 'fp_jacobian_erf_variance_log':
            case 'fp_jacobian_spectral_norm_log':
            case 'fp_icld_velocity':
            case 'fp_icld_delta_loss':
              return (
                <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace', color: fingerprintTone(col.key, entry[col.key]) }}>
                  {metricDisplay(entry[col.key], 3)}
                </td>
              );
            case 'init_sensitivity_std':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.init_sensitivity_std != null ? Number(entry.init_sensitivity_std).toFixed(4) : '--'}</td>;
            case 'wikitext_perplexity':
            case 'hellaswag_acc':
            case 'induction_auc':
            case 'induction_v2_investigation_auc':
            case 'binding_auc':
            case 'binding_v2_investigation_auc':
            case 'binding_composite':
            case 'ar_auc':
            case 'blimp_overall_accuracy':
              return (
                <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace', color: scoreCellTone(col.key, entry[col.key]), fontWeight: 600 }}>
                  {entry[col.key] != null ? Number(entry[col.key]).toFixed(col.key === 'wikitext_perplexity' ? 2 : 3) : '--'}
                </td>
              );
            case 'ncd_score':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.ncd_score != null ? Number(entry.ncd_score).toFixed(3) : '--'}</td>;
            case 'rapid_screening_passed':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'center', fontFamily: 'monospace' }}>{entry.rapid_screening_passed == null ? '--' : (entry.rapid_screening_passed ? '✓' : '✗')}</td>;
            case 'stage_at_death':
              return <td key={col.key} style={{ ...tdStyle, fontFamily: 'monospace', color: entry.stage_at_death ? 'var(--accent-red)' : 'var(--text-muted)' }}>{entry.stage_at_death || '--'}</td>;
            case 'error_type':
              return <td key={col.key} style={{ ...tdStyle, fontFamily: 'monospace', color: entry.error_type ? 'var(--accent-red)' : 'var(--text-muted)' }} title={entry.error_message || ''}>{entry.error_type || '--'}</td>;
            case 'completeness_ratio':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.completeness_ratio != null ? `${(Number(entry.completeness_ratio) * 100).toFixed(0)}%` : '--'}</td>;
            case 'missing_metrics_count':
              return <td key={col.key} style={{ ...tdStyle, textAlign: 'right', fontFamily: 'monospace' }}>{entry.missing_metrics_count != null ? Number(entry.missing_metrics_count).toFixed(0) : '--'}</td>;
            case 'tier':
              return (
                <td key={col.key} style={statusCellStyle}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 0 }}>
                    <StatusBadge entry={entry} />
                  </div>
                </td>
              );
            case '_details':
              return (
                <td key={col.key} style={actionCellStyle} onClick={e => e.stopPropagation()}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <button
                      onClick={() => {
                        if (entry.result_id) onSelectProgram?.(entry.result_id);
                      }}
                      style={{
                        ...actionBtnStyle,
                        borderColor: 'var(--accent-blue)',
                        color: 'var(--accent-blue)',
                        background: 'transparent',
                      }}
                      title={entry.result_id ? 'Open detailed fingerprint side panel' : 'Detail view unavailable: missing result ID'}
                    >
                      View
                    </button>
                    {eligibility?.confirmationEligible && (
                      <button
                        onClick={() => {
                          if (entry.result_id) onConfirm?.([entry.result_id]);
                        }}
                        style={{
                          ...actionBtnStyle,
                          borderColor: 'rgba(255, 184, 108, 0.55)',
                          color: 'var(--score-elite)',
                          background: 'rgba(255, 184, 108, 0.10)',
                        }}
                        title="Run champion confirmation to test score stability under extended post-validation training"
                      >
                        Confirm
                      </button>
                    )}
                  </div>
                </td>
              );
            case '_compare':
              return (
                <td key={col.key} style={actionCellStyle} onClick={e => e.stopPropagation()}>
                  {onAddToComparison ? (
                    <button
                      onClick={() => {
                        if (entry.result_id) onAddToComparison(entry.result_id);
                      }}
                      disabled={!entry.result_id}
                      style={{
                        ...actionBtnStyle,
                        borderColor: 'var(--accent-green)',
                        color: 'var(--accent-green)',
                        opacity: entry.result_id ? 1 : 0.5,
                        cursor: entry.result_id ? 'pointer' : 'not-allowed',
                      }}
                      title={entry.result_id ? 'Add architecture to side-by-side comparison' : 'Comparison unavailable: missing result ID'}
                    >
                      Cmp
                    </button>
                  ) : <span style={{ color: 'var(--text-muted)' }}>--</span>}
                </td>
              );
            case '_designer':
              return (
                <td key={col.key} style={actionCellStyle} onClick={e => e.stopPropagation()}>
                  {onOpenInDesigner ? (
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
                      Open
                    </button>
                  ) : <span style={{ color: 'var(--text-muted)' }}>--</span>}
                </td>
              );
            default:
              return null;
          }
        })}
      </tr>
    </React.Fragment>
  );
});

export default Discoveries;
