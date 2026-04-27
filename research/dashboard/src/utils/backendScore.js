function finiteNumber(value) {
  if (value == null) return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

const BPE_METRIC_VERSION = 'bpe_eval_v1';

export function evalMetricQuality(entry) {
  const version = String(entry?.screening_wikitext_metric_version || '').trim();
  const tokenizer = String(entry?.tokenizer_mode || '').trim().toLowerCase();
  const status = String(entry?.rescore_status || entry?.blimp_status || entry?.screening_wikitext_status || '').trim().toLowerCase();
  const tags = String(entry?.tags || '').trim().toLowerCase();
  const isBpe = version === BPE_METRIC_VERSION || ['tiktoken', 'bpe', 'cl100k_base'].includes(tokenizer);
  const explicitFailure = (
    status.includes('failed')
    || status.includes('error')
    || tags.includes('bpe_eval_failed')
  );
  const required = [
    ['WikiText', entry?.wikitext_perplexity],
    ['TinyStories', entry?.tinystories_perplexity],
    ['HellaSwag', entry?.hellaswag_acc],
    ['BLiMP', entry?.blimp_overall_accuracy],
  ];
  const missing = required
    .filter(([, value]) => finiteNumber(value) == null)
    .map(([label]) => label);

  if (isBpe && missing.length === 0) {
    return {
      key: 'trusted_bpe',
      label: 'Trusted BPE',
      reliability: 'high',
      color: 'var(--accent-green)',
      missing,
      version: version || tokenizer,
    };
  }

  if (isBpe) {
    return {
      key: 'partial_bpe',
      label: 'Partial BPE',
      reliability: 'medium',
      color: 'var(--accent-yellow)',
      missing,
      version: version || tokenizer,
    };
  }

  if (explicitFailure || required.every(([, value]) => finiteNumber(value) == null)) {
    return {
      key: 'failed_eval',
      label: 'Eval Missing',
      reliability: 'low',
      color: 'var(--accent-red)',
      missing,
      version: version || 'unversioned',
    };
  }

  return {
    key: 'legacy_eval',
    label: 'Legacy Eval',
    reliability: 'low',
    color: 'var(--text-muted)',
    missing,
    version: version || 'legacy',
  };
}

export const CANONICAL_SCORE_COMPONENT_META = {
  _v10_base_v8style_total: { label: 'Loss + Understanding Base', color: 'var(--accent-green)' },
  _v10_capability_total: { label: 'Capability Total', color: '#79c0ff' },
  _v10_aux_trajectory_total: { label: 'Aux Trajectory Total', color: 'var(--accent-yellow)' },
  perf_short: { label: 'Screening Loss', color: 'var(--accent-blue)' },
  perf_medium: { label: 'Investigation Loss', color: '#1f6feb' },
  perf_long: { label: 'Validation Loss', color: 'var(--accent-green)' },
  sLoss: { label: 'Screening Loss', color: 'var(--accent-blue)' },
  iLoss: { label: 'Investigation Loss', color: '#1f6feb' },
  vBase: { label: 'Baseline', color: 'var(--accent-green)' },
  baseline: { label: 'Baseline', color: 'var(--accent-green)' },
  novelty: { label: 'Novelty', color: 'var(--accent-purple)' },
  robustness: { label: 'Robustness', color: 'var(--accent-yellow)' },
  robust: { label: 'Robustness', color: 'var(--accent-yellow)' },
  robustnessBonus: { label: 'Robustness', color: 'var(--accent-yellow)' },
  long_context: { label: 'Long Context', color: '#79c0ff' },
  speed: { label: 'Speed', color: 'var(--text-muted)' },
  throughput: { label: 'Throughput', color: 'var(--text-muted)' },
  binding: { label: 'Binding Range', color: '#a371f7' },
  blimp: { label: 'BLiMP Linguistic', color: '#79c0ff' },
  compression: { label: 'Compression', color: '#56d364' },
  sparsity: { label: 'Sparsity', color: '#3fb950' },
  adaptive_computation: { label: 'Adaptive Compute', color: '#c77dff' },
  adaptiveBonus: { label: 'Adaptive Compute', color: '#c77dff' },
  routing_savings: { label: 'Routing', color: '#58a6ff' },
  routingBonus: { label: 'Routing', color: '#3fb950' },
  param_efficiency: { label: 'Param Efficiency', color: '#e3b341' },
  efficiencyBonus: { label: 'Efficiency', color: '#58a6ff' },
  learning_efficiency: { label: 'Learning Efficiency', color: '#db61a2' },
  early_convergence: { label: 'Early Convergence', color: '#f0883e' },
  consistency: { label: 'Consistency', color: '#d29922' },
  cross_task: { label: 'Cross Task', color: '#3fb950' },
  diagnostic: { label: 'Diagnostic', color: '#d29922' },
  hellaswag: { label: 'HellaSwag', color: 'var(--accent-orange)' },
  hierarchy: { label: 'Hierarchy', color: '#58a6ff' },
  tinystories: { label: 'TinyStories', color: '#56d364' },
  tierBonus: { label: 'Tier Bonus', color: 'var(--accent-orange)' },
  referenceDeltaBonus: { label: 'Baseline Delta', color: 'var(--accent-orange)' },
  cap_ar: { label: 'AR Probe', color: '#a371f7' },
  cap_induction: { label: 'Induction Probe', color: '#79c0ff' },
  cap_binding: { label: 'Binding Probe', color: '#a371f7' },
  cap_erf_density: { label: 'ERF Density', color: '#56d364' },
  cap_id_collapse: { label: 'ID Collapse', color: '#3fb950' },
  cap_erf_decay: { label: 'ERF Decay', color: '#c77dff' },
  cap_logit_margin: { label: 'Logit Margin', color: '#f0883e' },
  aux_erf_variance: { label: 'ERF Variance', color: 'var(--text-muted)' },
  aux_icld: { label: 'ICLD Velocity', color: 'var(--text-muted)' },
};

const V10_ADDITIVE_TOTAL_KEYS = [
  '_v10_base_v8style_total',
  '_v10_capability_total',
  '_v10_aux_trajectory_total',
];

function scoreComponentFor(key, weight) {
  return {
    key,
    weight: Number(weight),
    ...(CANONICAL_SCORE_COMPONENT_META[key] || { label: key, color: 'var(--text-muted)' }),
  };
}

function isDisplayableScoreComponent(key, weight) {
  return (
    Number.isFinite(Number(weight))
    && Number(weight) > 0
    && !String(key).includes('penalty')
    && !String(key).startsWith('_')
  );
}

export function canonicalCompositeScore(entry) {
  return finiteNumber(entry?.composite_score);
}

export function canonicalScoreBreakdown(entry) {
  return entry?.score_breakdown && typeof entry.score_breakdown === 'object'
    ? entry.score_breakdown
    : {};
}

export function canonicalScoreComponents(entry) {
  const breakdown = canonicalScoreBreakdown(entry);
  const v10Totals = V10_ADDITIVE_TOTAL_KEYS
    .filter(key => Number.isFinite(Number(breakdown[key])) && Number(breakdown[key]) > 0)
    .map(key => scoreComponentFor(key, breakdown[key]));

  if (v10Totals.length > 0) {
    return v10Totals;
  }

  return Object.entries(breakdown)
    .filter(([key, weight]) => isDisplayableScoreComponent(key, weight))
    .map(([key, weight]) => scoreComponentFor(key, weight));
}

export function canonicalDiscoveryScore(entry) {
  const discovery = finiteNumber(entry?.discovery_score);
  if (discovery != null) return discovery;
  return canonicalCompositeScore(entry);
}

export function canonicalDiscoveryScoreBreakdown(entry) {
  return entry?.discovery_score_breakdown && typeof entry.discovery_score_breakdown === 'object'
    ? entry.discovery_score_breakdown
    : {};
}

export function promotionEvidenceView(entry) {
  const evidence = entry?.promotion_evidence && typeof entry.promotion_evidence === 'object'
    ? entry.promotion_evidence
    : {};
  const score = finiteNumber(evidence.score) ?? 0;
  const std = finiteNumber(evidence.std);

  let label = 'Low';
  let color = 'var(--accent-red)';
  if (score >= 75) {
    label = 'High';
    color = 'var(--accent-green)';
  } else if (score >= 45) {
    label = 'Moderate';
    color = 'var(--accent-yellow)';
  }

  const uncertaintyLabel = std == null
    ? 'unknown'
    : std <= 0.05 ? 'tight'
      : std <= 0.12 ? 'bounded'
        : 'high';

  return {
    ...evidence,
    score,
    std,
    label,
    color,
    uncertaintyLabel,
    evidenceCount: Number(evidence.evidence_count ?? evidence.evidenceCount ?? 0),
    totalChecks: Number(evidence.total_checks ?? evidence.totalChecks ?? 0),
    seenRuns: Number(evidence.seen_runs ?? evidence.seenRuns ?? 0),
  };
}
