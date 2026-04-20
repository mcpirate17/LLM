function finiteNumber(value) {
  if (value == null) return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

export const CANONICAL_SCORE_COMPONENT_META = {
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
};

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
  return Object.entries(breakdown)
    .filter(([key, weight]) => Number.isFinite(Number(weight)) && Number(weight) > 0 && !String(key).includes('penalty'))
    .map(([key, weight]) => ({
      key,
      weight: Number(weight),
      ...(CANONICAL_SCORE_COMPONENT_META[key] || { label: key, color: 'var(--text-muted)' }),
    }));
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
