/**
 * Centralised scoring engine.
 *
 * Every 0-100 score displayed in the dashboard is computed here.
 * Components import from this single module so that the same candidate
 * always receives the same score regardless of which tab renders it.
 */

// ── Shared constants ────────────────────────────────────────────────

export const TIER_ORDER = {
  breakthrough: 4,
  validation: 3,
  investigation: 2,
  screening: 1,
};

// ── Shared normalizers ──────────────────────────────────────────────

export function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function clampScore(score) {
  return Math.round(Math.max(0, Math.min(100, score)));
}

/**
 * Normalize a loss ratio to 0-1 where lower loss = higher score.
 * Maps 0.2 → 1.0, 1.0 → 0.0.
 */
export function normalizeLossRatio(lossRatio) {
  return lossRatio != null ? Math.max(0, 1 - (lossRatio - 0.2) / 0.8) : 0;
}

// ── Candidate score (programs + leaderboard entries) ────────────────
//
// This is the ONE score function for individual architectures.
// It detects which pipeline stage the candidate has reached and applies
// tier-appropriate weights so a breakthrough candidate scores identically
// whether viewed in TopPrograms, Leaderboard, or ExperimentDetail.

function hasTieredFields(entry) {
  return entry.screening_loss_ratio != null
    || entry.investigation_loss_ratio != null
    || entry.validation_baseline_ratio != null;
}

/**
 * Tier-aware candidate score breakdown (validation/investigation/screening).
 * Used when the data includes stage-prefixed fields from the leaderboard or
 * program_results table.
 */
function tieredBreakdown(entry, tierOrder) {
  const screeningLoss = normalizeLossRatio(entry.screening_loss_ratio);
  const novelty = entry.screening_novelty != null ? Math.min(entry.screening_novelty, 1.0) : 0;
  const investigationLoss = normalizeLossRatio(entry.investigation_loss_ratio);
  const robustness = entry.investigation_robustness != null ? Math.min(entry.investigation_robustness, 1.0) : 0;
  const validationBaseline = entry.validation_baseline_ratio != null ? clamp01(1.5 - entry.validation_baseline_ratio) : 0;
  const consistency = entry.validation_multi_seed_std != null ? Math.max(0, 1 - entry.validation_multi_seed_std * 10) : 0;
  const tierBonus = (tierOrder[entry.tier] || 0) / 4;
  const tier = entry.tier || 'screening';

  if (tier === 'breakthrough' || tier === 'validation') {
    return {
      sLoss: screeningLoss * 10,
      novelty: novelty * 10,
      iLoss: investigationLoss * 10,
      robust: robustness * 10,
      vBase: validationBaseline * 25,
      consistency: consistency * 15,
      tierBonus: tierBonus * 20,
    };
  }

  if (tier === 'investigation') {
    return {
      sLoss: screeningLoss * 15,
      novelty: novelty * 15,
      iLoss: investigationLoss * 20,
      robust: robustness * 15,
      tierBonus: tierBonus * 35,
    };
  }

  return {
    sLoss: screeningLoss * 35,
    novelty: novelty * 25,
    tierBonus: tierBonus * 40,
  };
}

/**
 * Flat breakdown for programs that only have basic fields
 * (loss_ratio, novelty_score, baseline_loss_ratio, throughput_tok_s)
 * and no stage-prefixed columns.
 */
function flatBreakdown(program) {
  const lossScore = normalizeLossRatio(program.loss_ratio);
  const noveltyScore = program.novelty_score != null ? Math.min(program.novelty_score, 1.0) : 0;
  const baselineScore = program.baseline_loss_ratio != null ? clamp01(1.5 - program.baseline_loss_ratio) : 0;
  const throughputScore = program.throughput_tok_s != null ? Math.min(program.throughput_tok_s / 5000, 1.0) : 0;

  return {
    loss: lossScore * 35,
    novelty: noveltyScore * 25,
    baseline: baselineScore * 25,
    throughput: throughputScore * 15,
  };
}

/**
 * Unified candidate score breakdown.
 * Automatically selects tiered vs flat formula based on available fields.
 */
export function candidateScoreBreakdown(entry, tierOrder = TIER_ORDER) {
  if (hasTieredFields(entry)) {
    return tieredBreakdown(entry, tierOrder);
  }
  return flatBreakdown(entry);
}

/**
 * Unified candidate score (0-100).
 * This is the canonical score for any individual architecture.
 */
export function candidateScore(entry, tierOrder = TIER_ORDER) {
  const breakdown = candidateScoreBreakdown(entry, tierOrder);
  const score = Object.values(breakdown).reduce((sum, v) => sum + v, 0);
  return clampScore(score);
}

// Backward-compatible aliases
export const programScore = candidateScore;
export const programScoreBreakdown = candidateScoreBreakdown;
export const leaderboardEntryScore = candidateScore;
export const leaderboardEntryScoreBreakdown = candidateScoreBreakdown;

// ── Discovery score (ResearchReport) ────────────────────────────────

export function discoveryScoreBreakdown(program) {
  const loss = normalizeLossRatio(program.loss_ratio) * 35;
  const novelty = program.novelty_score != null ? Math.min(program.novelty_score, 1.0) * 25 : 0;
  const baseline = program.baseline_loss_ratio != null ? clamp01(1.5 - program.baseline_loss_ratio) * 30 : 0;
  const id = program.most_similar_to ? 10 : 0;
  const total = clampScore(loss + novelty + baseline + id);
  return { total, loss, novelty, baseline, id };
}

export function discoveryScore(program) {
  return discoveryScoreBreakdown(program).total;
}

// ── Experiment score (ExperimentList) ───────────────────────────────

/**
 * Score an experiment run 0-100.
 * Weights: S1 pass rate (40%), best loss (30%), best novelty (20%), completion (10%).
 */
export function experimentScoreBreakdown(exp) {
  if (exp?.status === 'running' && exp?.experiment_type === 'validation') {
    return { passRate: 10, loss: 0, novelty: 0, completion: 15 };
  }

  const n = exp.n_programs_generated || 0;
  const s1 = exp.n_stage1_passed || 0;
  const passRate = n > 0 ? Math.min(s1 / n / 0.10, 1.0) * 40 : 0;
  const loss = exp.best_loss_ratio != null
    ? normalizeLossRatio(exp.best_loss_ratio) * 30
    : 0;
  const novelty = exp.best_novelty_score != null
    ? Math.min(exp.best_novelty_score, 1.0) * 20
    : 0;
  const completion = exp.status === 'completed' ? 10 : 0;

  return { passRate, loss, novelty, completion };
}

export function experimentScore(exp) {
  const b = experimentScoreBreakdown(exp);
  return clampScore(b.passRate + b.loss + b.novelty + b.completion);
}

// ── Trend score (TrendCharts) ───────────────────────────────────────

/**
 * Score a trend data point 0-100.
 * Weights: S1 rate (35%), loss (30%), novelty (25%), efficiency (10%).
 * Applies reliabilityMultiplier from trend_weight when available.
 */
export function trendScoreBreakdown(d) {
  const stabilizedS1Rate = d.adjusted_s1_pass_rate != null
    ? d.adjusted_s1_pass_rate
    : (d.s1_pass_rate || 0);
  const passRate = Math.min(stabilizedS1Rate / 0.10, 1.0) * 35;
  const loss = d.best_loss_ratio != null
    ? normalizeLossRatio(d.best_loss_ratio) * 30
    : 0;
  const novelty = d.best_novelty_score != null
    ? Math.min(d.best_novelty_score, 1.0) * 25
    : 0;
  const efficiency = (d.duration_seconds && d.n_programs_generated)
    ? Math.min((d.n_programs_generated / d.duration_seconds) / 2, 1.0) * 10
    : 0;
  const reliabilityMultiplier = 0.5 + 0.5 * (d.trend_weight != null ? d.trend_weight : 1.0);
  return { passRate, loss, novelty, efficiency, reliabilityMultiplier };
}

export function trendScore(d) {
  const b = trendScoreBreakdown(d);
  return clampScore((b.passRate + b.loss + b.novelty + b.efficiency) * b.reliabilityMultiplier);
}

// ── Op score (LearningPanel) ────────────────────────────────────────

/**
 * Score an operation's success profile 0-100.
 * Weights: S1 rate (40%), S0.5 rate (20%), S0 rate (10%), novelty (20%), usage (10%).
 */
export function opScoreBreakdown(stats) {
  const s1 = Math.min((stats.s1_rate || 0) / 0.15, 1.0) * 40;
  const s05 = Math.min((stats.s05_rate || 0), 1.0) * 20;
  const s0 = Math.min((stats.s0_rate || 0), 1.0) * 10;
  const novelty = Math.min((stats.avg_novelty || 0), 1.0) * 20;
  const usage = Math.min((stats.n_used || 0) / 100, 1.0) * 10;
  return { s1, s05, s0, novelty, usage };
}

export function opScore(stats) {
  const b = opScoreBreakdown(stats);
  return clampScore(b.s1 + b.s05 + b.s0 + b.novelty + b.usage);
}

// ── Insight score (InsightsPanel) ───────────────────────────────────

const CATEGORY_ORDER = { success_factor: 4, pattern: 3, hypothesis: 2, failure_mode: 1 };
const STATUS_ORDER = { confirmed: 3, active: 2, superseded: 1, refuted: 0 };

/**
 * Score an insight 0-100.
 * Weights: confidence (40%), category importance (30%), status (20%), evidence (10%).
 */
export function insightScore(insight) {
  const conf = (insight.confidence || 0.5) * 40;
  const cat = ((CATEGORY_ORDER[insight.category] || 0) / 4) * 30;
  const status = ((STATUS_ORDER[insight.status] || 0) / 3) * 20;
  const evidence = insight.supporting_evidence ? 10 : 0;
  return clampScore(conf + cat + status + evidence);
}

// ── Promotion evidence (Leaderboard + ResearchReport) ───────────────

/**
 * Compute promotion-readiness evidence for a candidate.
 * Returns { label, color, score, seenRuns, std, uncertaintyLabel,
 *           evidenceCount, totalChecks, missing }.
 *
 * Accepts either a leaderboard entry or a program_results row —
 * uses validation_baseline_ratio when available, falls back to baseline_loss_ratio.
 */
export function promotionEvidence(entry) {
  const seenRuns = Number(entry?.cross_run_stability?.seen_runs || 0);
  const rawBaseline = entry.validation_baseline_ratio ?? entry.baseline_loss_ratio;
  const baselineRatioValue = Number(rawBaseline);
  const stdValue = Number(entry?.validation_multi_seed_std);
  const baselineRatio = Number.isFinite(baselineRatioValue) ? baselineRatioValue : null;
  const std = Number.isFinite(stdValue) ? stdValue : null;

  const checks = {
    lossEvidence: entry?.loss_ratio != null || entry?.screening_loss_ratio != null,
    noveltyEvidence: entry?.novelty_score != null || entry?.screening_novelty != null,
    baselineEvidence: baselineRatio != null,
    baselineBeat: baselineRatio != null && baselineRatio < 1.0,
    multiSeedStd: std != null,
    boundedStd: std != null && std <= 0.12,
    ckaArtifactBacked: entry?.cka_source === 'artifact',
    repeatObserved: seenRuns >= 3,
  };

  const totalChecks = Object.keys(checks).length;
  const evidenceCount = Object.values(checks).filter(Boolean).length;
  const completeness = evidenceCount / totalChecks;

  const stdSignal = std == null ? 0 : std <= 0.05 ? 1 : std <= 0.12 ? 0.65 : std <= 0.2 ? 0.35 : 0.1;
  const repeatSignal = seenRuns >= 5 ? 1 : seenRuns >= 3 ? 0.65 : seenRuns >= 2 ? 0.4 : seenRuns >= 1 ? 0.2 : 0;
  const margin = baselineRatio == null ? null : 1 - baselineRatio;
  const marginSignal = margin == null ? 0 : margin >= 0.1 ? 1 : margin > 0 ? 0.7 : 0.15;

  const score = Math.round((completeness * 0.5 + stdSignal * 0.2 + repeatSignal * 0.2 + marginSignal * 0.1) * 100);

  const confidence = score >= 75
    ? { label: 'High', color: 'var(--accent-green)' }
    : score >= 45
      ? { label: 'Moderate', color: 'var(--accent-yellow)' }
      : { label: 'Low', color: 'var(--accent-red)' };

  const uncertaintyLabel = std == null
    ? 'unknown'
    : std <= 0.05 ? 'tight'
      : std <= 0.12 ? 'bounded'
        : 'high';

  const missing = Object.entries(checks)
    .filter(([, ok]) => !ok)
    .map(([name]) => name);

  return {
    ...confidence,
    score,
    seenRuns,
    std,
    uncertaintyLabel,
    evidenceCount,
    totalChecks,
    missing,
  };
}
