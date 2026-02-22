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

const BONUS_WEIGHTS = {
  efficiency: 12,
  routing: 10,
  adaptive: 8,
};

const ARCHITECTURE_TARGETS = {
  moe: { capacity_multiplier: 4.0, flops_iso: true },
  mod: { throughput_multiplier: 2.0, accuracy_iso: true },
  mor: { accuracy_gain: 0.02, params_iso: true },
  mamba: { scaling: 'linear', throughput_vs_transformer: 5.0 },
};

const PARAM_EXP_RANGE = { min: 5, max: 10 };
const FLOPS_EXP_RANGE = { min: 6, max: 13 };

function normalizeInverseLog10(value, minExp, maxExp) {
  if (value == null || value <= 0) return null;
  const exp = Math.log10(value);
  const score = 1 - (exp - minExp) / (maxExp - minExp);
  return clamp01(score);
}

function parseJsonValue(value) {
  if (!value) return null;
  if (typeof value === 'object') return value;
  if (typeof value === 'string') {
    try {
      return JSON.parse(value);
    } catch (err) {
      return null;
    }
  }
  return null;
}

function pickFirstNumber(entry, keys) {
  for (const key of keys) {
    const value = entry?.[key];
    if (value == null) continue;
    const num = Number(value);
    if (Number.isFinite(num)) return num;
  }
  return null;
}

function averageScores(scores) {
  if (!scores.length) return null;
  const total = scores.reduce((sum, v) => sum + v, 0);
  return total / scores.length;
}

function getExpertCount(entry) {
  const direct = pickFirstNumber(entry, ['routing_expert_count', 'expert_count', 'n_experts']);
  if (direct != null) return direct;
  const parsed = parseJsonValue(entry?.routing_expert_utilization_json);
  if (Array.isArray(parsed)) return parsed.length;
  if (parsed && typeof parsed === 'object') return Object.keys(parsed).length;
  return null;
}

function normalizeRoutingEntropy(entropy, nExperts) {
  if (entropy == null) return null;
  if (nExperts != null && nExperts > 1) {
    const maxEntropy = Math.log2(nExperts);
    if (maxEntropy > 0) return clamp01(entropy / maxEntropy);
  }
  return clamp01(entropy);
}

function computeRoutingBonus(entry) {
  const entropy = entry?.routing_utilization_entropy;
  const dropRate = entry?.routing_drop_rate;
  const overflow = entry?.routing_capacity_overflow_count;
  const confMean = entry?.routing_confidence_mean;
  const confStd = entry?.routing_confidence_std;
  const tokensTotal = entry?.routing_tokens_total;
  const tokensProcessed = entry?.routing_tokens_processed;

  const scores = [];
  const nExperts = getExpertCount(entry);
  const entropyScore = normalizeRoutingEntropy(entropy, nExperts);
  if (entropyScore != null) scores.push(entropyScore);
  if (dropRate != null) scores.push(clamp01(1 - Number(dropRate)));
  if (overflow != null) scores.push(clamp01(1 - Math.min(Number(overflow) / 5, 1)));
  if (confMean != null) scores.push(clamp01(Number(confMean)));
  if (confStd != null) scores.push(clamp01(1 - Number(confStd) / 0.3));
  if (tokensTotal && tokensProcessed) {
    const procRate = Number(tokensProcessed) / Number(tokensTotal);
    // Reward processed rate, especially if it meets high utilization targets
    scores.push(clamp01(procRate / 0.95));
  }

  const avg = averageScores(scores);
  // Apply MoE multiplier if expert utilization is balanced (high entropy)
  const moeFactor = (entropyScore && entropyScore > 0.8) ? 1.2 : 1.0;
  return avg == null ? null : avg * BONUS_WEIGHTS.routing * moeFactor;
}

function computeEfficiencyBonus(entry) {
  const scores = [];
  const params = entry?.param_count ?? entry?.graph_n_params_estimate;
  const flops = entry?.flops_forward;
  const throughput = entry?.throughput_tok_s;

  const paramScore = normalizeInverseLog10(params, PARAM_EXP_RANGE.min, PARAM_EXP_RANGE.max);
  if (paramScore != null) scores.push(paramScore);
  const flopsScore = normalizeInverseLog10(flops, FLOPS_EXP_RANGE.min, FLOPS_EXP_RANGE.max);
  if (flopsScore != null) scores.push(flopsScore);
  
  if (throughput != null) {
    // Baseline target for dense is 5000 tok/s; Mamba/MoD targets 10000+
    const targetThroughput = (entry.routing_mode || entry.compute_routing === 'mod_topk') ? 10000 : 5000;
    scores.push(clamp01(Number(throughput) / targetThroughput));
  }

  const avg = averageScores(scores);
  return avg == null ? null : avg * BONUS_WEIGHTS.efficiency;
}

function computeAdaptiveBonus(entry) {
  const scores = [];
  const depthSavings = pickFirstNumber(entry, [
    'depth_savings_ratio',
    'adaptive_depth_savings',
    'depth_compute_savings',
    'depth_efficiency_gain',
  ]);
  // Target for MoD is 50% savings
  if (depthSavings != null) scores.push(clamp01(depthSavings / ARCHITECTURE_TARGETS.mod.throughput_multiplier * 2));

  const depthUtil = pickFirstNumber(entry, [
    'effective_depth_ratio',
    'depth_utilization_ratio',
    'avg_depth_ratio',
  ]);
  if (depthUtil != null) scores.push(clamp01(1 - depthUtil));

  const recursionSavings = pickFirstNumber(entry, [
    'recursion_savings_ratio',
    'recursion_compute_savings',
    'adaptive_recursion_savings',
    'recursion_efficiency_gain',
  ]);
  if (recursionSavings != null) scores.push(clamp01(recursionSavings));

  const avg = averageScores(scores);
  return avg == null ? null : avg * BONUS_WEIGHTS.adaptive;
}

function computeBonusBreakdown(entry) {
  return {
    efficiencyBonus: computeEfficiencyBonus(entry) ?? 0,
    routingBonus: computeRoutingBonus(entry) ?? 0,
    adaptiveBonus: computeAdaptiveBonus(entry) ?? 0,
  };
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
  const bonus = computeBonusBreakdown(entry);

  if (tier === 'breakthrough' || tier === 'validation') {
    return {
      sLoss: screeningLoss * 10,
      novelty: novelty * 10,
      iLoss: investigationLoss * 10,
      robust: robustness * 10,
      vBase: validationBaseline * 25,
      consistency: consistency * 15,
      tierBonus: tierBonus * 20,
      efficiencyBonus: bonus.efficiencyBonus,
      routingBonus: bonus.routingBonus,
      adaptiveBonus: bonus.adaptiveBonus,
    };
  }

  if (tier === 'investigation') {
    return {
      sLoss: screeningLoss * 15,
      novelty: novelty * 15,
      iLoss: investigationLoss * 20,
      robust: robustness * 15,
      tierBonus: tierBonus * 35,
      efficiencyBonus: bonus.efficiencyBonus,
      routingBonus: bonus.routingBonus,
      adaptiveBonus: bonus.adaptiveBonus,
    };
  }

  return {
    sLoss: screeningLoss * 35,
    novelty: novelty * 25,
    tierBonus: tierBonus * 40,
    efficiencyBonus: bonus.efficiencyBonus,
    routingBonus: bonus.routingBonus,
    adaptiveBonus: bonus.adaptiveBonus,
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
  const bonus = computeBonusBreakdown(program);

  return {
    loss: lossScore * 35,
    novelty: noveltyScore * 25,
    baseline: baselineScore * 25,
    throughput: throughputScore * 15,
    efficiencyBonus: bonus.efficiencyBonus,
    routingBonus: bonus.routingBonus,
    adaptiveBonus: bonus.adaptiveBonus,
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
  const bonus = computeBonusBreakdown(program);
  const total = clampScore(loss + novelty + baseline + id + bonus.efficiencyBonus + bonus.routingBonus + bonus.adaptiveBonus);
  return {
    total,
    loss,
    novelty,
    baseline,
    id,
    efficiencyBonus: bonus.efficiencyBonus,
    routingBonus: bonus.routingBonus,
    adaptiveBonus: bonus.adaptiveBonus,
  };
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
    return { passRate: 10, loss: 0, novelty: 0, completion: 5, quality: 0 };
  }

  const n = exp.n_programs_generated || 0;
  const s1 = exp.n_stage1_passed || 0;
  
  // Z15: More rigorous pass rate (linear scale up to 100%)
  const passRate = n > 0 ? (s1 / n) * 30 : 0;
  
  // Quality component (30 points)
  const lossScore = exp.best_loss_ratio != null
    ? normalizeLossRatio(exp.best_loss_ratio) * 15
    : 0;
  const noveltyScore = exp.best_novelty_score != null
    ? Math.min(exp.best_novelty_score, 1.0) * 15
    : 0;
  const quality = lossScore + noveltyScore;

  // Completion bonus (5 points)
  const completion = exp.status === 'completed' ? 5 : 0;
  
  // Discovery bonus (35 points) - rewarded for finding survivors
  const discovery = s1 > 0 ? Math.min(s1 * 5, 35) : 0;

  let total = passRate + quality + completion + discovery;
  
  // Strict penalty for total failure (compiled but nothing learned)
  if (exp.status === 'completed' && s1 === 0) {
    total = total * 0.5; // 50% penalty
  }

  return { passRate, quality, completion, discovery, total };
}

export function experimentScore(exp) {
  const b = experimentScoreBreakdown(exp);
  return clampScore(b.total);
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
  const score = (b.passRate + b.loss + b.novelty + b.efficiency) * b.reliabilityMultiplier;
  return Number.isFinite(score) ? clampScore(score) : 0;
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
