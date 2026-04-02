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
  screened_out: 0,
};

export const TIER_COLORS = {
  screening: 'var(--accent-blue)',
  screened_out: 'var(--text-muted)',
  investigation: 'var(--accent-yellow)',
  validation: 'var(--accent-purple)',
  breakthrough: 'var(--accent-green)',
};

export const TIER_LABELS = {
  screening: 'Screening',
  screened_out: 'Failed Investigation',
  investigation: 'Investigation',
  validation: 'Validation',
  breakthrough: 'Breakthrough',
};

// ── GPT-2 reference metrics (measured on d_model=256, 6-layer config) ──

const GPT2_REF = {
  loss_ratio: 0.2646,
  param_count: 9_767_424,
  flops_forward: 19_534_848,
  throughput_tok_s: 1_200_845,
  peak_memory_mb: 115.0,
  forward_time_ms: 0.43,
};

/**
 * Geometric mean of per-dimension ratios vs GPT-2.
 * All ratios >1.0 = better than GPT-2.
 * Requires at least 3 of 6 dimensions. Returns null if insufficient data.
 */
function computeEfficiencyMultiple(entry) {
  const ref = GPT2_REF;
  const ratios = {};

  const loss = bestLossForEfficiency(entry);
  if (loss != null && loss > 0) ratios.x_quality = ref.loss_ratio / loss;

  const params = entry?.param_count ?? entry?.graph_n_params_estimate;
  if (params != null && params > 0) ratios.x_params = ref.param_count / params;

  const flops = entry?.flops_forward;
  if (flops != null && flops > 0) ratios.x_flops = ref.flops_forward / flops;

  const tput = entry?.throughput_tok_s;
  if (tput != null && tput > 0) ratios.x_throughput = tput / ref.throughput_tok_s;

  const mem = entry?.peak_memory_mb;
  if (mem != null && mem > 0) ratios.x_memory = ref.peak_memory_mb / mem;

  const lat = entry?.forward_time_ms;
  if (lat != null && lat > 0) ratios.x_latency = ref.forward_time_ms / lat;

  const keys = Object.keys(ratios);
  if (keys.length < 3) return null;

  let product = 1;
  for (const k of keys) product *= ratios[k];
  const geomean = Math.pow(product, 1.0 / keys.length);

  return { ...ratios, geomean, n_dimensions: keys.length };
}

/** Pick the best available loss ratio for efficiency computation. */
function bestLossForEfficiency(entry) {
  return pickFirstNumber(entry, [
    'validation_loss_ratio', 'investigation_loss_ratio',
    'screening_loss_ratio', 'loss_ratio',
  ]);
}

// ── Shared leaderboard helpers ──────────────────────────────────────

export function bestLoss(entry) {
  if (entry?.validation_loss_ratio != null) return Number(entry.validation_loss_ratio);
  if (entry?.investigation_loss_ratio != null) return Number(entry.investigation_loss_ratio);
  if (entry?.screening_loss_ratio != null) return Number(entry.screening_loss_ratio);
  if (entry?.loss_ratio != null) return Number(entry.loss_ratio);
  return null;
}

export function percentOfReference(entryLoss, refLoss) {
  const e = Number(entryLoss);
  const r = Number(refLoss);
  if (!Number.isFinite(e) || !Number.isFinite(r) || r <= 0) return null;
  return (e / r) * 100;
}

// ── Shared normalizers ──────────────────────────────────────────────

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

/**
 * Round a score to the nearest integer.  No upper cap — the score is the score.
 * Breakdown components are individually normalized to 0-100 by their callers.
 */
function roundScore(score) {
  return Math.round(Math.max(0, score));
}

/**
 * Normalize a loss ratio to 0-1 where lower loss = higher score.
 * Linear: 0.0 → 1.0, 0.5 → 0.5, 1.0 → 0.0. No floor — every improvement counts.
 */
function normalizeLossRatio(lossRatio) {
  return lossRatio != null ? clamp01(1 - lossRatio) : 0;
}

const BONUS_WEIGHTS = {
  efficiency: 6,
  routing: 7,
  adaptive: 6,
  sparsity: 5,
  learningSpeed: 4,
  externalComparison: 6,
  robustness: 7,
  referenceDelta: 8,
  binding: 10,
  blimp: 5,
};

// Maximum total bonus contribution (prevents bonus stacking from inflating scores)
const MAX_TOTAL_BONUS = 40;

const ARCHITECTURE_TARGETS = {
  moe: { capacity_multiplier: 4.0, flops_iso: true },
  mod: { throughput_multiplier: 2.0, accuracy_iso: true },
  mor: { accuracy_gain: 0.02, params_iso: true },
  mamba: { scaling: 'linear', throughput_vs_transformer: 5.0 },
};

// ── External baselines (efficiency multipliers relative to dense transformer) ──
// Sources: GPT-2/3, Switch Transformer, GShard, Mixtral, Mamba, Griffin, Jamba,
// MoD, PonderNet, FNet, gMLP, MLP-Mixer. Values conservative for 256 d_model scale.

const EXTERNAL_BASELINES = {
  'Attention':                { paramEfficiency: 1.0,  flopEfficiency: 1.0,  throughputRatio: 1.0, learningSpeedRatio: 1.0  },
  'Hybrid-Attention':         { paramEfficiency: 1.15, flopEfficiency: 1.05, throughputRatio: 0.9, learningSpeedRatio: 1.1  },
  'MoE-Attention':            { paramEfficiency: 3.5,  flopEfficiency: 1.0,  throughputRatio: 0.85, learningSpeedRatio: 1.2 },
  'Routed-MoE':               { paramEfficiency: 3.5,  flopEfficiency: 1.0,  throughputRatio: 0.85, learningSpeedRatio: 1.2 },
  'MoE-Hybrid-Attention':     { paramEfficiency: 3.0,  flopEfficiency: 0.95, throughputRatio: 0.8, learningSpeedRatio: 1.15 },
  'Mamba-SSM':                { paramEfficiency: 0.85, flopEfficiency: 1.2,  throughputRatio: 4.5, learningSpeedRatio: 0.9  },
  'Hybrid-SSM':               { paramEfficiency: 1.1,  flopEfficiency: 1.15, throughputRatio: 2.5, learningSpeedRatio: 1.1  },
  'MoE-Mamba-SSM':            { paramEfficiency: 3.0,  flopEfficiency: 1.1,  throughputRatio: 3.5, learningSpeedRatio: 1.05 },
  'Adaptive-Attention':       { paramEfficiency: 1.2,  flopEfficiency: 1.4,  throughputRatio: 1.5, learningSpeedRatio: 1.1  },
  'Adaptive-Hybrid-Attention':{ paramEfficiency: 1.25, flopEfficiency: 1.35, throughputRatio: 1.4, learningSpeedRatio: 1.1  },
  'Adaptive-Mamba-SSM':       { paramEfficiency: 0.9,  flopEfficiency: 1.5,  throughputRatio: 5.0, learningSpeedRatio: 0.95 },
  'Adaptive-MLP-Mixer':       { paramEfficiency: 1.3,  flopEfficiency: 1.1,  throughputRatio: 1.1, learningSpeedRatio: 1.05 },
  'Conv-Mixer':               { paramEfficiency: 0.95, flopEfficiency: 1.1,  throughputRatio: 1.2, learningSpeedRatio: 0.95 },
  'Spectral-Mixer':           { paramEfficiency: 0.9,  flopEfficiency: 1.05, throughputRatio: 1.15, learningSpeedRatio: 0.9 },
  'Spectral-Conv':            { paramEfficiency: 0.92, flopEfficiency: 1.08, throughputRatio: 1.2, learningSpeedRatio: 0.92 },
  'Gated-MLP':                { paramEfficiency: 0.85, flopEfficiency: 0.95, throughputRatio: 1.3, learningSpeedRatio: 0.85 },
  'MLP-Mixer':                { paramEfficiency: 0.8,  flopEfficiency: 0.9,  throughputRatio: 1.4, learningSpeedRatio: 0.8  },
  'Nonlinear-Mixer':          { paramEfficiency: 0.75, flopEfficiency: 0.85, throughputRatio: 1.5, learningSpeedRatio: 0.75 },
  'Hybrid-Mixer':             { paramEfficiency: 0.95, flopEfficiency: 1.0,  throughputRatio: 1.1, learningSpeedRatio: 0.95 },
};

/**
 * Resolve an architecture_family string to its external baseline entry.
 * Tries: exact match → progressive prefix stripping → longest substring → fallback.
 * Returns { key, baseline, fuzzy } or null if family is missing/Unknown.
 */
function resolveBaseline(family) {
  if (!family || family === 'Unknown') return null;

  // Exact match
  if (EXTERNAL_BASELINES[family]) {
    return { key: family, baseline: EXTERNAL_BASELINES[family], fuzzy: false };
  }

  // Progressive prefix stripping: "Adaptive-MoE-Attention" → "MoE-Attention" → "Attention"
  let stripped = family;
  while (stripped.includes('-')) {
    stripped = stripped.replace(/^[^-]+-/, '');
    if (EXTERNAL_BASELINES[stripped]) {
      return { key: stripped, baseline: EXTERNAL_BASELINES[stripped], fuzzy: true };
    }
  }

  // Longest-substring scan
  let bestKey = null;
  let bestLen = 0;
  for (const key of Object.keys(EXTERNAL_BASELINES)) {
    if (family.includes(key) && key.length > bestLen) {
      bestKey = key;
      bestLen = key.length;
    }
  }
  if (bestKey) {
    return { key: bestKey, baseline: EXTERNAL_BASELINES[bestKey], fuzzy: true };
  }

  // Fallback
  return { key: 'Hybrid-Mixer', baseline: EXTERNAL_BASELINES['Hybrid-Mixer'], fuzzy: true };
}

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
    scores.push(clamp01(procRate / 0.95));
  }

  const avg = averageScores(scores);
  // MoE quality multiplier: reward expert diversity and balanced utilization
  let moeFactor = 1.0;
  if (nExperts != null && nExperts > 1) {
    // Expert diversity: more experts = higher potential (log-scale)
    const expertMult = Math.min(1.5, 1.0 + Math.log2(nExperts) / 6.0);
    moeFactor *= expertMult;
    // Balanced utilization bonus
    if (entropyScore != null && entropyScore > 0.8) moeFactor *= 1.2;
  }
  // Confidence bonus: high confidence = routing is working
  if (confMean != null && Number(confMean) > 0.5) {
    moeFactor *= 1.0 + 0.3 * (Number(confMean) - 0.5);
  }
  // Drop rate penalty: high drop = wasted compute
  if (dropRate != null && Number(dropRate) > 0.3) {
    moeFactor *= Math.max(0.5, 1.0 - (Number(dropRate) - 0.3));
  }
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
    // Baseline throughput targets calibrated to current model performance
    const targetThroughput = (entry.routing_mode || entry.compute_routing === 'depth_token_mask') ? 50000 : 25000;
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
  // Target for MoD is 50% savings — stronger reward curve
  if (depthSavings != null) scores.push(clamp01(depthSavings / 0.5));

  const depthUtil = pickFirstNumber(entry, [
    'effective_depth_ratio',
    'depth_utilization_ratio',
    'avg_depth_ratio',
  ]);
  if (depthUtil != null) scores.push(clamp01(1 - depthUtil));

  const recursionSavings = pickFirstNumber(entry, [
    'recursion_savings_ratio',
    'recursion_compute_savings',
    'depth_weighted_proj_savings',
    'recursion_efficiency_gain',
  ]);
  // Recursion savings: 50% savings = full score
  if (recursionSavings != null) scores.push(clamp01(recursionSavings / 0.5));

  const avg = averageScores(scores);
  return avg == null ? null : avg * BONUS_WEIGHTS.adaptive;
}

function computeSparsityBonus(entry) {
  const scores = [];
  const sparsityRatio = entry?.sparsity_ratio;
  if (sparsityRatio != null) scores.push(clamp01(Number(sparsityRatio) / 0.5));

  const params = entry?.param_count ?? entry?.graph_n_params_estimate;
  const paramScore = normalizeInverseLog10(params, 4, 9);
  if (paramScore != null) scores.push(paramScore);

  const memory = entry?.peak_memory_mb;
  if (memory != null) scores.push(clamp01(1 - Number(memory) / 500));

  // Activation sparsity bonus (mirrors Python 3c)
  const activationSparsity = entry?.activation_sparsity_score;
  if (activationSparsity != null && Number(activationSparsity) > 0.3) {
    scores.push(clamp01((Number(activationSparsity) - 0.3) / 0.5));
  }

  const avg = averageScores(scores);
  if (avg == null) return null;
  const multiplier = (sparsityRatio != null && Number(sparsityRatio) > 0.5) ? 1.3 : 1.0;
  return avg * BONUS_WEIGHTS.sparsity * multiplier;
}

function computeLearningSpeedBonus(entry) {
  const scores = [];
  const lir = entry?.loss_improvement_rate;
  if (lir != null) scores.push(clamp01(Number(lir)));

  const throughput = entry?.throughput_tok_s;
  if (throughput != null) scores.push(clamp01(Number(throughput) / 25000));

  const forwardMs = entry?.forward_time_ms;
  if (forwardMs != null) scores.push(clamp01(1 - Number(forwardMs) / 50));

  const avg = averageScores(scores);
  return avg == null ? null : avg * BONUS_WEIGHTS.learningSpeed;
}

function computeExternalComparisonBonus(entry) {
  // If real scaling comparison data exists and shows poor efficiency,
  // don't award a bonus based on hardcoded baseline estimates.
  const scalingEff = entry?.scaling_param_efficiency;
  if (scalingEff != null && scalingEff < 1.5) return 0;

  const resolved = resolveBaseline(entry?.architecture_family);
  if (!resolved) return null;
  const { baseline } = resolved;
  const scores = [];

  // (a) Param efficiency sub-score
  const lossRatio = entry?.loss_ratio;
  const params = entry?.param_count ?? entry?.graph_n_params_estimate;
  if (lossRatio != null && params != null) {
    const learning = clamp01(1 - Number(lossRatio));
    const paramNorm = normalizeInverseLog10(params, PARAM_EXP_RANGE.min, PARAM_EXP_RANGE.max);
    if (paramNorm != null) {
      const actual = learning * paramNorm;
      const expected = 0.5 * baseline.paramEfficiency;
      scores.push(clamp01(actual / expected / 1.5));
    }
  }

  // (b) FLOP efficiency sub-score
  const flopsPerParam = entry?.flops_per_param ?? (
    entry?.flops_forward != null && params != null && Number(params) > 0
      ? Number(entry.flops_forward) / Number(params)
      : null
  );
  if (flopsPerParam != null && lossRatio != null) {
    const learning = clamp01(1 - Number(lossRatio));
    const flopNorm = normalizeInverseLog10(flopsPerParam, 0, 4);
    if (flopNorm != null) {
      const actual = learning * flopNorm;
      const expected = 0.5 * baseline.flopEfficiency;
      scores.push(clamp01(actual / expected / 1.5));
    }
  }

  // (c) Throughput sub-score
  const throughput = entry?.throughput_tok_s;
  if (throughput != null) {
    const expectedThroughput = 25000 * baseline.throughputRatio;
    scores.push(clamp01(Number(throughput) / expectedThroughput));
  }

  // (d) Learning speed sub-score
  const lir = entry?.loss_improvement_rate;
  if (lir != null) {
    const denseBaselineLIR = 0.5;
    const expectedLIR = denseBaselineLIR * baseline.learningSpeedRatio;
    scores.push(clamp01(Number(lir) / (expectedLIR * 1.5)));
  }

  const avg = averageScores(scores);
  if (avg == null) return null;

  const excellenceFactor = avg > 0.75 ? 1.3 : avg > 0.5 ? 1.1 : 1.0;
  return avg * BONUS_WEIGHTS.externalComparison * excellenceFactor;
}

// ── Robustness Scorers ──────────────────────────────────────────────

function computeRobustnessBonus(entry) {
  const noise = entry?.robustness_noise_score;
  const longCtx = entry?.robustness_long_ctx_score;
  const quant = entry?.quant_int8_retention;
  const initStd = entry?.init_sensitivity_std;
  const spectralNorm = entry?.jacobian_spectral_norm ?? entry?.fp_jacobian_spectral_norm;

  const scores = [];
  if (noise != null) scores.push(clamp01(1 - Number(noise)));
  if (longCtx != null) scores.push(clamp01(Number(longCtx)));
  if (quant != null) {
    const qPct = Number(quant) <= 1 ? Number(quant) : Number(quant) / 100;
    scores.push(clamp01((qPct - 0.5) / 0.5)); // 0.5 -> 0, 1.0 -> 1
  }
  if (initStd != null) scores.push(clamp01(1 - Number(initStd) / 0.2));
  if (spectralNorm != null) scores.push(clamp01(1 - Number(spectralNorm) / 20));

  // Long-context sub-scores (RULER-style) — boost robustness when detailed data available
  const passkey = entry?.robustness_long_ctx_passkey_score;
  const multiHop = entry?.robustness_long_ctx_multi_hop_score;
  const scaling = entry?.robustness_long_ctx_scaling_score;
  const assoc = entry?.robustness_long_ctx_assoc_score;
  if (passkey != null) scores.push(clamp01(Number(passkey)));
  if (multiHop != null) scores.push(clamp01(Number(multiHop)));
  if (scaling != null) scores.push(clamp01(Number(scaling)));
  if (assoc != null) scores.push(clamp01(Number(assoc)));

  const avg = averageScores(scores);
  return avg == null ? null : avg * BONUS_WEIGHTS.robustness;
}

function computeReferenceDeltaBonus(entry) {
  // If candidate explicitly beats a pinned reference in the same family/paradigm
  const blRatio = entry?.validation_baseline_ratio ?? entry?.baseline_loss_ratio;
  if (blRatio != null && blRatio < 0.90) {
    // 10% improvement over baseline baseline is worth a lot
    const gain = clamp01((1.0 - blRatio) / 0.2); // 0.9 -> 0.5, 0.8 -> 1.0
    return gain * BONUS_WEIGHTS.referenceDelta;
  }
  return 0;
}

/**
 * Routing overhead penalty: penalize when routing adds overhead without
 * improving loss. Mirrors Z14 in notebook.py compute_composite_score().
 * Returns a negative value (penalty) or 0.
 */
function computeRoutingOverheadPenalty(entry) {
  const savings = entry?.routing_savings_ratio;
  if (savings == null || savings >= 0.05) return 0;

  // Routing present but saves almost no compute
  const effectiveLR = entry?.validation_baseline_ratio
    ?? entry?.validation_loss_ratio
    ?? entry?.investigation_loss_ratio
    ?? entry?.screening_loss_ratio;

  if (effectiveLR == null || effectiveLR <= 0.95) return 0;

  // Loss barely improved — routing overhead not justified
  return -3 * (1.0 - savings / 0.05);  // Up to -3 points on 100-point scale
}

/** Sigmoid S-curve centered at ratio=1.0. Mirrors Python _scurve(). */
function _scurve(ratio, k = 4) {
  return 1.0 / (1.0 + Math.exp(-k * (ratio - 1.0)));
}

/** Binding probe bonus: 0.4*ar + 0.3*induction + 0.3*binding_auc, S-curved. */
function computeBindingBonus(entry) {
  const ar = entry?.ar_auc;
  const ind = entry?.induction_auc;
  const bind = entry?.binding_auc;

  if (ar == null && ind == null && bind == null) return null;

  let bc = 0;
  if (ar != null) bc += 0.4 * ar;
  if (ind != null) bc += 0.3 * ind;
  if (bind != null) bc += 0.3 * bind;

  if (bc <= 0) return 0;
  return _scurve(bc / 0.15, 6) * BONUS_WEIGHTS.binding;
}

/** BLiMP linguistic minimal pairs bonus. 50% = chance, S-curved at 60% frontier. */
function computeBlimpBonus(entry) {
  const acc = entry?.blimp_overall_accuracy;
  if (acc == null || acc <= 0.50) return 0;
  return _scurve(acc / 0.60, 6) * BONUS_WEIGHTS.blimp;
}

function computeBonusBreakdown(entry) {
  const raw = {
    efficiencyBonus: computeEfficiencyBonus(entry) ?? 0,
    routingBonus: computeRoutingBonus(entry) ?? 0,
    adaptiveBonus: computeAdaptiveBonus(entry) ?? 0,
    sparsityBonus: computeSparsityBonus(entry) ?? 0,
    learningSpeedBonus: computeLearningSpeedBonus(entry) ?? 0,
    externalComparisonBonus: computeExternalComparisonBonus(entry) ?? 0,
    robustnessBonus: computeRobustnessBonus(entry) ?? 0,
    referenceDeltaBonus: computeReferenceDeltaBonus(entry) ?? 0,
    bindingBonus: computeBindingBonus(entry) ?? 0,
    blimpBonus: computeBlimpBonus(entry) ?? 0,
    routingOverheadPenalty: computeRoutingOverheadPenalty(entry),
  };

  // Cap total bonus contribution to prevent score inflation
  const totalRaw = Object.values(raw).reduce((s, v) => s + v, 0);
  if (totalRaw > MAX_TOTAL_BONUS && totalRaw > 0) {
    const scale = MAX_TOTAL_BONUS / totalRaw;
    for (const key of Object.keys(raw)) {
      raw[key] *= scale;
    }
  }

  return raw;
}

/**
 * Scale bonus breakdown values to 0-100 range for display in ScoreBreakdown.
 * Raw bonus values are 0-10 (per-weight max), but core metrics use 0-100.
 * Without scaling, bonuses are invisible slivers in the proportional bar.
 */
function scaleBonusesForDisplay(bonuses) {
  const scaled = {};
  for (const [key, val] of Object.entries(bonuses)) {
    if (val === 0) continue;
    const maxWeight = BONUS_WEIGHTS[key.replace('Bonus', '').replace('Penalty', '')] || 10;
    if (val < 0) {
      // Penalties: scale negative values to -100..0 range
      scaled[key] = Math.round((val / maxWeight) * 100);
    } else {
      scaled[key] = Math.round((val / maxWeight) * 100);
    }
  }
  return scaled;
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
  const sLoss = entry.validation_loss_ratio ?? entry.screening_loss_ratio;
  const screeningLoss = normalizeLossRatio(sLoss);
  const novelty = entry.screening_novelty != null ? Math.min(entry.screening_novelty, 1.0) : 0;
  const investigationLoss = normalizeLossRatio(entry.investigation_loss_ratio);
  const robustness = entry.investigation_robustness != null ? Math.min(entry.investigation_robustness, 1.0) : 0;
  const validationBaseline = entry.validation_baseline_ratio != null ? clamp01(1.5 - entry.validation_baseline_ratio) : 0;
  const consistency = entry.validation_multi_seed_std != null ? Math.max(0, 1 - entry.validation_multi_seed_std * 10) : 0;
  const tierBonus = (tierOrder[entry.tier] || 0) / 4;
  const tier = entry.tier || 'screening';
  const bonus = scaleBonusesForDisplay(computeBonusBreakdown(entry));

  if (tier === 'breakthrough' || tier === 'validation') {
    return {
      sLoss: Math.round(screeningLoss * 100),
      novelty: Math.round(novelty * 100),
      iLoss: Math.round(investigationLoss * 100),
      robust: Math.round(robustness * 100),
      vBase: Math.round(validationBaseline * 100),
      consistency: Math.round(consistency * 100),
      tierBonus: Math.round(tierBonus * 100),
      ...bonus,
    };
  }

  if (tier === 'investigation') {
    return {
      sLoss: Math.round(screeningLoss * 100),
      novelty: Math.round(novelty * 100),
      iLoss: Math.round(investigationLoss * 100),
      robust: Math.round(robustness * 100),
      tierBonus: Math.round(tierBonus * 100),
      ...bonus,
    };
  }

  return {
    sLoss: Math.round(screeningLoss * 100),
    novelty: Math.round(novelty * 100),
    tierBonus: Math.round(tierBonus * 100),
    ...bonus,
  };
}

/**
 * Flat breakdown for programs that only have basic fields
 * (loss_ratio, novelty_score, baseline_loss_ratio, throughput_tok_s)
 * and no stage-prefixed columns.
 */
function flatBreakdown(program) {
  const lossRatio = program.validation_loss_ratio ?? program.loss_ratio;
  const lossScore = normalizeLossRatio(lossRatio);
  const noveltyScore = program.novelty_score != null ? Math.min(program.novelty_score, 1.0) : 0;
  const baselineScore = program.baseline_loss_ratio != null ? clamp01(1.5 - program.baseline_loss_ratio) : 0;
  const throughputScore = program.throughput_tok_s != null ? Math.min(program.throughput_tok_s / 25000, 1.0) : 0;
  const bonus = scaleBonusesForDisplay(computeBonusBreakdown(program));
  // Penalize programs that explicitly failed S1 (passed S0 but couldn't learn)
  const s1Penalty = program.stage1_passed === false || program.stage1_passed === 0 ? 0.5 : 1.0;

  return {
    loss: Math.round(lossScore * 100 * s1Penalty),
    novelty: Math.round(noveltyScore * 100),
    baseline: Math.round(baselineScore * 100 * s1Penalty),
    throughput: Math.round(throughputScore * 100),
    s1Penalty,
    ...bonus,
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
 * Unified candidate utility score (Scientific Utility).
 * This reflects the absolute merit of an architecture across all dimensions.
 */
export function candidateScore(entry) {
  let utility = 0.0;

  // 1. Performance Utility (Primary)
  // Use validation_baseline_ratio if available, otherwise fallback.
  // Apply confidence discount: screening-only metrics are less trustworthy.
  let perfLR, perfConfidence;
  if (entry.validation_baseline_ratio != null) {
    perfLR = entry.validation_baseline_ratio; perfConfidence = 1.0;
  } else if (entry.validation_loss_ratio != null) {
    perfLR = entry.validation_loss_ratio; perfConfidence = 1.0;
  } else if (entry.investigation_loss_ratio != null) {
    perfLR = entry.investigation_loss_ratio; perfConfidence = 0.85;
  } else if (entry.screening_loss_ratio != null) {
    perfLR = entry.screening_loss_ratio; perfConfidence = 0.65;
  } else if (entry.loss_ratio != null) {
    perfLR = entry.loss_ratio; perfConfidence = 0.65;
  } else {
    perfLR = null; perfConfidence = 0;
  }

  if (perfLR != null) {
    // Nonlinear utility curve: heavily reward strong loss ratios and
    // suppress mediocre survivors that only look good on novelty.
    const perfNorm = clamp01(1.0 - Number(perfLR));
    utility += 100.0 * Math.pow(perfNorm, 1.6) * perfConfidence;
  }
  
  // Discovery channel (random tokens)
  const discLR = entry.discovery_loss_ratio;
  if (discLR != null) {
    utility += 20.0 * Math.max(0, 1.0 - discLR);
  }

  // 2. Novelty Utility
  const novelty = entry.screening_novelty ?? entry.novelty_score ?? 0;
  const isRef = Boolean(entry.is_reference);
  const conf = entry.novelty_confidence ?? 1.0;
  const effectiveNov = isRef ? 1.0 : novelty;
  // Gate novelty by performance quality so novelty cannot dominate when
  // loss evidence is weak.
  let noveltyGate = 1.0;
  if (perfLR != null) {
    noveltyGate = clamp01((0.9 - Number(perfLR)) / 0.6);
  }
  utility += 40.0 * effectiveNov * conf * noveltyGate;

  // 3. Efficiency Utility — 5x TARGET
  const scalingEff = entry.scaling_param_efficiency;
  if (scalingEff != null) {
    const effAbove1 = Math.max(0, scalingEff - 1.0);
    utility += 25.0 * Math.sqrt(effAbove1);
    if (scalingEff >= 5.0) utility += 30.0;
    if (scalingEff >= 10.0) utility += 20.0;
  }

  // Compound efficiency bonus: routing + sparse + high efficiency
  const nRoutingOps = entry.n_routing_ops ?? 0;
  const nSparseOps = entry.n_sparse_ops ?? 0;
  if (nRoutingOps >= 1 && nSparseOps >= 1 && scalingEff != null && scalingEff >= 3.0) {
    utility += 15.0;
  }

  // Efficiency amplifier gate
  if (scalingEff != null && scalingEff > 3.0) {
    const amplifier = 1.0 + 0.05 * Math.min(5, scalingEff - 3.0);
    utility *= amplifier;
  }

  const savings = entry.routing_savings_ratio ?? entry.depth_savings_ratio;
  if (savings != null) {
    utility += 50.0 * savings;
  }
  
  const compRatio = entry.compression_ratio;
  if (compRatio != null) {
    utility += 20.0 * Math.max(0, 1.0 - compRatio);
  }

  const ncdScore = entry.ncd_score;
  if (ncdScore != null) {
    utility += 15.0 * Math.max(0, 1.0 - ncdScore);
  }

  // 4. Robustness & Stability Utility
  const spectral = entry.fp_jacobian_spectral_norm ?? entry.jacobian_spectral_norm;
  if (spectral != null) {
    // 1.0 -> 10 utility, 20.0 -> 0 utility
    utility += 10.0 * Math.max(0, 1.0 - (spectral / 20.0));
  }
  
  const noise = entry.robustness_noise_score;
  if (noise != null) {
    utility += 15.0 * Math.max(0, 1.0 - noise);
  }
  
  const quant = entry.quant_int8_retention;
  if (quant != null) {
    utility += 15.0 * Math.max(0, quant - 0.5) / 0.5;
  }
  
  // 4b. Expanded long-context scoring (total budget 50pts, up from 20)
  const longCtx = entry.robustness_long_ctx_score;
  if (longCtx != null) {
    utility += 20.0 * longCtx;
    // Sub-score bonuses: reward specific long-context capabilities
    const passkey = entry.robustness_long_ctx_passkey_score;
    if (passkey != null) utility += 10.0 * passkey;
    const multiHop = entry.robustness_long_ctx_multi_hop_score;
    if (multiHop != null) utility += 10.0 * multiHop;
    const lcScaling = entry.robustness_long_ctx_scaling_score;
    if (lcScaling != null) utility += 5.0 * lcScaling;
    const lcAssoc = entry.robustness_long_ctx_assoc_score;
    if (lcAssoc != null) utility += 5.0 * lcAssoc;
  }

  // Bonus for viable long sequences (log-scale, max 20pts)
  const maxSeqLen = entry.max_viable_seq_len;
  if (maxSeqLen != null && maxSeqLen > 512) {
    utility += 5.0 * Math.min(4.0, Math.log2(maxSeqLen / 512));
  }

  // 4c. Adaptive computation bonus (recursion + depth savings)
  const recursionSavings = entry.recursion_savings_ratio;
  if (recursionSavings != null && recursionSavings > 0) {
    utility += 15.0 * Math.min(1.0, recursionSavings / 0.5);
  }
  const depthSavings = entry.depth_savings_ratio;
  if (depthSavings != null && depthSavings > 0) {
    utility += 10.0 * Math.min(1.0, depthSavings / 0.5);
  }

  // 4d. Activation sparsity bonus
  const activationSparsity = entry.activation_sparsity_score;
  if (activationSparsity != null && activationSparsity > 0.3) {
    utility += 10.0 * Math.min(1.0, (activationSparsity - 0.3) / 0.5);
  }

  // 5. Penalties
  const std = entry.validation_multi_seed_std;
  if (std != null && std > 0.1) {
    utility -= 50.0 * Math.min(2.0, std / 0.5);
  }

  const vLR = entry.validation_loss_ratio;
  if (vLR != null && Number(vLR) >= 1.0) {
    utility -= 45.0 * Math.min(2.0, Number(vLR) - 1.0 + 0.5);
  }

  const vBase = entry.validation_baseline_ratio;
  if (vBase != null && Number(vBase) > 1.0) {
    utility -= 35.0 * Math.min(2.0, Number(vBase) - 1.0);
  }

  const iLR = entry.investigation_loss_ratio;
  if (iLR != null && Number(iLR) > 0.35) {
    utility -= 20.0 * Math.min(2.0, (Number(iLR) - 0.35) / 0.35);
  }
  
  const entropy = entry.routing_utilization_entropy;
  if (entropy != null && entropy > 0.8) {
    utility -= 10.0 * (entropy - 0.8);
  }

  // Scaling gate: stricter penalty for sub-baseline efficiency.
  // Models below 1x efficiency vs GPT-2 cannot plausibly beat GPT/Mamba by 5x.
  if (scalingEff != null && scalingEff < 1.0) {
    utility *= Math.max(0.1, scalingEff);
  }
  if (scalingEff != null && entry.scaling_gate_passed === 0) {
    const scalingPenalty = clamp01(scalingEff / 3.0);
    utility *= Math.max(0.3, scalingPenalty);
  }

  return Math.round(Math.max(0, utility));
}


// ── Discovery score (ResearchReport) ────────────────────────────────

export function discoveryScoreBreakdown(program) {
  const loss = normalizeLossRatio(program.loss_ratio) * 30;
  const novelty = program.novelty_score != null ? Math.min(program.novelty_score, 1.0) * 20 : 0;
  const baseline = program.baseline_loss_ratio != null ? clamp01(1.5 - program.baseline_loss_ratio) * 25 : 0;
  const id = program.most_similar_to ? 5 : 0;
  const params = program.param_count ?? program.graph_n_params_estimate;
  const paramEfficiency = normalizeInverseLog10(params, 4, 9);
  const paramEff = (paramEfficiency != null ? paramEfficiency : 0) * 10;
  const learningSpeed = program.loss_improvement_rate != null ? clamp01(program.loss_improvement_rate) * 10 : 0;
  const bonus = computeBonusBreakdown(program);
  const bonusTotal = Object.values(bonus).reduce((s, v) => s + v, 0);
  let total = loss + novelty + baseline + id + paramEff + learningSpeed + bonusTotal;

  // Scaling gate: sub-1x efficiency penalty (same as candidateScore)
  const scalingEff = program?.scaling_param_efficiency;
  if (scalingEff != null && scalingEff < 1.0) {
    total *= Math.max(0.1, scalingEff);
  }
  if (scalingEff != null && program?.scaling_gate_passed === 0) {
    const scalingPenalty = clamp01(scalingEff / 3.0);
    total *= Math.max(0.3, scalingPenalty);
  }

  // Normalize breakdown components to 0-100 by dividing by each category's max
  return {
    total: roundScore(total),
    loss: Math.round(loss / 30 * 100),
    novelty: Math.round(novelty / 20 * 100),
    baseline: Math.round(baseline / 25 * 100),
    id: Math.round(id / 5 * 100),
    paramEfficiency: Math.round(paramEff / 10 * 100),
    learningSpeed: Math.round(learningSpeed / 10 * 100),
    ...bonus,
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
function experimentScoreBreakdown(exp) {
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

function experimentScore(exp) {
  const b = experimentScoreBreakdown(exp);
  return roundScore(b.total);
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
  const passRate = Math.min(stabilizedS1Rate / 0.10, 1.0) * 30;
  const loss = d.best_loss_ratio != null
    ? normalizeLossRatio(d.best_loss_ratio) * 25
    : 0;
  const novelty = d.best_novelty_score != null
    ? Math.min(d.best_novelty_score, 1.0) * 20
    : 0;
  const efficiency = (d.duration_seconds && d.n_programs_generated)
    ? Math.min((d.n_programs_generated / d.duration_seconds) / 2, 1.0) * 15
    : 0;
  const learningSpeed = d.best_loss_improvement_rate != null
    ? clamp01(d.best_loss_improvement_rate) * 10
    : 0;
  const reliabilityMultiplier = 0.5 + 0.5 * (d.trend_weight != null ? d.trend_weight : 1.0);
  return { passRate, loss, novelty, efficiency, learningSpeed, reliabilityMultiplier };
}

export function trendScore(d) {
  const b = trendScoreBreakdown(d);
  const raw = (b.passRate + b.loss + b.novelty + b.efficiency + b.learningSpeed) * b.reliabilityMultiplier;
  // Penalize experiments with zero S1 survivors (same logic as experimentScore)
  const score = b.passRate === 0 ? raw * 0.5 : raw;
  return Number.isFinite(score) ? roundScore(score) : 0;
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
  return roundScore(b.s1 + b.s05 + b.s0 + b.novelty + b.usage);
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
  return roundScore(conf + cat + status + evidence);
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
