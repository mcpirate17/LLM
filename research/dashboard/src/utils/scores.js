const DEFAULT_TIER_ORDER = {
  breakthrough: 4,
  validation: 3,
  investigation: 2,
  screening: 1,
};

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function normalizeLossRatio(lossRatio) {
  return lossRatio != null ? Math.max(0, 1 - (lossRatio - 0.2) / 0.8) : 0;
}

export function programScoreBreakdown(program) {
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

export function programScore(program) {
  const breakdown = programScoreBreakdown(program);
  const score = breakdown.loss + breakdown.novelty + breakdown.baseline + breakdown.throughput;
  return Math.round(Math.max(0, Math.min(100, score)));
}

export function leaderboardEntryScoreBreakdown(entry, tierOrder = DEFAULT_TIER_ORDER) {
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

export function discoveryScoreBreakdown(program) {
  const loss = normalizeLossRatio(program.loss_ratio) * 35;
  const novelty = program.novelty_score != null ? Math.min(program.novelty_score, 1.0) * 25 : 0;
  const baseline = program.baseline_loss_ratio != null ? clamp01(1.5 - program.baseline_loss_ratio) * 30 : 0;
  const id = program.most_similar_to ? 10 : 0;
  const total = Math.round(Math.max(0, Math.min(100, loss + novelty + baseline + id)));
  return { total, loss, novelty, baseline, id };
}

export function discoveryScore(program) {
  return discoveryScoreBreakdown(program).total;
}

export function leaderboardEntryScore(entry, tierOrder = DEFAULT_TIER_ORDER) {
  const breakdown = leaderboardEntryScoreBreakdown(entry, tierOrder);
  const score = Object.values(breakdown).reduce((total, value) => total + value, 0);
  return Math.round(Math.max(0, Math.min(100, score)));
}
