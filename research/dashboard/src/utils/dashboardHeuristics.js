function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function roundScore(score) {
  return Math.round(Math.max(0, score));
}

function normalizeLossRatio(lossRatio) {
  return lossRatio != null ? clamp01(1 - lossRatio) : 0;
}

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
  const score = b.passRate === 0 ? raw * 0.5 : raw;
  return Number.isFinite(score) ? roundScore(score) : 0;
}

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

const CATEGORY_ORDER = { success_factor: 4, pattern: 3, hypothesis: 2, failure_mode: 1 };
const STATUS_ORDER = { confirmed: 3, active: 2, superseded: 1, refuted: 0 };

export function insightScore(insight) {
  const conf = (insight.confidence || 0.5) * 40;
  const cat = ((CATEGORY_ORDER[insight.category] || 0) / 4) * 30;
  const status = ((STATUS_ORDER[insight.status] || 0) / 3) * 20;
  const evidence = insight.supporting_evidence ? 10 : 0;
  return roundScore(conf + cat + status + evidence);
}
