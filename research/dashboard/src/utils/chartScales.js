const SCALE_STORAGE_PREFIX = 'dashboard.chart_scale.v1.';

function clampValue(value, min, max) {
  if (value == null || Number.isNaN(value)) return min;
  return Math.min(Math.max(value, min), max);
}

function computeScale(values, defaultMin, defaultMax, padding = 0) {
  const filtered = (values || []).filter(v => v != null && isFinite(v));
  if (filtered.length === 0) {
    return { min: defaultMin, max: defaultMax };
  }
  let min = Math.min(...filtered);
  let max = Math.max(...filtered);
  if (padding > 0) {
    const range = max - min || 1;
    min -= range * padding;
    max += range * padding;
  }
  if (min === max) {
    min = defaultMin;
    max = defaultMax;
  }
  return { min, max };
}

export function getFixedScale(metric, values, options = {}) {
  const {
    defaultMin = 0,
    defaultMax = 1,
    padding = 0,
  } = options;

  if (typeof window !== 'undefined') {
    const key = `${SCALE_STORAGE_PREFIX}${metric}`;
    try {
      const raw = window.localStorage.getItem(key);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (typeof parsed?.min === 'number' && typeof parsed?.max === 'number') {
          return parsed;
        }
      }
      const scale = computeScale(values, defaultMin, defaultMax, padding);
      window.localStorage.setItem(key, JSON.stringify(scale));
      return scale;
    } catch {
      return computeScale(values, defaultMin, defaultMax, padding);
    }
  }

  return computeScale(values, defaultMin, defaultMax, padding);
}

export function clampToScale(value, scale) {
  if (!scale) return value;
  return clampValue(value, scale.min, scale.max);
}

export const CHART_DEFAULTS = {
  s1_rate: { min: 0, max: 1 },
  novelty: { min: 0, max: 1 },
  loss_ratio: { min: 0, max: 2 },
  baseline_ratio: { min: 0, max: 1.5 },
  training_loss: { min: 0, max: 2 },
  grammar_weight: { min: 0, max: 1 },
  throughput_tok_s: { min: 0, max: 50000 },
  step_time_ms: { min: 0, max: 500 },
  gpu_starvation_ms: { min: 0, max: 200 },
  programs: { min: 0, max: 200 },
  survivors: { min: 0, max: 100 },
  routing_entropy: { min: 0, max: 2.5 },
  routing_drop_rate: { min: 0, max: 1 },
  routing_confidence: { min: 0, max: 1 },
  routing_token_retention: { min: 0, max: 1 },
  depth_savings_ratio: { min: 0, max: 1 },
  effective_depth_ratio: { min: 0, max: 1 },
  recursion_savings_ratio: { min: 0, max: 1 },
  recursion_depth_ratio: { min: 0, max: 1 },
  efficiency_flops: { min: 0, max: 1e12 },
  efficiency_loss: { min: 0, max: 2 },
  efficiency_log_flops: { min: 6, max: 13 },
};
