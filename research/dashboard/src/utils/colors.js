/**
 * Shared color utility functions for consistent metric coloring across dashboard components.
 * Each function maps a numeric value to a CSS color variable.
 */

/** Loss ratio: lower is better. <0.5 green, <0.7 yellow, else orange */
export function lossColor(ratio) {
  if (ratio == null) return 'var(--text-muted)';
  if (ratio < 0.5) return 'var(--accent-green)';
  if (ratio < 0.7) return 'var(--accent-yellow)';
  return 'var(--accent-orange, #f0883e)';
}

/** Novelty score: higher is better. >0.8 green, >0.5 yellow, else muted */
export function noveltyColor(score) {
  const v = score || 0;
  if (v > 0.8) return 'var(--accent-green)';
  if (v > 0.5) return 'var(--accent-yellow)';
  return 'var(--text-muted)';
}

/** Confidence: higher is better. >=0.7 green, >=0.4 yellow, else red */
export function confidenceColor(confidence) {
  const v = confidence || 0;
  if (v >= 0.7) return 'var(--accent-green)';
  if (v >= 0.4) return 'var(--accent-yellow)';
  return 'var(--accent-red)';
}

/** Reliability level: high=green, medium=yellow, else red */
export function reliabilityColor(level) {
  if (level === 'high') return 'var(--accent-green)';
  if (level === 'medium') return 'var(--accent-yellow)';
  return 'var(--accent-red)';
}

function finiteNumber(value) {
  if (value == null) return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

export function boundedMetricColor(value, stops, fallback = 'var(--text-muted)') {
  const num = finiteNumber(value);
  if (num == null) return fallback;
  for (const [min, color] of stops) {
    if (num >= min) return color;
  }
  return fallback;
}

export function pplColor(value) {
  const num = finiteNumber(value);
  if (num == null) return 'var(--text-muted)';
  if (num <= 5) return 'var(--score-champion, #ffd166)';
  if (num <= 8) return 'var(--score-elite, #e3b341)';
  if (num <= 12) return 'var(--score-reference, #2dd4bf)';
  if (num <= 25) return 'var(--score-contender, #58a6ff)';
  return 'var(--text-secondary)';
}

export function hellaswagColor(value) {
  return boundedMetricColor(value, [
    [0.31, 'var(--score-champion, #ffd166)'],
    [0.28, 'var(--score-elite, #e3b341)'],
    [0.25, 'var(--score-reference, #2dd4bf)'],
    [0.18, 'var(--score-contender, #58a6ff)'],
  ]);
}

export function blimpColor(value) {
  return boundedMetricColor(value, [
    [0.60, 'var(--score-champion, #ffd166)'],
    [0.54, 'var(--score-elite, #e3b341)'],
    [0.50, 'var(--score-reference, #2dd4bf)'],
    [0.45, 'var(--score-contender, #58a6ff)'],
  ]);
}

export function probeAucColor(value) {
  return boundedMetricColor(value, [
    [0.70, 'var(--score-champion, #ffd166)'],
    [0.45, 'var(--score-elite, #e3b341)'],
    [0.20, 'var(--score-reference, #2dd4bf)'],
    [0.05, 'var(--score-contender, #58a6ff)'],
  ]);
}
