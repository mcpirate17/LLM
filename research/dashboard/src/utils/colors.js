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
