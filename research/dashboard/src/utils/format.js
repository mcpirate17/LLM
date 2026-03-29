/**
 * Shared formatting utilities used across dashboard components.
 * Import these instead of defining local copies.
 */

export function formatTime(timestamp) {
  if (!timestamp) return '--';
  return new Date(timestamp * 1000).toLocaleString();
}

export function formatDuration(seconds) {
  if (!seconds) return '--';
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

/**
 * Map a v7 composite score to a color (565pt max, reference avg ~150).
 *   150+ = gold   (reference-tier performance)
 *   100-150 = green (strong candidate)
 *   60-100 = blue  (promising, above median)
 *   30-60 = grey   (below median)
 *   <30 = red      (didn't learn / marginal)
 */
export function scoreColor(score) {
  if (score >= 150) return 'var(--accent-yellow, #d29922)';
  if (score >= 100) return 'var(--accent-green)';
  if (score >= 60) return 'var(--accent-blue)';
  if (score >= 30) return 'var(--text-muted)';
  return 'var(--accent-red)';
}

export function fmtNumber(value, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function fmtPct(value, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return `${(Number(value) * 100).toFixed(digits)}%`;
}
