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
 * Map a v6 score to a color (GPT-2 = 100 anchor, open-ended).
 *   105+ = gold  (clearly beats GPT-2)
 *   100-105 = green (beats or matches GPT-2)
 *   90-99 = white  (competitive, below GPT-2)
 *   <90 = grey   (below competitive threshold)
 *   <20 = red    (didn't learn)
 */
export function scoreColor(score) {
  if (score >= 105) return 'var(--accent-yellow, #d29922)';
  if (score >= 100) return 'var(--accent-green)';
  if (score >= 90) return 'var(--text-primary)';
  if (score >= 20) return 'var(--text-muted)';
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
