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
 * Map a 0-100 score to a color.
 * Used consistently across all scored tables.
 *   70+ = green (strong)
 *   40-69 = yellow (moderate)
 *   20-39 = orange (weak)
 *   <20 = red (poor)
 */
export function scoreColor(score) {
  if (score >= 70) return 'var(--accent-green)';
  if (score >= 40) return 'var(--accent-yellow)';
  if (score >= 20) return 'var(--accent-orange, #f0883e)';
  return 'var(--accent-red)';
}
