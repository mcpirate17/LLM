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

export const SCORE_MAX = 855;

/**
 * Canonical v10 score color ramp.
 *
 * Colors are anchored to the scoring rubric ceiling, not to current references
 * or the live leaderboard distribution. v10 can award 855 points before
 * penalties: 660 legacy/base points + 175 capability points + 20 aux trajectory
 * points. Bands are fixed percentages of that ceiling.
 */
export const SCORE_STOPS = [
  { min: SCORE_MAX * 0.90, color: 'var(--score-apex, #ff7b72)', start: '#ffd166', end: '#ff7b72', label: 'Near Ceiling' },
  { min: SCORE_MAX * 0.75, color: 'var(--score-champion, #ffd166)', start: '#e3b341', end: '#ffd166', label: 'Exceptional' },
  { min: SCORE_MAX * 0.60, color: 'var(--score-elite, #e3b341)', start: '#2dd4bf', end: '#e3b341', label: 'Strong' },
  { min: SCORE_MAX * 0.45, color: 'var(--score-reference, #2dd4bf)', start: '#58a6ff', end: '#2dd4bf', label: 'Competitive' },
  { min: SCORE_MAX * 0.30, color: 'var(--score-contender, #58a6ff)', start: '#3b82f6', end: '#58a6ff', label: 'Developing' },
  { min: SCORE_MAX * 0.15, color: 'var(--score-scout, #8b949e)', start: '#6e7681', end: '#8b949e', label: 'Early Signal' },
  { min: -Infinity, color: 'var(--text-muted)', start: '#484f58', end: '#6e7681', label: 'Exploratory' },
];

export function scoreStop(score) {
  const value = Number(score);
  if (!Number.isFinite(value)) return SCORE_STOPS[SCORE_STOPS.length - 1];
  return SCORE_STOPS.find((stop) => value >= stop.min) || SCORE_STOPS[SCORE_STOPS.length - 1];
}

export function scoreColor(score) {
  return scoreStop(score).color;
}

export function scoreGradient(score) {
  const stop = scoreStop(score);
  return `linear-gradient(90deg, ${stop.start}, ${stop.end})`;
}

export function scoreGradientStops(score) {
  const stop = scoreStop(score);
  return [stop.start, stop.end];
}

export function scoreToneLabel(score) {
  return scoreStop(score).label;
}

export function scoreScaleDomain(scores, { minMode = 'average', padding = 0.04 } = {}) {
  const values = (scores || [])
    .map((score) => Number(score))
    .filter((score) => Number.isFinite(score));
  if (values.length === 0) {
    return { min: 0, max: SCORE_MAX };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const max = sorted[sorted.length - 1];
  const average = values.reduce((acc, score) => acc + score, 0) / values.length;
  const p25 = sorted[Math.floor((sorted.length - 1) * 0.25)];
  const rawMin = minMode === 'p25' ? p25 : average;
  const span = Math.max(1, max - rawMin);
  return {
    min: Math.max(0, rawMin - span * padding),
    max: Math.min(SCORE_MAX, max + span * padding),
  };
}

export function scoreScalePercent(score, domain, minPercent = 4) {
  const value = Number(score);
  if (!Number.isFinite(value)) return 0;
  const min = Number(domain?.min);
  const max = Number(domain?.max);
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    return Math.max(minPercent, Math.min(100, (value / SCORE_MAX) * 100));
  }
  const percent = ((value - min) / (max - min)) * 100;
  return Math.max(minPercent, Math.min(100, percent));
}

export function scoreScaleRatio(score, domain) {
  const value = Number(score);
  const min = Number(domain?.min);
  const max = Number(domain?.max);
  if (!Number.isFinite(value) || !Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    return 0;
  }
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

export function fmtNumber(value, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function fmtPct(value, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function fmtLoss(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return Number(value).toFixed(3);
}
