export function formatNum(n) {
  if (n == null) return '-';
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return typeof n === 'number' ? n.toFixed(n % 1 ? 2 : 0) : String(n);
}

export function formatBenchmarkValue(value, unit) {
  if (value == null) return '-';
  if (unit === 'percent') return `${Number(value).toFixed(2)}%`;
  if (unit === 'x') return `${Number(value).toFixed(2)}x`;
  if (unit === 'ms') return `${formatNum(Number(value))}ms`;
  if (unit === 'score') return Number(value).toFixed(2);
  return formatNum(Number(value));
}
