import React from 'react';

export function MetricRow({ label, value, tone, title }) {
  if (value === null || value === undefined || value === '--') return null;
  const toneColor = tone === 'positive'
    ? 'var(--accent-green)'
    : tone === 'negative'
      ? 'var(--accent-red)'
      : tone === 'neutral'
        ? 'var(--accent-yellow)'
        : 'var(--text-primary)';
  return (
    <div
      title={title}
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        gap: 10,
        padding: '3px 0',
        borderBottom: '1px solid var(--border)',
      }}
    >
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{ color: toneColor, textAlign: 'right' }}>{value}</span>
    </div>
  );
}

export default React.memo(MetricRow);
