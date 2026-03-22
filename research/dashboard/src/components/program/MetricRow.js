import React from 'react';

export function MetricRow({ label, value }) {
  if (value === null || value === undefined || value === '--') return null;
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: '1px solid var(--border)' }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <span>{value}</span>
    </div>
  );
}

export default React.memo(MetricRow);
