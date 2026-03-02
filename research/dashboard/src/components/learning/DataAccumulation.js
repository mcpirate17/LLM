import React from 'react';

export function DataAccumulation({ title, current, threshold, children }) {
  if (current >= threshold) return children;
  const pct = threshold > 0 ? Math.min(100, Math.round((current / threshold) * 100)) : 0;
  return (
    <div className="card" style={{ position: 'relative', overflow: 'hidden' }}>
      <div className="card-title">{title}</div>
      <div style={{ padding: '16px 0 8px' }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
          Accumulating data — {current} of {threshold} samples needed for statistically
          meaningful results.
        </div>
        <div style={{
          height: 6, borderRadius: 3,
          background: 'var(--bg-tertiary)',
          overflow: 'hidden',
        }}>
          <div style={{
            height: '100%', borderRadius: 3,
            width: `${pct}%`,
            background: 'var(--accent-purple)',
            opacity: 0.6,
            transition: 'width 0.4s ease',
          }} />
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6, textAlign: 'right' }}>
          {pct}% ({current}/{threshold})
        </div>
      </div>
      {/* Skeleton rows */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, opacity: 0.25 }}>
        {[1, 2, 3].map(i => (
          <div key={i} style={{
            height: 14, borderRadius: 4,
            background: 'var(--bg-tertiary)',
            width: `${85 - i * 12}%`,
          }} />
        ))}
      </div>
    </div>
  );
}

export default DataAccumulation;
