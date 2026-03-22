import React from 'react';

export function HypothesisLineage({ chain }) {
  if (!chain || chain.length === 0) return <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No linked hypothesis</div>;
  const statusColors = {
    confirmed: 'var(--accent-green)',
    refuted: 'var(--accent-red)',
    inconclusive: 'var(--accent-yellow)',
    pending: 'var(--text-muted)',
    testing: 'var(--accent-blue)',
  };
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {chain.map((h, i) => (
        <div key={h.hypothesis_id || i} style={{
          fontSize: 12, padding: '4px 8px',
          borderLeft: `3px solid ${statusColors[h.status] || 'var(--border)'}`,
          color: 'var(--text-secondary)',
        }}>
          <span style={{ fontWeight: 600, color: statusColors[h.status] || 'var(--text-muted)', textTransform: 'uppercase', fontSize: 10, marginRight: 6 }}>
            [{h.status}]
          </span>
          {h.prediction || h.title || 'Untitled hypothesis'}
        </div>
      ))}
    </div>
  );
}

export default React.memo(HypothesisLineage);
