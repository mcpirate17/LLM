import React from 'react';

export function TierBadge({ tier, entry }) {
  if (!tier && !entry?.tier) return null;
  const t = (tier || entry?.tier).toLowerCase();
  
  const TIER_COLORS = {
    screening: '#7a8591',
    investigation: '#58a6ff',
    validation: '#bc8cff',
    breakthrough: '#3fb950',
    champion: '#f0883e',
  };

  const label = t.charAt(0).toUpperCase() + t.slice(1);
  
  return (
    <span className={`badge tier-${t}`} style={{
      background: TIER_COLORS[t] || 'var(--border)',
      color: '#fff',
      padding: '2px 8px',
      borderRadius: 4,
      fontSize: 10,
      fontWeight: 700,
      textTransform: 'uppercase'
    }}>
      {label}
    </span>
  );
}

export default TierBadge;
