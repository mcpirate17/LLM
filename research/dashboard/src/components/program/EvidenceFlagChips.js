import React from 'react';

export function EvidenceFlagChips({ flags }) {
  if (!flags) return null;
  const entries = [
    { key: 'has_baseline', label: 'Baseline' },
    { key: 'has_cka_artifact', label: 'CKA Artifact' },
    { key: 'has_multi_seed', label: 'Multi-Seed' },
    { key: 'has_hypothesis', label: 'Hypothesis' },
  ];
  return (
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
      {entries.map(({ key, label }) => {
        const ok = flags[key];
        return (
          <span key={key} style={{
            fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 4,
            color: ok ? 'var(--accent-green)' : 'var(--accent-red)',
            background: ok ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
            border: `1px solid ${ok ? 'var(--accent-green)' : 'var(--accent-red)'}44`,
          }}>
            {ok ? '\u2713' : '\u2717'} {label}
          </span>
        );
      })}
    </div>
  );
}

export default EvidenceFlagChips;
