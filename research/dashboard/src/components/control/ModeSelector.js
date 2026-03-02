import React from 'react';

const MODES = [
  { id: 'single', label: 'Single', desc: 'Run one set of architectures once.' },
  { id: 'continuous', label: 'Continuous', desc: 'Run back-to-back experiments, evolving strategy.' },
  { id: 'evolve', label: 'Evolution', desc: 'Population-based GA search for best layers.' },
  { id: 'novelty', label: 'Novelty Search', desc: 'Search for maximum structural diversity.' },
  { id: 'scale_up', label: 'Scale Up', desc: 'Train best survivors for more steps/tokens.' },
  { id: 'investigation', label: 'Investigation', desc: 'Deeper analysis of screening survivors.' },
  { id: 'validation', label: 'Validation', desc: 'Final multi-seed verification of candidates.' },
];

export function ModeSelector({ selectedMode, onModeChange, disabled }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 8 }}>
        Search Mode
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8 }}>
        {MODES.map((m) => (
          <button
            key={m.id}
            disabled={disabled}
            onClick={() => onModeChange(m.id)}
            style={{
              padding: '8px 10px',
              borderRadius: 6,
              background: selectedMode === m.id ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
              border: '1px solid var(--border)',
              color: selectedMode === m.id ? '#fff' : 'var(--text-secondary)',
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.15s ease',
            }}
            title={m.desc}
          >
            <div style={{ fontSize: 12, fontWeight: 600 }}>{m.label}</div>
            <div style={{ fontSize: 9, opacity: 0.8, marginTop: 2 }}>{m.id}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

export default ModeSelector;
