import React from 'react';

export function OutcomesByPhase({ outcomes }) {
  if (!outcomes) return null;
  const phases = [
    { key: 'screening', label: 'Screening' },
    { key: 'investigation', label: 'Investigation' },
    { key: 'validation', label: 'Validation' },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {phases.map(({ key, label }) => {
        const data = outcomes[key];
        if (!data) return (
          <div key={key} style={{ fontSize: 12, color: 'var(--text-muted)', padding: '4px 0', borderBottom: '1px solid var(--border)' }}>
            {label}: --
          </div>
        );
        const passed = data.passed;
        return (
          <div key={key} style={{ fontSize: 12, padding: '4px 0', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
            <span style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              {data.discovery_loss_ratio != null && <span>D.LR: {Number(data.discovery_loss_ratio).toFixed(4)}</span>}
              {data.validation_loss_ratio != null && <span>V.LR: {Number(data.validation_loss_ratio).toFixed(4)}</span>}
              {data.loss_ratio != null && <span>LR: {Number(data.loss_ratio).toFixed(4)}</span>}
              {data.novelty != null && <span>Nov: {Number(data.novelty).toFixed(3)}</span>}
              {data.robustness != null && <span>Rob: {Number(data.robustness).toFixed(3)}</span>}
              {data.baseline_ratio != null && <span>BL: {Number(data.baseline_ratio).toFixed(3)}</span>}
              <span className={`badge ${passed ? 'badge-pass' : 'badge-fail'}`} style={{ minWidth: 40, textAlign: 'center' }}>
                {passed ? 'PASS' : 'FAIL'}
              </span>
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default OutcomesByPhase;
