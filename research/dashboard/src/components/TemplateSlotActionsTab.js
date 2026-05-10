import React, { useMemo } from 'react';
import { fmtPct } from '../utils/format';

function rateTone(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 'var(--text-muted)';
  if (num >= 0.25) return 'var(--score-champion, var(--accent-green))';
  if (num >= 0.15) return 'var(--score-reference, var(--accent-blue))';
  if (num > 0) return 'var(--accent-yellow)';
  return 'var(--text-muted)';
}

export default function TemplateSlotActionsTab({ recommendations, templates }) {
  const actionRows = useMemo(() => (
    (templates || [])
      .filter((row) => Array.isArray(row.actions) && row.actions.length > 0)
      .sort((a, b) => {
        const aRate = Number(a.s1_rate ?? 1);
        const bRate = Number(b.s1_rate ?? 1);
        if (aRate !== bRate) return aRate - bRate;
        return Number(b.n_used || 0) - Number(a.n_used || 0);
      })
      .slice(0, 24)
  ), [templates]);

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
          Recommended Fixes
        </div>
        {recommendations.length > 0 ? (
          <div style={{ display: 'grid', gap: 8 }}>
            {recommendations.map((item, idx) => (
              <div key={idx} style={{ padding: '9px 10px', background: 'var(--bg-tertiary)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 12, color: 'var(--text-primary)', lineHeight: 1.5 }}>
                {item}
              </div>
            ))}
          </div>
        ) : (
          <div className="ux-state ux-state-empty">No recommendations yet.</div>
        )}
      </div>

      <div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
          Template Actions
        </div>
        {actionRows.length > 0 ? (
          <div style={{ display: 'grid', gap: 10 }}>
            {actionRows.map((row) => (
              <div key={row.name} style={{ padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap', marginBottom: 4 }}>
                  <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'monospace' }}>{row.name}</span>
                  <span style={{ fontSize: 11, color: rateTone(row.s1_rate), fontWeight: 700 }}>S1 {fmtPct(row.s1_rate, 1)}</span>
                </div>
                {Array.isArray(row.diagnosis) && row.diagnosis.length > 0 && (
                  <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: 3 }}>
                    Why: {row.diagnosis.join(' ')}
                  </div>
                )}
                <div style={{ fontSize: 11, color: 'var(--accent-blue)', lineHeight: 1.5 }}>
                  Change: {row.actions.join(' ')}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="ux-state ux-state-empty">No template-specific actions yet.</div>
        )}
      </div>
    </div>
  );
}
