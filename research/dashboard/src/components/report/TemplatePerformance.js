import React, { useMemo } from 'react';

export default function TemplatePerformance({ hitRates }) {
  const sorted = useMemo(() => {
    if (!hitRates) return [];
    return Object.entries(hitRates)
      .map(([name, data]) => ({ name, ...data }))
      .sort((a, b) => (b.s1_rate || 0) - (a.s1_rate || 0));
  }, [hitRates]);

  if (sorted.length === 0) return null;

  const maxRate = Math.max(0.01, ...sorted.map(t => t.s1_rate || 0));

  return (
    <div className="card">
      <div className="card-title">Template Success Rates (Phase 5.2)</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Tracks the performance of architectural blocks. Higher S1 rates indicate 
        stable, high-quality architectural patterns.
      </p>
      
      <div style={{ display: 'grid', gap: 10 }}>
        {sorted.map((t) => {
          const width = Math.max(5, (t.s1_rate / maxRate) * 100);
          const color = t.s1_rate > 0.5 ? 'var(--accent-green)' : (t.s1_rate > 0.2 ? 'var(--accent-blue)' : 'var(--text-muted)');
          
          return (
            <div key={t.name} style={{ display: 'grid', gridTemplateColumns: '140px 1fr 80px', gap: 12, alignItems: 'center' }}>
              <span style={{ fontSize: 11, color: 'var(--text-secondary)', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={t.name}>
                {t.name.replace(/_template$/, '').replace(/apply_/, '')}
              </span>
              <div style={{ height: 12, background: 'var(--bg-tertiary)', border: '1px solid var(--border)', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{ width: `${width}%`, height: '100%', background: color, opacity: 0.8 }} />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                <span style={{ fontSize: 11, color: t.s1_rate > 0 ? 'var(--text-primary)' : 'var(--text-muted)', fontWeight: 600 }}>
                  {(t.s1_rate * 100).toFixed(1)}%
                </span>
                <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>
                  {t.n_s1}/{t.n_used} runs
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid var(--border)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>Best Loss Ratio</div>
          {sorted.filter(t => t.avg_loss_ratio != null).sort((a, b) => a.avg_loss_ratio - b.avg_loss_ratio).slice(0, 3).map(t => (
            <div key={t.name} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
              <span style={{ color: 'var(--text-secondary)' }}>{t.name.replace(/_template$/, '').replace(/apply_/, '')}</span>
              <span style={{ fontWeight: 600, color: 'var(--accent-green)' }}>{t.avg_loss_ratio.toFixed(4)}</span>
            </div>
          ))}
        </div>
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>Most Used Pattern</div>
          {sorted.sort((a, b) => b.n_used - a.n_used).slice(0, 3).map(t => (
            <div key={t.name} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
              <span style={{ color: 'var(--text-secondary)' }}>{t.name.replace(/_template$/, '').replace(/apply_/, '')}</span>
              <span style={{ color: 'var(--text-muted)' }}>{t.n_used}x</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
