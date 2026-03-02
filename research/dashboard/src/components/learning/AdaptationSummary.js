import React, { useMemo } from 'react';

export function AdaptationSummary({ log }) {
  const summary = useMemo(() => {
    if (!log || log.length === 0) return null;
    let improved = 0, neutral = 0, regressed = 0;
    for (const entry of log) {
      const desc = (entry.description || '').toLowerCase();
      if (desc.includes('improved') || desc.includes('better') || desc.includes('positive')) {
        improved++;
      } else if (desc.includes('regressed') || desc.includes('worse') || desc.includes('negative') || desc.includes('declined')) {
        regressed++;
      } else {
        neutral++;
      }
    }
    return { total: log.length, improved, neutral, regressed };
  }, [log]);

  if (!summary) return null;

  return (
    <div className="card">
      <div className="card-title">Adaptation Outcomes</div>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 13, color: 'var(--text-secondary)', marginBottom: 8 }}>
        <span><strong>{summary.total}</strong> grammar adaptations</span>
        <span style={{ color: 'var(--accent-green)' }}>{summary.improved} improved</span>
        <span style={{ color: 'var(--text-muted)' }}>{summary.neutral} neutral</span>
        <span style={{ color: 'var(--accent-red)' }}>{summary.regressed} regressed</span>
      </div>
      {summary.total > 0 && (
        <div style={{
          height: 8, borderRadius: 4, display: 'flex', overflow: 'hidden',
          background: 'var(--bg-tertiary)',
        }}>
          {summary.improved > 0 && (
            <div style={{ width: `${(summary.improved / summary.total) * 100}%`, background: 'var(--accent-green)', height: '100%' }} />
          )}
          {summary.neutral > 0 && (
            <div style={{ width: `${(summary.neutral / summary.total) * 100}%`, background: 'var(--text-muted)', opacity: 0.4, height: '100%' }} />
          )}
          {summary.regressed > 0 && (
            <div style={{ width: `${(summary.regressed / summary.total) * 100}%`, background: 'var(--accent-red)', height: '100%' }} />
          )}
        </div>
      )}
    </div>
  );
}

export default AdaptationSummary;
