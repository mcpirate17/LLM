import React, { useMemo } from 'react';

export default function MirroredOpsChart({ bestOps, worstOps, rows = 6 }) {
  const model = useMemo(() => {
    const safeBest = Array.isArray(bestOps) ? bestOps.slice(0, rows) : [];
    const safeWorst = Array.isArray(worstOps)
      ? [...worstOps].sort((a, b) => (b.total_count || 0) - (a.total_count || 0)).slice(0, rows)
      : [];
    const count = Math.max(safeBest.length, safeWorst.length);
    const maxBestRate = Math.max(1, ...safeBest.map((op) => (op.s1_rate || 0) * 100));
    const maxWorstUses = Math.max(1, ...safeWorst.map((op) => op.total_count || 0));
    return { safeBest, safeWorst, count, maxBestRate, maxWorstUses };
  }, [bestOps, worstOps, rows]);

  if (model.count === 0) {
    return null;
  }

  return (
    <div className="card">
      <div className="card-title">Op Signal Mirror</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Left: most frequent failing ops. Right: highest S1-rate ops.
      </p>
      <div style={{ display: 'grid', gap: 8 }}>
        {Array.from({ length: model.count }).map((_, idx) => {
          const fail = model.safeWorst[idx];
          const win = model.safeBest[idx];
          const failWidth = fail ? Math.max(8, ((fail.total_count || 0) / model.maxWorstUses) * 100) : 0;
          const winWidth = win ? Math.max(8, (((win.s1_rate || 0) * 100) / model.maxBestRate) * 100) : 0;
          return (
            <div key={`mirror-row-${idx}`} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, alignItems: 'center' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '110px 1fr 46px', gap: 8, alignItems: 'center' }}>
                <span style={{ fontFamily: 'monospace', fontSize: 11, color: 'var(--text-secondary)', textAlign: 'right', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {fail?.op_name || '--'}
                </span>
                <div style={{ height: 10, background: 'var(--bg-tertiary)', border: '1px solid var(--border)', borderRadius: 999, display: 'flex', justifyContent: 'flex-end', overflow: 'hidden' }}>
                  <div style={{ width: `${failWidth}%`, background: 'var(--accent-red)', opacity: 0.75 }} />
                </div>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{fail ? `${fail.total_count || 0}x` : '--'}</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '46px 1fr 110px', gap: 8, alignItems: 'center' }}>
                <span style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'right' }}>
                  {win ? `${((win.s1_rate || 0) * 100).toFixed(1)}%` : '--'}
                </span>
                <div style={{ height: 10, background: 'var(--bg-tertiary)', border: '1px solid var(--border)', borderRadius: 999, overflow: 'hidden' }}>
                  <div style={{ width: `${winWidth}%`, background: 'var(--accent-green)', opacity: 0.75 }} />
                </div>
                <span style={{ fontFamily: 'monospace', fontSize: 11, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {win?.op_name || '--'}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
