import React from 'react';

export function RefinementLineage({ program, onViewInLeaderboard }) {
  const lineage = Array.isArray(program?.lineage_chain) ? program.lineage_chain : [];
  if (lineage.length === 0) return null;

  const short = (value, n = 12) => {
    const s = String(value || '').trim();
    if (!s) return '--';
    return s.length > n ? s.slice(0, n) : s;
  };

  return (
    <div style={{
      padding: 12,
      background: 'var(--bg-tertiary)',
      borderRadius: 6,
      border: '1px solid var(--border)',
    }}>
      <div style={{
        fontSize: 12,
        color: 'var(--text-secondary)',
        fontWeight: 600,
        textTransform: 'uppercase',
        marginBottom: 8,
      }}>
        Refinement Lineage
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
        New refinements create new fingerprints. Lineage tracks each child back to its parent result so you can iteratively improve from a base fingerprint.
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {lineage.map((entry, idx) => (
          <div
            key={`${entry?.result_id || 'lineage'}-${idx}`}
            style={{
              display: 'grid',
              gridTemplateColumns: '36px 1fr auto',
              gap: 8,
              alignItems: 'center',
              padding: '6px 8px',
              borderRadius: 4,
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border)',
            }}
          >
            <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600 }}>
              L{idx}
            </span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <span style={{ fontFamily: 'monospace', fontSize: 11 }}>
                {short(entry?.graph_fingerprint, 20)}
              </span>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                result {short(entry?.result_id, 12)}
                {entry?.refinement?.intent ? ` · intent ${entry.refinement.intent}` : ''}
                {entry?.refinement?.analysis_driven && (
                  <span
                    style={{
                      marginLeft: 4, padding: '1px 5px', borderRadius: 3, fontSize: 9,
                      background: 'var(--accent-purple)', color: '#fff', fontWeight: 700,
                    }}
                    title={entry?.refinement?.analysis_recipe?.primary_target || 'Data-driven refinement'}
                  >
                    DATA-DRIVEN
                  </span>
                )}
              </span>
            </div>
            {entry?.result_id && onViewInLeaderboard && (
              <button
                className="refresh-btn"
                style={{ fontSize: 10, padding: '2px 8px' }}
                onClick={() => onViewInLeaderboard(entry.result_id)}
              >
                Open
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default RefinementLineage;
