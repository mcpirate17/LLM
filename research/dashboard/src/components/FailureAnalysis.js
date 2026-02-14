import React, { useState, useEffect } from 'react';

const API_BASE = process.env.REACT_APP_API_URL || '';

/**
 * FailureAnalysis — Funnel chart, error distribution, stage-at-death histogram.
 */

function FunnelBar({ label, value, total, color }) {
  const pct = total > 0 ? (value / total * 100) : 0;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
        <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
        <span style={{ color: 'var(--text-primary)' }}>{value} ({pct.toFixed(0)}%)</span>
      </div>
      <div style={{ height: 18, background: 'var(--bg-tertiary)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          width: `${pct}%`,
          height: '100%',
          background: color,
          borderRadius: 3,
          transition: 'width 0.3s ease',
          minWidth: pct > 0 ? 2 : 0,
        }} />
      </div>
    </div>
  );
}

function FailureAnalysis({ experimentId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!experimentId) return;
    setLoading(true);
    fetch(`${API_BASE}/api/experiments/${experimentId}/failures`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [experimentId]);

  if (loading) return <div className="card"><p style={{ color: 'var(--text-muted)' }}>Loading failure analysis...</p></div>;
  if (!data || data.total === 0) return <div className="card"><p style={{ color: 'var(--text-muted)' }}>No data</p></div>;

  const funnel = data.funnel || {};
  const errors = data.errors || {};
  const deaths = data.stage_deaths || {};

  return (
    <div className="card">
      <div className="card-title">Failure Analysis</div>

      {/* Stage Funnel */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
          Stage Funnel
        </div>
        <FunnelBar label="Generated" value={funnel.generated || 0} total={funnel.generated || 1} color="var(--accent-blue)" />
        <FunnelBar label="Stage 0 (Compilation)" value={funnel.stage0_passed || 0} total={funnel.generated || 1} color="var(--accent-green)" />
        <FunnelBar label="Stage 0.5 (Stability)" value={funnel.stage05_passed || 0} total={funnel.generated || 1} color="var(--accent-yellow)" />
        <FunnelBar label="Stage 1 (Learning)" value={funnel.stage1_passed || 0} total={funnel.generated || 1} color="var(--accent-purple)" />
      </div>

      {/* Stage-at-death */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
          Stage at Death
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          {Object.entries(deaths).map(([stage, count]) => (
            <div key={stage} style={{
              flex: 1,
              textAlign: 'center',
              padding: '8px 4px',
              background: 'var(--bg-tertiary)',
              borderRadius: 4,
            }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-red)' }}>{count}</div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>{stage}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Error Distribution */}
      {Object.keys(errors).length > 0 && (
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
            Top Errors
          </div>
          {Object.entries(errors).slice(0, 5).map(([err, count], i) => (
            <div key={i} style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '4px 0',
              borderBottom: '1px solid var(--border)',
              fontSize: 12,
            }}>
              <span style={{ color: 'var(--text-secondary)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginRight: 8 }}>
                {err}
              </span>
              <span style={{ color: 'var(--accent-red)', fontWeight: 600, fontFamily: 'monospace' }}>
                {count}x
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default FailureAnalysis;
