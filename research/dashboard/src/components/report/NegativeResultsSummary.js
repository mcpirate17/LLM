import React, { useState, useEffect } from 'react';
import { apiCall } from '../../services/apiService';

const API_BASE = process.env.REACT_APP_API_URL || '';

export default function NegativeResultsSummary() {
  const [data, setData] = useState(null);

  useEffect(() => {
    apiCall(`/api/analytics/negative-results`)
      .then(r => r.ok ? r.json() : null)
      .then(d => setData(d))
      .catch(() => {});
  }, []);

  if (!data) return null;
  const hasContent = (data.failed_ops?.length > 0) || (data.toxic_bigrams?.length > 0) || (data.refuted_hypotheses?.length > 0);
  if (!hasContent) return null;

  return (
    <div className="card">
      <div className="card-title">Do Not Pursue</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Aggregated negative results: operations, patterns, and hypotheses that have been repeatedly tested and failed.
        Use this as a blacklist when designing future experiments.
      </p>
      <p style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.5 }}>
        {data.summary}
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {data.failed_ops?.length > 0 && (
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
              Zero-Success Ops
            </div>
            {data.failed_ops.slice(0, 8).map(op => (
              <div key={op.op_name} style={{
                display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                borderBottom: '1px solid var(--border)',
              }}>
                <span style={{ fontSize: 12, fontFamily: 'monospace', color: 'var(--accent-red)' }}>{op.op_name}</span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>0/{op.n_used} S1</span>
              </div>
            ))}
          </div>
        )}
        {data.toxic_bigrams?.length > 0 && (
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
              Toxic Patterns
            </div>
            {data.toxic_bigrams.slice(0, 8).map(tb => (
              <div key={tb.pattern} style={{
                display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                borderBottom: '1px solid var(--border)',
              }}>
                <span style={{ fontSize: 12, fontFamily: 'monospace', color: 'var(--accent-red)' }}>{tb.pattern}</span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>pen {tb.penalty.toFixed(2)}</span>
              </div>
            ))}
          </div>
        )}
        {data.refuted_hypotheses?.length > 0 && (
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
              Refuted Hypotheses
            </div>
            {data.refuted_hypotheses.slice(0, 5).map((h, i) => (
              <div key={i} style={{
                padding: '4px 6px', borderLeft: '2px solid var(--accent-red)',
                marginBottom: 4, fontSize: 11, color: 'var(--text-secondary)',
              }}>
                {(h.content || '').slice(0, 120)}{(h.content || '').length > 120 ? '...' : ''}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
