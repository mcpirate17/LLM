import React, { useState, useCallback } from 'react';
import { apiCall, postJson } from '../../services/apiService';
import { INTENTS } from './architectureUtils';

export function MorphPanel({ resultId, onSelectCandidate }) {
  const [intent, setIntent] = useState('balanced');
  const [candidates, setCandidates] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sourceOps, setSourceOps] = useState([]);

  const handleGenerate = useCallback(async () => {
    setLoading(true);
    setError(null);
    setCandidates(null);
    try {
      const res = await postJson(`/api/programs/${resultId}/morph`, { intent, n_candidates: 6, use_analysis: true });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
      setCandidates(data.candidates || []);
      setSourceOps(data.source_ops || []);
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setLoading(false);
    }
  }, [resultId, intent]);

  const intentColor = INTENTS.find(i => i.key === intent)?.color || 'var(--text-secondary)';

  return (
    <div style={{ padding: '10px 14px', borderTop: '1px solid var(--border)', maxHeight: 320, overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>Smart Morph</span>
        <div style={{ display: 'flex', gap: 4 }}>
          {INTENTS.map(i => (
            <button
              key={i.key}
              onClick={() => setIntent(i.key)}
              style={{
                fontSize: 10,
                padding: '2px 8px',
                borderRadius: 10,
                border: intent === i.key ? `1.5px solid ${i.color}` : '1px solid var(--border)',
                background: intent === i.key ? 'var(--bg-tertiary)' : 'none',
                color: intent === i.key ? i.color : 'var(--text-muted)',
                cursor: 'pointer',
                fontWeight: intent === i.key ? 600 : 400,
              }}
            >
              {i.label}
            </button>
          ))}
        </div>
        <button
          onClick={handleGenerate}
          disabled={loading}
          style={{
            marginLeft: 'auto',
            fontSize: 11,
            padding: '3px 12px',
            borderRadius: 4,
            border: `1px solid ${intentColor}`,
            background: 'none',
            color: intentColor,
            cursor: loading ? 'wait' : 'pointer',
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? 'Generating\u2026' : 'Generate'}
        </button>
      </div>

      {error && (
        <div style={{ fontSize: 11, color: 'var(--accent-red)', marginBottom: 6 }}>{error}</div>
      )}

      {candidates && candidates.length === 0 && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', padding: 12 }}>
          No valid mutations generated. Try a different intent.
        </div>
      )}

      {candidates && candidates.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {candidates.map((c, idx) => (
            <div
              key={c.fingerprint}
              onClick={() => onSelectCandidate && onSelectCandidate(c)}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '6px 10px',
                borderRadius: 6,
                border: '1px solid var(--border)',
                background: 'var(--bg-secondary)',
                cursor: 'pointer',
                transition: 'border-color 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.borderColor = intentColor}
              onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
            >
              <span style={{ fontSize: 14, fontWeight: 700, color: intentColor, minWidth: 22, textAlign: 'center' }}>
                #{idx + 1}
              </span>
              <div style={{ minWidth: 48, textAlign: 'center' }}>
                <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
                  {(c.score * 100).toFixed(0)}
                </div>
                <div style={{ fontSize: 8, color: 'var(--text-muted)', textTransform: 'uppercase' }}>score</div>
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-secondary)', minWidth: 70 }}>
                <div>{c.n_ops} ops, d={c.depth}</div>
                <div>{(c.params_estimate / 1000).toFixed(0)}K params</div>
              </div>
              <div style={{ flex: 1, display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                {c.added_ops.map(op => (
                  <span key={`+${op}`} style={{ fontSize: 9, padding: '1px 5px', borderRadius: 3, background: 'rgba(80,200,120,0.15)', color: 'var(--accent-green)' }}>+{op}</span>
                ))}
                {c.removed_ops.map(op => (
                  <span key={`-${op}`} style={{ fontSize: 9, padding: '1px 5px', borderRadius: 3, background: 'rgba(255,100,100,0.15)', color: 'var(--accent-red)' }}>-{op}</span>
                ))}
              </div>
              <div style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
                {Object.entries(c.score_breakdown || {}).map(([k, v]) => (
                  <div key={k} title={`${k}: ${(v * 100).toFixed(0)}%`} style={{ width: 4, height: Math.max(4, v * 28), background: intentColor, borderRadius: 2, opacity: 0.5 + v * 0.5 }} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default MorphPanel;
