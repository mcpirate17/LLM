import React, { useState, useEffect } from 'react';
import apiService from '../../services/apiService';

export function ReferenceComparison({ program, leaderboardEntry }) {
  const [references, setReferences] = useState([]);
  const [selectedRefId, setSelectedRefId] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    apiService.getReferences()
      .then(d => {
        setReferences(d.entries || []);
        // Auto-select a reference from same family if possible
        const sameFamily = (d.entries || []).find(r => r.architecture_family === (leaderboardEntry?.architecture_family || program?.architecture_family));
        if (sameFamily) setSelectedRefId(sameFamily.result_id);
        else if (d.entries?.length > 0) setSelectedRefId(d.entries[0].result_id);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [program.result_id]);

  const selectedRef = references.find(r => r.result_id === selectedRefId);
  if (references.length === 0 && !loading) return null;

  const metrics = [
    { key: '_loss_ratio_compare', label: 'Loss Ratio', higherIsBetter: false },
    { key: 'param_efficiency', label: 'Param Efficiency', higherIsBetter: true },
    { key: 'quant_int8_retention', label: 'Quant Retention', higherIsBetter: true },
    { key: 'robustness_long_ctx_score', label: 'Long-Context', higherIsBetter: true },
    { key: 'robustness_noise_score', label: 'Noise Score', higherIsBetter: false },
  ];

  const toFiniteNumber = (value) => {
    if (value === undefined || value === null) return null;
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  };

  const firstFinite = (...values) => {
    for (const value of values) {
      const num = toFiniteNumber(value);
      if (num !== null) return num;
    }
    return null;
  };

  const getCandidateValue = (key) => {
    if (key === '_loss_ratio_compare') {
      return firstFinite(
        leaderboardEntry?.validation_loss_ratio,
        program?.validation_loss_ratio,
        leaderboardEntry?.investigation_loss_ratio,
        leaderboardEntry?.screening_loss_ratio,
        program?.loss_ratio,
      );
    }
    return firstFinite(
      leaderboardEntry?.[key],
      program?.[key],
    );
  };

  const getReferenceValue = (key, ref) => {
    if (key === '_loss_ratio_compare') {
      return firstFinite(
        ref?.validation_loss_ratio,
        ref?.investigation_loss_ratio,
        ref?.screening_loss_ratio,
      );
    }
    return firstFinite(ref?.[key]);
  };

  return (
    <div className="card" style={{ padding: 12, minWidth: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, marginBottom: 12 }}>
        <div className="card-title" style={{ margin: 0 }}>Reference Comparison</div>
        <select 
          value={selectedRefId} 
          onChange={e => setSelectedRefId(e.target.value)}
          style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border)', borderRadius: 4, fontSize: 11, padding: '2px 4px', minWidth: 0, maxWidth: 180 }}
        >
          {references.map(r => (
            <option key={r.result_id} value={r.result_id}>
              {r.architecture_family} ({r.architecture_name || r.result_id.slice(0, 8)})
            </option>
          ))}
        </select>
      </div>

      {!selectedRef ? (
        <div style={{ textAlign: 'center', padding: 20, color: 'var(--text-muted)', fontSize: 12 }}>
          {loading ? 'Loading references...' : 'No reference selected'}
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(88px, 1fr))', gap: 8 }}>
          {metrics.map(m => {
            const candVal = getCandidateValue(m.key);
            const refVal = getReferenceValue(m.key, selectedRef);
            
            let diff = null;
            let better = null;
            if (candVal != null && refVal != null) {
              diff = candVal - refVal;
              better = m.higherIsBetter ? diff > 0 : diff < 0;
            }

            return (
              <div key={m.key} style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>{m.label}</div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>
                  {candVal != null ? (m.key.includes('retention') ? `${candVal.toFixed(1)}%` : candVal.toFixed(3)) : '--'}
                </div>
                <div style={{ fontSize: 10, color: refVal != null ? 'var(--text-muted)' : 'transparent' }}>
                  vs {refVal != null ? (m.key.includes('retention') ? `${refVal.toFixed(1)}%` : refVal.toFixed(3)) : '--'}
                </div>
                {diff != null && (
                  <div style={{ fontSize: 10, fontWeight: 700, color: better ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                    {diff > 0 ? '+' : ''}{m.key.includes('retention') ? `${diff.toFixed(1)}%` : diff.toFixed(3)}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default React.memo(ReferenceComparison);
