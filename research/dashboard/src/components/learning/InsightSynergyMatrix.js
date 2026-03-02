import React from 'react';

export function InsightSynergyMatrix({ data }) {
  const synergistic = Array.isArray(data?.synergistic_pairs) ? data.synergistic_pairs : [];
  const antagonistic = Array.isArray(data?.antagonistic_pairs) ? data.antagonistic_pairs : [];
  const available = Boolean(data?.available) && (synergistic.length > 0 || antagonistic.length > 0);
  
  const trim = (text) => {
    const t = String(text || '').trim();
    if (t.length <= 88) return t;
    return `${t.slice(0, 85)}...`;
  };

  if (!available) {
    return (
      <div className="card">
        <h3 style={{ margin: 0, marginBottom: 8 }}>Insight Synergy Matrix</h3>
        <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: 12 }}>
          Not enough resolved insight-bundle trials yet.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <h3 style={{ margin: 0, marginBottom: 8 }}>Insight Synergy Matrix</h3>
      <p style={{ margin: '0 0 10px', color: 'var(--text-secondary)', fontSize: 12 }}>
        Learns which insight combinations improve downstream outcomes and which conflict.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent-green)', marginBottom: 6 }}>
            Positive Pairs
          </div>
          {synergistic.slice(0, 5).map((row, idx) => (
            <div key={`syn-${idx}`} style={{ fontSize: 11, marginBottom: 6, color: 'var(--text-secondary)' }}>
              <div>{trim(row?.insight_a_content)} + {trim(row?.insight_b_content)}</div>
              <div style={{ color: 'var(--text-muted)' }}>
                reward {Number(row?.mean_reward || 0).toFixed(3)} · trials {row?.n_trials || 0} · {row?.confidence_label || 'low'}
              </div>
            </div>
          ))}
        </div>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent-red)', marginBottom: 6 }}>
            Conflicting Pairs
          </div>
          {antagonistic.slice(0, 5).map((row, idx) => (
            <div key={`ant-${idx}`} style={{ fontSize: 11, marginBottom: 6, color: 'var(--text-secondary)' }}>
              <div>{trim(row?.insight_a_content)} + {trim(row?.insight_b_content)}</div>
              <div style={{ color: 'var(--text-muted)' }}>
                reward {Number(row?.mean_reward || 0).toFixed(3)} · trials {row?.n_trials || 0} · {row?.confidence_label || 'low'}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default InsightSynergyMatrix;
