import React, { useState } from 'react';
import { AVAILABLE_OPS_REFERENCE } from '../../utils/categoryConfig';

export function CategoryWeightsControl({ weights, onChange }) {
  const [showOpRef, setShowOpRef] = useState(false);

  if (!weights) return null;

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700, marginBottom: 12, display: 'flex', justifyContent: 'space-between' }}>
        <span>Category Probabilities</span>
        <span style={{ fontWeight: 'normal', textTransform: 'none', opacity: 0.6 }}>(1.0 = default)</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: '12px' }}>
        {Object.keys(weights).sort().map(cat => (
          <div key={cat} style={{ display: 'flex', flexDirection: 'column', gap: 4, background: 'rgba(255,255,255,0.03)', padding: '8px 10px', borderRadius: 6, border: '1px solid var(--border)' }}>
            <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'capitalize', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={cat.replace(/_/g, ' ')}>
              {cat.replace(/_/g, ' ')}
            </label>
            <input
              type="number" min="0.1" max="10" step="0.5"
              style={{ width: '100%', background: 'var(--bg-primary)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11, padding: '4px 6px' }}
              value={weights[cat] ?? 1.0}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 1.0;
                onChange(cat, val);
              }}
            />
          </div>
        ))}
      </div>

      <button
        style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 11, padding: '12px 0 4px', textAlign: 'left', display: 'flex', alignItems: 'center', gap: 4 }}
        onClick={() => setShowOpRef(!showOpRef)}
      >
        <span>{showOpRef ? '\u25BC' : '\u25B6'}</span>
        <span>Available Ops Reference ({Object.keys(weights).length} categories)</span>
      </button>
      
      {showOpRef && (
        <div style={{ fontSize: 10, color: 'var(--text-muted)', background: 'var(--bg-secondary)', borderRadius: 6, border: '1px solid var(--border)', padding: 10, maxHeight: 200, overflowY: 'auto', lineHeight: 1.6, marginTop: 4 }}>
          {Object.entries(AVAILABLE_OPS_REFERENCE).map(([cat, ops]) => (
            <div key={cat} style={{ marginBottom: 4 }}>
              <b style={{ color: 'var(--text-secondary)' }}>{cat}:</b> {ops}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default CategoryWeightsControl;
