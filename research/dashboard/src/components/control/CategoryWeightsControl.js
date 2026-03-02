import React, { useState } from 'react';
import { AVAILABLE_OPS_REFERENCE } from '../../utils/categoryConfig';

export function CategoryWeightsControl({ weights, onChange }) {
  const [showOpRef, setShowOpRef] = useState(false);

  if (!weights) return null;

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 8, borderTop: '1px solid var(--border)', paddingTop: 12 }}>
        Category Weights <span style={{ fontWeight: 'normal', textTransform: 'none' }}>(1.0 = default, higher = more likely)</span>
      </div>
      <div className="config-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))' }}>
        {Object.keys(weights).map(cat => (
          <div className="config-item" key={cat}>
            <label style={{ fontSize: 11 }}>{cat.replace(/_/g, ' ')}</label>
            <input
              type="number" min="0.1" max="10" step="0.5"
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
