import React, { useState } from 'react';

function formatComparison(ds) {
  if (ds.comparison === 'context' || ds.comparison === 'status' || ds.comparison === 'nominal') {
    return `${ds.metric} = ${ds.value}`;
  }
  if (ds.comparison === 'ratio') {
    return `${ds.metric}: ${ds.value}`;
  }
  if (ds.threshold != null) {
    return `${ds.metric} ${ds.comparison} ${ds.threshold} (actual: ${ds.value})`;
  }
  return `${ds.metric}: ${ds.value}`;
}

export default function DataSourceBadge({ dataSources, onNavigateEvidence }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const hasSources = Array.isArray(dataSources) && dataSources.length > 0;

  return (
    <span
      style={{ position: 'relative', display: 'inline-flex' }}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <span style={{
        fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
        letterSpacing: 0.3,
        color: 'var(--accent-green)',
        background: 'rgba(63, 185, 80, 0.16)',
        border: '1px solid var(--accent-green)',
        borderRadius: 4,
        padding: '1px 6px',
        cursor: hasSources ? 'help' : 'default',
      }}>
        Recommended Action
      </span>
      {showTooltip && hasSources && (
        <div style={{
          position: 'absolute',
          top: '100%',
          left: 0,
          marginTop: 6,
          minWidth: 280,
          maxWidth: 380,
          padding: '10px 12px',
          background: 'var(--bg-secondary, #1c2128)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          zIndex: 100,
          fontSize: 11,
          lineHeight: 1.6,
        }}>
          <div style={{
            fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
            color: 'var(--accent-green)', marginBottom: 6,
            letterSpacing: 0.5,
          }}>
            Data Sources
          </div>
          {dataSources.map((ds, i) => (
            <div
              key={i}
              style={{
                display: 'flex', alignItems: 'baseline', gap: 6,
                padding: '2px 0',
                color: 'var(--text-secondary)',
              }}
            >
              <span style={{
                color: ds.comparison === '<' || ds.comparison === '>'
                  ? 'var(--accent-yellow)' : 'var(--text-muted)',
                flexShrink: 0,
              }}>
                {ds.comparison === '<' || ds.comparison === '>' ? '\u26A0' : '\u2022'}
              </span>
              <span style={{ flex: 1 }}>
                {formatComparison(ds)}
              </span>
              {ds.tab && onNavigateEvidence && (
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); onNavigateEvidence(ds.tab); }}
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    fontSize: 10, color: 'var(--accent-blue)',
                    padding: 0, textDecoration: 'underline', flexShrink: 0,
                  }}
                >
                  {ds.tab}
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </span>
  );
}
