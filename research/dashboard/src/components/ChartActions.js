import React from 'react';

/**
 * ChartActions — thin bar below a chart showing reset-zoom, insight, and action pills.
 *
 * Props:
 *   isZoomed     — show "Reset Zoom" button
 *   onResetZoom  — callback for reset
 *   insight      — optional 1-line string (e.g. "Yield declining")
 *   insightColor — optional color for the insight text
 *   actions      — array of { id, label, detail, color, onClick } (max 3)
 */
export default function ChartActions({ isZoomed, onResetZoom, insight, insightColor, actions = [] }) {
  const visibleActions = actions.slice(0, 3);
  const hasContent = isZoomed || insight || visibleActions.length > 0;
  if (!hasContent) return null;

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      marginTop: 8,
      flexWrap: 'wrap',
      fontSize: 11,
    }}>
      {isZoomed && (
        <button
          className="refresh-btn"
          style={{ fontSize: 10, padding: '2px 8px' }}
          onClick={onResetZoom}
        >
          Reset Zoom
        </button>
      )}
      {insight && (
        <span style={{ color: insightColor || 'var(--text-muted)', fontStyle: 'italic', fontSize: 10 }}>
          {insight}
        </span>
      )}
      {visibleActions.map(action => (
        <button
          key={action.id}
          onClick={action.onClick}
          title={action.detail || ''}
          style={{
            fontSize: 10,
            padding: '2px 10px',
            borderRadius: 12,
            border: `1px solid ${action.color || 'var(--border)'}`,
            background: 'transparent',
            color: action.color || 'var(--text-secondary)',
            cursor: 'pointer',
            whiteSpace: 'nowrap',
          }}
        >
          {action.label}
        </button>
      ))}
    </div>
  );
}
