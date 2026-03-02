import React from 'react';

export function Section({ title, id, isOpen, onToggle, children }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div
        onClick={() => onToggle(id)}
        style={{
          padding: '10px 16px',
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border)',
          borderRadius: isOpen ? '6px 6px 0 0' : '6px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          cursor: 'pointer',
          userSelect: 'none'
        }}
      >
        <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{title}</h3>
        <span style={{ fontSize: 14, color: 'var(--text-muted)' }}>{isOpen ? '\u25be' : '\u25b8'}</span>
      </div>
      {isOpen && (
        <div style={{
          padding: '16px 0',
          display: 'flex',
          flexDirection: 'column',
          gap: 16
        }}>
          {children}
        </div>
      )}
    </div>
  );
}

export default Section;
