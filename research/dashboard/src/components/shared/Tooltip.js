import React, { useState } from 'react';

export function Tooltip({ children, content }) {
  const [show, setShow] = useState(false);

  if (!content) return children;

  return (
    <div
      style={{ position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <div style={{
          position: 'absolute',
          bottom: '100%',
          left: '50%',
          transform: 'translateX(-50%)',
          marginBottom: 8,
          padding: '8px 12px',
          background: '#161b22',
          border: '1px solid var(--border)',
          borderRadius: 6,
          boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
          zIndex: 1000,
          minWidth: 200,
          whiteSpace: 'pre-wrap',
          fontSize: 11,
          fontWeight: 400,
          lineHeight: 1.4,
          color: 'var(--text-primary)',
          pointerEvents: 'none',
          textAlign: 'center',
        }}>
          {content}
          <div style={{
            position: 'absolute',
            top: '100%',
            left: '50%',
            marginLeft: -6,
            border: '6px solid transparent',
            borderTopColor: 'var(--border)'
          }} />
        </div>
      )}
    </div>
  );
}

export default Tooltip;
