import React from 'react';

export function FailureContext({ context }) {
  if (!context) return null;
  return (
    <div style={{
      marginTop: 8,
      padding: 10,
      background: 'rgba(248, 81, 73, 0.05)',
      borderRadius: 6,
      border: '1px solid rgba(248, 81, 73, 0.2)',
      fontSize: 12,
      color: 'var(--text-secondary)',
      fontFamily: 'monospace',
      whiteSpace: 'pre-wrap',
      lineHeight: 1.4,
    }}>
      {context}
    </div>
  );
}

export default FailureContext;
