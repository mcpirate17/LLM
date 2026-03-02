import React, { useState } from 'react';

export function LearningLog({ log }) {
  const [showRaw, setShowRaw] = useState(false);

  if (!log || log.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Learning Log</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>No learning events yet.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="card-title" style={{ margin: 0 }}>Learning Log ({log.length})</div>
        <button
          className="refresh-btn"
          style={{ fontSize: 11, padding: '2px 8px' }}
          onClick={() => setShowRaw(!showRaw)}
        >
          {showRaw ? 'Hide raw entries' : 'Show raw entries'}
        </button>
      </div>
      {showRaw && (
        <div style={{ maxHeight: 300, overflow: 'auto', marginTop: 8 }}>
          {log.map((entry, i) => (
            <div key={entry.id || i} style={{
              padding: '8px 12px',
              borderLeft: '3px solid var(--accent-purple)',
              marginBottom: 8,
              background: 'var(--bg-tertiary)',
              borderRadius: '0 4px 4px 0',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-purple)', textTransform: 'uppercase' }}>
                  {entry.event_type?.replace(/_/g, ' ')}
                </span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  {entry.timestamp ? new Date(entry.timestamp * 1000).toLocaleString() : ''}
                </span>
              </div>
              <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                {entry.description}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default LearningLog;
