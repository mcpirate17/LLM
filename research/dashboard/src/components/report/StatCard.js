import React from 'react';

export default function StatCard({ label, value, color }) {
  return (
    <div style={{
      padding: '12px 16px', background: 'var(--bg-tertiary)', borderRadius: 6,
      borderLeft: `3px solid ${color || 'var(--accent-blue)'}`,
    }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || 'var(--text-primary)' }}>{value}</div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>{label}</div>
    </div>
  );
}
