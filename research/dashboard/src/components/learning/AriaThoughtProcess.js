import React from 'react';
import { useNarrative } from '../../hooks/useNarrative';

export function AriaThoughtProcess() {
  const ctx = useNarrative();
  if (!ctx?.narrative) return null;

  const { narrative, trend } = ctx;

  const trendColor = trend === 'improving'
    ? 'var(--accent-green)'
    : trend === 'declining'
      ? 'var(--accent-red, #e74c3c)'
      : 'var(--accent-purple)';

  return (
    <div className="card" style={{
      padding: '14px 16px',
      borderLeft: `3px solid ${trendColor}`,
      background: 'var(--bg-secondary)',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8,
      }}>
        <span style={{
          fontSize: 11, fontWeight: 700, textTransform: 'uppercase',
          letterSpacing: 0.5, color: trendColor,
        }}>
          Aria's Thought Process
        </span>
        <span style={{
          fontSize: 9, fontWeight: 600,
          color: trendColor,
          background: `color-mix(in srgb, ${trendColor} 12%, transparent)`,
          border: `1px solid ${trendColor}`,
          borderRadius: 4,
          padding: '1px 5px',
        }}>
          Live
        </span>
      </div>
      <div style={{
        fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65,
      }}>
        {narrative}
      </div>
    </div>
  );
}

export default AriaThoughtProcess;
