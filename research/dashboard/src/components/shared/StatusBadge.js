import React from 'react';

/**
 * StatusBadge - Displays qualitative architectural flags.
 * Used to distinguish capability from robustness/trajectory.
 */
export default function StatusBadge({ type, label, title }) {
  if (!type) return null;

  const getStyle = () => {
    switch (type.toUpperCase()) {
      case 'ROBUST':
        return {
          color: 'var(--accent-green, #3fb950)',
          bg: 'rgba(63, 185, 80, 0.15)',
          border: 'rgba(63, 185, 80, 0.3)',
        };
      case 'STABLE':
        return {
          color: 'var(--accent-blue, #58a6ff)',
          bg: 'rgba(88, 166, 255, 0.15)',
          border: 'rgba(88, 166, 255, 0.3)',
        };
      case 'FRAGILE':
        return {
          color: 'var(--accent-yellow, #d29922)',
          bg: 'rgba(210, 153, 34, 0.15)',
          border: 'rgba(210, 153, 34, 0.3)',
        };
      case 'FRONTIER_SIGNAL':
        return {
          color: 'var(--accent-purple, #bc8cff)',
          bg: 'rgba(188, 140, 255, 0.15)',
          border: 'rgba(188, 140, 255, 0.3)',
        };
      case 'SLOW_BURN':
        return {
          color: 'var(--accent-orange, #f0883e)',
          bg: 'rgba(240, 136, 62, 0.15)',
          border: 'rgba(240, 136, 62, 0.3)',
        };
      case 'FAILED':
        return {
          color: 'var(--accent-red, #f85149)',
          bg: 'rgba(248, 81, 73, 0.15)',
          border: 'rgba(248, 81, 73, 0.3)',
        };
      case 'DIVERGED':
        return {
          color: 'var(--accent-red, #f85149)',
          bg: 'rgba(248, 81, 73, 0.15)',
          border: 'rgba(248, 81, 73, 0.3)',
        };
      case 'STABLE_GENERALIZER':
        return {
          color: 'var(--accent-cyan, #39d2c0)',
          bg: 'rgba(57, 210, 192, 0.15)',
          border: 'rgba(57, 210, 192, 0.3)',
        };
      default:
        return {
          color: 'var(--text-muted, #8b949e)',
          bg: 'rgba(139, 148, 158, 0.1)',
          border: 'rgba(139, 148, 158, 0.2)',
        };
    }
  };

  const style = getStyle();

  return (
    <span
      title={title}
      style={{
        display: 'inline-block',
        padding: '1px 5px',
        borderRadius: 3,
        fontSize: 9,
        fontWeight: 700,
        textTransform: 'uppercase',
        letterSpacing: '0.02em',
        color: style.color,
        background: style.bg,
        border: `1px solid ${style.border}`,
        whiteSpace: 'nowrap',
        marginRight: 4,
        marginBottom: 2,
      }}
    >
      {label || type.replace('_', ' ')}
    </span>
  );
}
