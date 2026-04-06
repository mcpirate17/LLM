import React from 'react';

export default function ReportCard({ label, stats, highlight, selected = false, onClick }) {
  const isEmpty = stats && stats.experiments === 0;
  const isThemeCard = stats && stats.experiments === null;

  return (
    <div
      className="card"
      onClick={onClick}
      style={{
        cursor: 'pointer',
        opacity: isEmpty ? 0.5 : 1,
        borderLeft: selected
          ? '3px solid var(--accent-blue)'
          : highlight
            ? '3px solid var(--accent-purple)'
            : undefined,
        borderColor: selected ? 'var(--accent-blue)' : undefined,
        boxShadow: selected ? '0 0 0 1px rgba(88, 166, 255, 0.24), var(--shadow-elevated)' : undefined,
        transition: 'transform 0.1s ease, box-shadow 0.1s ease',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.transform = 'translateY(-2px)';
        e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
      }}
      onMouseLeave={e => {
        e.currentTarget.style.transform = '';
        e.currentTarget.style.boxShadow = '';
      }}
      role="button"
      tabIndex={0}
      onKeyDown={e => { if ((e.key === 'Enter' || e.key === ' ') && onClick) { e.preventDefault(); onClick(); } }}
      aria-label={`Open ${label} report`}
    >
      <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>
        {label}
      </div>

      {isEmpty && (
        <div style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>
          No experiments
        </div>
      )}

      {isThemeCard && (
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Click to generate scoped report
        </div>
      )}

      {stats && !isEmpty && !isThemeCard && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-blue)' }}>
              {stats.experiments}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Experiments
            </div>
          </div>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-purple)' }}>
              {(stats.programs || 0).toLocaleString()}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Programs
            </div>
          </div>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-green)' }}>
              {stats.s1Survivors || 0}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              S1 Survivors
            </div>
          </div>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: stats.passRate > 0.05 ? 'var(--accent-green)' : 'var(--accent-yellow)' }}>
              {(stats.passRate * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Pass Rate
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
