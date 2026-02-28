import React from 'react';

function ConfidenceBar({ label, value, color }) {
  const clamped = Math.max(0, Math.min(100, Number(value) || 0));
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '150px 1fr 44px', gap: 10, alignItems: 'center' }}>
      <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{label}</span>
      <div style={{ position: 'relative', height: 10, borderRadius: 999, background: 'var(--bg-tertiary)', border: '1px solid var(--border)', overflow: 'hidden' }}>
        <div
          style={{
            width: `${clamped}%`,
            height: '100%',
            background: color,
            opacity: 0.75,
            transition: 'width 0.25s ease',
          }}
        />
        <div style={{ position: 'absolute', left: '45%', top: -1, bottom: -1, width: 1, background: 'var(--border)' }} />
        <div style={{ position: 'absolute', left: '75%', top: -1, bottom: -1, width: 1, background: 'var(--border)' }} />
      </div>
      <span style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'right' }}>{clamped.toFixed(0)}%</span>
    </div>
  );
}

export default function ConfidenceInfographic({
  factors,
  decisionReadyCount,
  totalCandidates,
  avgPromotionScore,
  avgReproCompleteness,
}) {
  return (
    <div style={{ display: 'grid', gap: 8, marginBottom: 12 }}>
      <ConfidenceBar label="Experiment depth" value={(factors.experiments || 0) * 100} color="var(--accent-blue)" />
      <ConfidenceBar label="Program volume" value={(factors.programs || 0) * 100} color="var(--accent-purple)" />
      <ConfidenceBar label="Ranking coverage" value={(factors.rankings || 0) * 100} color="var(--accent-green)" />
      <ConfidenceBar label="Op coverage" value={(factors.opCoverage || 0) * 100} color="var(--accent-yellow)" />
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 4, fontSize: 12, color: 'var(--text-secondary)' }}>
        <span>Decision-ready: {decisionReadyCount}/{totalCandidates || 0}</span>
        <span>Promotion confidence: {avgPromotionScore}%</span>
        <span>Repro completeness: {avgReproCompleteness}%</span>
      </div>
    </div>
  );
}
