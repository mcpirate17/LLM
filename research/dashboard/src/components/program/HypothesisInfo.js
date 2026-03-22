import React from 'react';

export function HypothesisInfo({ hypothesis }) {
  if (!hypothesis) return null;
  const colors = {
    confirmed: 'var(--accent-green)',
    refuted: 'var(--accent-red)',
    inconclusive: 'var(--accent-yellow)',
    pending: 'var(--text-muted)',
    testing: 'var(--accent-blue)',
  };
  return (
    <div style={{
      padding: 12, background: 'var(--bg-tertiary)', borderRadius: 4,
      borderLeft: `2px solid ${colors[hypothesis.status] || 'var(--border)'}`,
      fontSize: 13,
    }}>
      <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', marginBottom: 4,
        color: colors[hypothesis.status] || 'var(--text-muted)' }}>
        Hypothesis: {hypothesis.status}
      </div>
      <div style={{ marginBottom: 4 }}>{hypothesis.prediction}</div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
        <em>Metric:</em> {hypothesis.success_metric}
      </div>
      {hypothesis.outcome_summary && (
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4 }}>
          <em>Outcome:</em> {hypothesis.outcome_summary}
        </div>
      )}
    </div>
  );
}

export default React.memo(HypothesisInfo);
