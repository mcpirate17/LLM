import React, { useMemo } from 'react';

export function FeedbackLoopSummary({ weights, trajectory, controlComparison, title }) {
  const summary = useMemo(() => {
    const parts = [];

    if (weights?.default && weights?.learned) {
      const deltas = Object.keys(weights.default).map(cat => ({
        cat,
        delta: (weights.learned[cat] || 0) - weights.default[cat]
      })).sort((a, b) => b.delta - a.delta);

      if (deltas.length > 0 && deltas[0].delta > 0.2) {
        parts.push(`Grammar is shifting toward **${deltas[0].cat.replace(/_/g, ' ')}** (+${deltas[0].delta.toFixed(1)}).`);
      }
    }

    if (trajectory?.trend) {
      const trend = trajectory.trend === 'improving' ? 'improving' : trajectory.trend === 'declining' ? 'declining' : 'plateaued';
      parts.push(`Search productivity (S1 pass rate) is currently **${trend}**.`);
    }

    if (controlComparison?.interpretation) {
      parts.push(`Aria's verdict: **${controlComparison.interpretation}**.`);
    }

    return parts;
  }, [weights, trajectory, controlComparison]);

  if (summary.length === 0) return null;

  return (
    <div className="card" style={{ background: 'var(--bg-secondary)', borderLeft: '3px solid var(--accent-purple)' }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent-purple)', marginBottom: 8, textTransform: 'uppercase' }}>
        {title || 'Feedback Loop Summary'}
      </div>
      <div style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.6 }}>
        {summary.map((p, i) => (
          <div key={i} style={{ marginBottom: 4 }}>
            {p.split('**').map((text, idx) => (
              idx % 2 === 1 ? <strong key={idx}>{text}</strong> : text
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export default FeedbackLoopSummary;
