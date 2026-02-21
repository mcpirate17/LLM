import React from 'react';

export default function RatingBadge({ program }) {
  const lr = program.loss_ratio;
  const nov = program.novelty_score || 0;
  const bl = program.baseline_loss_ratio;

  let color, label;
  if (bl != null && bl < 1 && lr < 0.5 && nov > 0.7) {
    color = 'var(--accent-green)'; label = 'S1 - Exceptional';
  } else if (lr < 0.5 && nov > 0.5) {
    color = 'var(--accent-green)'; label = 'S1 - Strong';
  } else if (lr < 0.7) {
    color = 'var(--accent-yellow)'; label = 'S1 - Moderate';
  } else {
    color = 'var(--accent-orange, #f0883e)'; label = 'S1 - Marginal';
  }

  return (
    <span style={{
      padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
      background: `${color}22`, color, border: `1px solid ${color}44`,
    }}>
      {label}
    </span>
  );
}
