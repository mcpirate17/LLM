import React from 'react';

export function WhatIHaveLearned({ summary }) {
  if (!summary || !summary.bullets || summary.bullets.length === 0) {
    return null;
  }

  return (
    <div className="card">
      <div className="card-title">What I've learned</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Aria's synthesized takeaways across grammar adaptation, frontier quality, clusters, and recent experiment outcomes.
      </p>
      <ul style={{ margin: 0, paddingLeft: 18, color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: 6 }}>
        {summary.bullets.map((bullet, index) => (
          <li key={index} style={{ fontSize: 12, lineHeight: 1.5 }}>
            {bullet}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default WhatIHaveLearned;
