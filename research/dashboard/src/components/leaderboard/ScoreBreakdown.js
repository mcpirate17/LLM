import React, { useState } from 'react';
import { scoreColor } from '../../utils/format';
import { candidateScore, candidateScoreBreakdown, TIER_ORDER } from '../../utils/scoringEngine';

export function ScoreBreakdown({ entry }) {
  const [show, setShow] = useState(false);
  const breakdown = candidateScoreBreakdown(entry, TIER_ORDER);
  const score = candidateScore(entry, TIER_ORDER);

  const keyMap = {
    sLoss: { label: 'Screening Loss', color: 'var(--accent-blue)' },
    iLoss: { label: 'Investigation Loss', color: '#1f6feb' },
    loss: { label: 'Loss', color: 'var(--accent-blue)' },
    novelty: { label: 'Novelty', color: 'var(--accent-purple)' },
    vBase: { label: 'Baseline', color: 'var(--accent-green)' },
    baseline: { label: 'Baseline', color: 'var(--accent-green)' },
    robust: { label: 'Robustness', color: 'var(--accent-yellow)' },
    consistency: { label: 'Consistency', color: '#d29922' },
    tierBonus: { label: 'Tier Bonus', color: 'var(--accent-orange)' },
    throughput: { label: 'Throughput', color: 'var(--text-muted)' },
    efficiencyBonus: { label: 'Efficiency', color: '#58a6ff' },
    routingBonus: { label: 'Routing', color: '#3fb950' },
    adaptiveBonus: { label: 'Adaptive Compute', color: '#c77dff' },
  };

  const components = Object.entries(breakdown)
    .filter(([, weight]) => weight > 0)
    .map(([key, weight]) => ({
      key,
      weight,
      ...(keyMap[key] || { label: key, color: 'var(--border)' })
    }));

  const total = components.reduce((acc, c) => acc + (Number(c.weight) || 0), 0) || 1;

  return (
    <div
      style={{ minWidth: 80, position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <div style={{ fontWeight: 600, color: scoreColor(score), marginBottom: 4 }}>
        {score}
      </div>
      <div style={{ display: 'flex', height: 4, borderRadius: 2, overflow: 'hidden', background: 'var(--bg-tertiary)' }}>
        {components.map(c => (
          <div
            key={c.key}
            style={{
              width: `${(c.weight / total) * 100}%`,
              background: c.color,
              height: '100%'
            }}
          />
        ))}
      </div>
      {show && (
        <div style={{
          position: 'absolute',
          top: '100%',
          left: '50%',
          transform: 'translateX(-50%)',
          marginTop: 8,
          padding: '10px 12px',
          background: '#161b22',
          border: '1px solid var(--border)',
          borderRadius: 6,
          boxShadow: '0 6px 16px rgba(0,0,0,0.45)',
          zIndex: 1000,
          minWidth: 220,
          fontSize: 11,
          color: 'var(--text-primary)',
        }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Score Breakdown</div>
          {components.map(c => (
            <div key={`break-${c.key}`} style={{ marginBottom: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span>{c.label}</span>
                <span>{Number(c.weight).toFixed(1)}</span>
              </div>
              <div style={{ height: 4, background: 'var(--bg-tertiary)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{ width: `${(c.weight / total) * 100}%`, height: '100%', background: c.color }} />
              </div>
            </div>
          ))}
          <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Internal composite only.</div>
        </div>
      )}
    </div>
  );
}

export default ScoreBreakdown;
