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
    bindingBonus: { label: 'Binding Range', color: '#a371f7' },
    blimpBonus: { label: 'BLiMP Linguistic', color: '#79c0ff' },
    routingOverheadPenalty: { label: 'Routing Overhead', color: 'var(--accent-red)' },
    sparsityBonus: { label: 'Sparsity', color: '#56d364' },
    learningSpeedBonus: { label: 'Learning Speed', color: '#db61a2' },
    externalComparisonBonus: { label: 'vs Baseline', color: '#f0883e' },
    referenceDeltaBonus: { label: 'Ref Delta', color: '#e3b341' },
    robustnessBonus: { label: 'Robustness Bonus', color: '#d29922' },
  };

  const positives = Object.entries(breakdown)
    .filter(([, weight]) => weight > 0)
    .map(([key, weight]) => ({
      key,
      weight,
      ...(keyMap[key] || { label: key, color: 'var(--border)' })
    }));

  const penalties = Object.entries(breakdown)
    .filter(([, weight]) => weight < 0)
    .map(([key, weight]) => ({
      key,
      weight,
      ...(keyMap[key] || { label: key, color: 'var(--accent-red)' })
    }));

  const components = [...positives, ...penalties];
  const total = positives.reduce((acc, c) => acc + (Number(c.weight) || 0), 0) || 1;

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
        {positives.map(c => (
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
          {positives.map(c => (
            <div key={`break-${c.key}`} style={{ marginBottom: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span>{c.label}</span>
                <span>+{Number(c.weight).toFixed(1)}</span>
              </div>
              <div style={{ height: 4, background: 'var(--bg-tertiary)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{ width: `${(c.weight / total) * 100}%`, height: '100%', background: c.color }} />
              </div>
            </div>
          ))}
          {penalties.length > 0 && (
            <>
              <div style={{ borderTop: '1px solid var(--border)', margin: '6px 0', paddingTop: 4, fontWeight: 600, fontSize: 10, color: 'var(--accent-red)' }}>Penalties</div>
              {penalties.map(c => (
                <div key={`pen-${c.key}`} style={{ marginBottom: 4 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--accent-red)' }}>{c.label}</span>
                    <span style={{ color: 'var(--accent-red)', fontWeight: 600 }}>{Number(c.weight).toFixed(1)}</span>
                  </div>
                </div>
              ))}
            </>
          )}
          <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Internal composite only.</div>
        </div>
      )}
    </div>
  );
}

export default ScoreBreakdown;
