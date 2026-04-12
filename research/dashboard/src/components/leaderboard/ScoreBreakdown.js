import React, { useState } from 'react';
import { scoreColor } from '../../utils/format';

export function ScoreBreakdown({ entry }) {
  const [show, setShow] = useState(false);
  const breakdown = entry?.score_breakdown || {};
  const score = Number(entry?.composite_score || 0);

  const keyMap = {
    perf_short: { label: 'Screening Loss', color: 'var(--accent-blue)' },
    perf_medium: { label: 'Investigation Loss', color: '#1f6feb' },
    perf_long: { label: 'Validation Loss', color: 'var(--accent-green)' },
    novelty: { label: 'Novelty', color: 'var(--accent-purple)' },
    robustness: { label: 'Robustness', color: 'var(--accent-yellow)' },
    long_context: { label: 'Long Context', color: '#79c0ff' },
    speed: { label: 'Speed', color: 'var(--text-muted)' },
    binding: { label: 'Binding Range', color: '#a371f7' },
    blimp: { label: 'BLiMP Linguistic', color: '#79c0ff' },
    compression: { label: 'Compression', color: '#56d364' },
    sparsity: { label: 'Sparsity', color: '#3fb950' },
    adaptive_computation: { label: 'Adaptive Compute', color: '#c77dff' },
    routing_savings: { label: 'Routing', color: '#58a6ff' },
    param_efficiency: { label: 'Param Efficiency', color: '#e3b341' },
    learning_efficiency: { label: 'Learning Efficiency', color: '#db61a2' },
    early_convergence: { label: 'Early Convergence', color: '#f0883e' },
    cross_task: { label: 'Cross Task', color: '#3fb950' },
    diagnostic: { label: 'Diagnostic', color: '#d29922' },
    hellaswag: { label: 'HellaSwag', color: 'var(--accent-orange)' },
    hierarchy: { label: 'Hierarchy', color: '#58a6ff' },
    tinystories: { label: 'TinyStories', color: '#56d364' },
  };

  const positives = Object.entries(breakdown)
    .filter(([key, weight]) => Number.isFinite(Number(weight)) && Number(weight) > 0 && !key.includes('penalty'))
    .map(([key, weight]) => ({
      key,
      weight: Number(weight),
      ...(keyMap[key] || { label: key, color: 'var(--border)' })
    }));

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
          <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Internal composite only.</div>
        </div>
      )}
    </div>
  );
}

export default ScoreBreakdown;
