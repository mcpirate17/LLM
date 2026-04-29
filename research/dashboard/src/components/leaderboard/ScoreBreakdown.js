import React, { useState } from 'react';
import { SCORE_MAX, scoreColor, scoreGradient, scoreToneLabel } from '../../utils/format';
import { canonicalScoreComponents } from '../../utils/backendScore';

export function ScoreBreakdown({ entry }) {
  const [show, setShow] = useState(false);
  const score = Number(entry?.composite_score || 0);
  const positives = canonicalScoreComponents(entry);
  const scorePercent = Math.max(4, Math.min(100, (score / SCORE_MAX) * 100));

  const total = positives.reduce((acc, c) => acc + (Number(c.weight) || 0), 0) || 1;

  return (
    <div
      style={{ minWidth: 80, position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <div
        title={`${scoreToneLabel(score)} score`}
        style={{
          fontWeight: 700,
          color: scoreColor(score),
          marginBottom: 4,
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {Number.isFinite(score) ? score.toFixed(1) : '--'}
      </div>
      <div className="champion-strip" title="Post-BPE champion score ramp">
        <div
          className="champion-strip-fill"
          style={{
            width: `${scorePercent}%`,
            background: scoreGradient(score),
          }}
        />
      </div>
      <div style={{ display: 'flex', height: 3, borderRadius: 2, overflow: 'hidden', background: 'var(--bg-tertiary)', marginTop: 3 }}>
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
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Score Totals</div>
          <div style={{ color: scoreColor(score), fontSize: 10, marginBottom: 8 }}>
            {scoreToneLabel(score)} band after tiktoken/BPE rescore
          </div>
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
          <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            Loss and understanding are split from the v10 base total; metric rows are not double-counted.
          </div>
        </div>
      )}
    </div>
  );
}

export default ScoreBreakdown;
