import React from 'react';

export function BenchmarkEvidenceSnapshot({ program, leaderboardEntry }) {
  const tier = leaderboardEntry?.tier;
  const isBreakthrough = tier === 'breakthrough';
  const ratio = Number(program?.baseline_loss_ratio);
  const hasRatio = Number.isFinite(ratio);
  const beatsBaseline = hasRatio && ratio < 1;

  if (!hasRatio && !isBreakthrough) return null;

  return (
    <div style={{
      marginTop: 12,
      padding: 10,
      background: 'var(--bg-tertiary)',
      borderRadius: 6,
      borderLeft: `3px solid ${beatsBaseline ? 'var(--accent-green)' : 'var(--accent-yellow)'}`,
    }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 6 }}>
        Benchmark Evidence Snapshot
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
        <div>
          <strong>Fixed-seed baseline ratio:</strong>{' '}
          {hasRatio ? ratio.toFixed(3) : 'Unavailable'}
          {hasRatio && (
            <span style={{ marginLeft: 6, color: beatsBaseline ? 'var(--accent-green)' : 'var(--accent-red)' }}>
              {beatsBaseline ? '(< 1.0, beats baseline)' : '(≥ 1.0, below baseline)'}
            </span>
          )}
        </div>
        <div>
          <strong>Interpretation:</strong>{' '}
          {hasRatio
            ? (beatsBaseline
              ? 'This architecture outperforms the fixed-seed transformer baseline on the same setup.'
              : 'This architecture does not yet beat the fixed-seed transformer baseline on this snapshot.')
            : 'Baseline comparison was not recorded for this result.'}
        </div>
        {isBreakthrough && (
          <div>
            <strong>Breakthrough note:</strong> tier promotion also requires multi-seed stability and robustness checks beyond this fixed-seed snapshot.
          </div>
        )}
      </div>
    </div>
  );
}

export default React.memo(BenchmarkEvidenceSnapshot);
