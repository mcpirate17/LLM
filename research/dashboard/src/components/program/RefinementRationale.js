import React from 'react';

export function RefinementRationale({ program }) {
  const graphMetadata = program?.graph_json_parsed?.metadata;
  const refinement = graphMetadata?.refinement;
  if (!refinement || typeof refinement !== 'object') return null;

  const intent = refinement.intent || 'balanced';
  const intentScore = Number(refinement.intent_score);
  const hasIntentScore = Number.isFinite(intentScore);
  const sourceResultId = refinement.source_result_id;
  const seedFingerprint = refinement.seed_fingerprint;
  const fallback = Boolean(refinement.fallback);
  const scoreBreakdown = refinement.intent_score_breakdown || {};
  const weightedTerms = scoreBreakdown?.weighted_terms || {};
  const weightedEntries = Object.entries(weightedTerms)
    .filter(([, value]) => Number.isFinite(Number(value)))
    .sort((a, b) => Number(b[1]) - Number(a[1]));
  const scoreBreakdownTooltip = weightedEntries.length > 0
    ? weightedEntries.map(([name, value]) => `${name}: ${Number(value).toFixed(4)}`).join('\n')
    : 'No score components available';

  return (
    <div style={{
      padding: 12,
      background: 'var(--bg-tertiary)',
      borderRadius: 6,
      border: '1px solid var(--border)',
    }}>
      <div style={{
        fontSize: 12,
        color: 'var(--text-secondary)',
        fontWeight: 600,
        textTransform: 'uppercase',
        marginBottom: 8,
      }}>
        Refinement Rationale
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12 }}>
        <div style={{ color: 'var(--text-muted)' }}>Intent</div>
        <div style={{ fontWeight: 600 }}>{intent}</div>
        <div style={{ color: 'var(--text-muted)' }}>Intent Score</div>
        <div
          style={{ fontWeight: 600, cursor: weightedEntries.length > 0 ? 'help' : 'default' }}
          title={scoreBreakdownTooltip}
        >
          {hasIntentScore ? intentScore.toFixed(4) : '--'}
        </div>
        {sourceResultId && (
          <>
            <div style={{ color: 'var(--text-muted)' }}>Parent Result</div>
            <div style={{ fontFamily: 'monospace' }}>{String(sourceResultId).slice(0, 12)}</div>
          </>
        )}
        {seedFingerprint && (
          <>
            <div style={{ color: 'var(--text-muted)' }}>Parent Fingerprint</div>
            <div style={{ fontFamily: 'monospace' }}>{seedFingerprint}</div>
          </>
        )}
        <div style={{ color: 'var(--text-muted)' }}>Selection Path</div>
        <div style={{ color: fallback ? 'var(--accent-yellow)' : 'var(--accent-green)' }}>
          {fallback ? 'fallback generation' : 'learning-guided refinement'}
        </div>
      </div>
      {weightedEntries.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
          Components:{' '}
          {weightedEntries
            .slice(0, 3)
            .map(([name, value]) => `${name} ${Number(value).toFixed(3)}`)
            .join(' · ')}
        </div>
      )}
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
        Score combines learned op success and intent-specific objective weighting.
      </div>
    </div>
  );
}

export default RefinementRationale;
