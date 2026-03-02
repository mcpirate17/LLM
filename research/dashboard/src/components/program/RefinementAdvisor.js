import React from 'react';

export function RefinementAdvisor({ analysis, loading, error, onLaunchRefinement, actionStarting }) {
  if (loading) return <div style={{ fontSize: 12, color: 'var(--text-muted)', padding: 12 }}>Analyzing program...</div>;
  if (error) return <div style={{ fontSize: 12, color: 'var(--accent-red)', padding: 12 }}>Analysis error: {error}</div>;
  if (!analysis || analysis.analysis_quality === 'no_data') return null;

  const recipe = analysis.recipe || {};
  const opHealth = analysis.op_health || [];
  const additions = analysis.recommended_additions || [];
  const gaps = (analysis.behavioral_gaps || []).filter(g => g.severity !== 'low');
  const stats = analysis.population_stats || {};

  const intentColors = { quality: 'var(--accent-red)', novelty: 'var(--accent-blue)', compression: '#1f7a4f', balanced: 'var(--accent-yellow)' };
  const healthColors = { strong: 'var(--accent-green)', weak: 'var(--accent-red)', risky: 'var(--accent-yellow)', untested: 'var(--text-muted)', neutral: 'var(--text-secondary)' };

  return (
    <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6, border: '1px solid var(--border)' }}>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
        Refinement Advisor
      </div>

      {/* Recipe banner */}
      <div style={{
        padding: 10, borderRadius: 6, marginBottom: 10,
        background: 'var(--bg-secondary)', border: `1px solid ${intentColors[recipe.recommended_intent] || 'var(--border)'}`,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <span style={{
            fontSize: 10, fontWeight: 700, textTransform: 'uppercase', padding: '2px 8px', borderRadius: 3,
            background: intentColors[recipe.recommended_intent] || 'var(--border)', color: '#fff',
          }}>
            {recipe.recommended_intent || 'balanced'}
          </span>
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            confidence: {recipe.confidence || 'low'}
          </span>
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            {stats.n_stage1_passed || 0} S1 survivors analyzed
          </span>
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{recipe.human_summary}</div>
      </div>

      {/* Op health grid */}
      {opHealth.length > 0 && (
        <div style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Op Health</div>
          <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4 }}>
            {opHealth.map(op => (
              <div
                key={op.op_name}
                style={{
                  minWidth: 100, padding: '6px 8px', borderRadius: 4, fontSize: 11,
                  background: 'var(--bg-secondary)', border: `1px solid ${healthColors[op.health] || 'var(--border)'}`,
                }}
                title={op.swap_candidates?.length
                  ? `Swap candidates: ${op.swap_candidates.map(c => `${c.op_name} (${(c.s1_rate * 100).toFixed(0)}%)`).join(', ')}`
                  : `${op.recommendation} — S1 rate: ${(op.global_s1_rate * 100).toFixed(1)}%`}
              >
                <div style={{ fontFamily: 'monospace', fontWeight: 600, marginBottom: 2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {op.op_name}
                </div>
                <div style={{ color: healthColors[op.health], fontWeight: 600 }}>
                  {op.health}
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>
                  S1: {(op.global_s1_rate * 100).toFixed(0)}% ({op.n_used} uses)
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                  {op.recommendation}
                  {op.swap_candidates?.length > 0 && ` (${op.swap_candidates.length} alt)`}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommended additions */}
      {additions.length > 0 && (
        <details style={{ marginBottom: 10 }}>
          <summary style={{ fontSize: 11, color: 'var(--text-muted)', cursor: 'pointer', marginBottom: 4 }}>
            Recommended Additions ({additions.length})
          </summary>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 4 }}>
            {additions.map(a => (
              <div key={a.op_name} style={{
                display: 'flex', gap: 8, alignItems: 'center', fontSize: 11,
                padding: '4px 8px', background: 'var(--bg-secondary)', borderRadius: 4,
              }}>
                <span style={{ fontFamily: 'monospace', fontWeight: 600, minWidth: 120 }}>{a.op_name}</span>
                <span style={{ color: 'var(--accent-green)' }}>S1: {(a.global_s1_rate * 100).toFixed(0)}%</span>
                <span style={{ color: 'var(--text-muted)' }}>{a.top_performer_frequency} uses</span>
                <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>{a.rationale}</span>
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Behavioral gaps */}
      {gaps.length > 0 && (
        <div style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Behavioral Gaps</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
            {gaps.map(g => (
              <div key={g.metric} style={{
                padding: '6px 8px', background: 'var(--bg-secondary)', borderRadius: 4, fontSize: 11,
                borderLeft: `3px solid ${g.severity === 'high' ? 'var(--accent-red)' : 'var(--accent-yellow)'}`,
              }}>
                <div style={{ fontWeight: 600, marginBottom: 2 }}>{g.label}</div>
                <div style={{ color: 'var(--text-muted)' }}>
                  Program: {g.program_value?.toFixed(3)} vs Pop: {g.population_mean?.toFixed(3)} (z={g.z_score > 0 ? '+' : ''}{g.z_score?.toFixed(1)})
                </div>
                {g.improvement_ops?.length > 0 && (
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                    Try: {g.improvement_ops.join(', ')}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action button */}
      <button
        className="start-btn"
        disabled={actionStarting === 'refine_advisor'}
        onClick={() => onLaunchRefinement(recipe.recommended_intent || 'balanced', 'refine_advisor', 'Failed to start analysis-driven refinement')}
        style={{ padding: '6px 16px', fontSize: 12, background: 'var(--accent-purple)', borderColor: 'var(--accent-purple)' }}
        title={recipe.primary_target || 'Refine using data-driven analysis'}
      >
        {actionStarting === 'refine_advisor' ? 'Starting...' : 'Refine with Recommendation'}
      </button>
    </div>
  );
}

export default RefinementAdvisor;
