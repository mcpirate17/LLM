import React from 'react';

export function ControlComparison({ data, onStartExperiment }) {
  if (!data || data.status === 'insufficient_data') {
    const nControl = data?.control?.experiments || 0;
    const nLearned = data?.learned?.experiments || 0;
    return (
      <div className="card">
        <div className="card-title">Learning Effectiveness</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
          Compares experiments using learned grammar weights vs control experiments with default weights.
          This tells you whether Aria's learning is actually improving search quality.
        </p>
        <div style={{
          padding: '10px 12px', borderRadius: 6, marginBottom: 10,
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border)',
          fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6,
        }}>
          <div style={{ marginBottom: 4 }}>
            <strong>What's needed:</strong> {'\u2265'}2 control + {'\u2265'}2 learned experiments
          </div>
          <div style={{ marginBottom: 4 }}>
            <strong>Current:</strong> {nControl} control, {nLearned} learned
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            Control experiments run automatically every 5th continuous-mode experiment with default grammar weights.
            Run 5 continuous experiments to guarantee at least 1 control.
          </div>
        </div>
        {onStartExperiment && (
          <button
            className="refresh-btn"
            style={{ fontSize: 11, padding: '4px 10px' }}
            onClick={() => onStartExperiment({
              mode: 'continuous', n_cycles: 5,
              source: 'control_comparison', auto_harden: true,
              preflight_override: true, enforce_preflight: true,
            })}
          >
            Run 5 Continuous
          </button>
        )}
      </div>
    );
  }

  const { control, learned, s1_rate_difference, z_score, significant_at_p05, learned_is_better, interpretation, caveat, matched_pairs } = data;

  const verdictColor = significant_at_p05
    ? (learned_is_better ? 'var(--accent-green)' : 'var(--accent-red, #e74c3c)')
    : 'var(--accent-yellow)';

  return (
    <div className="card">
      <div className="card-title">Learning Effectiveness</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Statistical comparison of experiments using learned grammar weights vs control experiments
        using default weights. A positive difference means learning is helping find better architectures.
      </p>

      <div style={{
        padding: '8px 12px', borderRadius: 6, marginBottom: 12,
        background: significant_at_p05
          ? (learned_is_better ? 'rgba(63,185,80,0.12)' : 'rgba(248,81,73,0.12)')
          : 'rgba(210,153,34,0.12)',
        border: `1px solid ${verdictColor}`,
      }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: verdictColor, marginBottom: 4 }}>
          {interpretation}
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
          z-score: {z_score} {significant_at_p05 ? '(p < 0.05)' : '(not significant)'}
          {matched_pairs ? ` · ${matched_pairs} time-matched pairs` : ''}
        </div>
        {caveat && (
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4, fontStyle: 'italic' }}>
            {caveat}
          </div>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div style={{ padding: '8px 12px', borderRadius: 6, background: 'var(--bg-tertiary)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase' }}>
            Control (Default Weights)
          </div>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)' }}>
            {(control.s1_rate * 100).toFixed(2)}%
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {control.s1_passed}/{control.programs} passed | {control.experiments} experiments
          </div>
        </div>
        <div style={{ padding: '8px 12px', borderRadius: 6, background: 'var(--bg-tertiary)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase' }}>
            Learned Weights
          </div>
          <div style={{ fontSize: 18, fontWeight: 700, color: learned_is_better ? 'var(--accent-green)' : 'var(--text-primary)' }}>
            {(learned.s1_rate * 100).toFixed(2)}%
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {learned.s1_passed}/{learned.programs} passed | {learned.experiments} experiments
          </div>
        </div>
      </div>

      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>
        S1 rate difference: {s1_rate_difference > 0 ? '+' : ''}{(s1_rate_difference * 100).toFixed(2)} percentage points
      </div>
    </div>
  );
}

export default ControlComparison;
