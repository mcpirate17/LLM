import React from 'react';

const CRITIQUE_VERDICT_STYLES = {
  proceed: { color: 'var(--accent-green)', label: 'Proceed', icon: '\u2714' },
  caution: { color: 'var(--accent-yellow)', label: 'Caution', icon: '\u26A0' },
  revise: { color: 'var(--accent-red)', label: 'Revise', icon: '\u2718' },
};

const CRITIQUE_GATE_STYLES = {
  pass: { color: 'var(--accent-green)', bg: 'rgba(63, 185, 80, 0.18)', label: 'Pass' },
  warn: { color: 'var(--accent-yellow)', bg: 'rgba(210, 153, 34, 0.18)', label: 'Warn' },
  fail: { color: 'var(--accent-red)', bg: 'rgba(248, 81, 73, 0.18)', label: 'Fail' },
};

export function HypothesisCritique({ critique }) {
  if (!critique) return null;
  
  const style = CRITIQUE_VERDICT_STYLES[critique.verdict] || CRITIQUE_VERDICT_STYLES.caution;
  const gate = typeof critique.gate === 'string' ? critique.gate : (critique.verdict === 'proceed' ? 'pass' : critique.verdict === 'revise' ? 'fail' : 'warn');
  const gateStyle = CRITIQUE_GATE_STYLES[gate] || CRITIQUE_GATE_STYLES.warn;
  const checks = Array.isArray(critique.checks) ? critique.checks : [];
  const missingFields = Array.isArray(critique.missing_fields)
    ? critique.missing_fields.filter(Boolean)
    : [];
  const missingFieldLabels = {
    source_selection_rule: 'source_selection_rule',
    mutation_mechanism: 'mutation_mechanism',
    intent_weights: 'intent_weights',
    primary_metric: 'primary_metric',
    success_criteria: 'success_criteria',
    confounders_checklist: 'confounders_checklist',
    fallback_plan: 'fallback_plan',
  };
  const hasConcerns = critique.concerns && critique.concerns.length > 0;
  const hasSuggestions = critique.suggestions && critique.suggestions.length > 0;

  return (
    <div style={{
      marginBottom: 10,
      padding: '8px 10px',
      borderRadius: 6,
      border: `1px solid \${style.color}`,
      background: `\${style.color}11`,
      fontSize: 12,
      lineHeight: 1.5,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: hasConcerns || hasSuggestions ? 6 : 0 }}>
        <span style={{ fontSize: 14 }}>{style.icon}</span>
        <strong style={{ color: style.color }}>Hypothesis Review: {style.label}</strong>
        <span style={{
          marginLeft: 8,
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: 0.3,
          textTransform: 'uppercase',
          color: gateStyle.color,
          background: gateStyle.bg,
          border: `1px solid \${gateStyle.color}`,
          borderRadius: 4,
          padding: '1px 6px',
        }}>
          Gate: {gateStyle.label}
        </span>
        {critique.confidence != null && (
          <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 'auto' }}>
            confidence {(critique.confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>
      {checks.length > 0 && (
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 6,
          marginBottom: hasConcerns || hasSuggestions ? 6 : 0,
          paddingLeft: 20,
        }}>
          {checks.map((check, idx) => {
            const checkStyle = CRITIQUE_GATE_STYLES[check?.status] || CRITIQUE_GATE_STYLES.warn;
            const label = check?.label || check?.key || `Check \${idx + 1}`;
            return (
              <span
                key={`\${label}-\${idx}`}
                style={{
                  fontSize: 10,
                  color: checkStyle.color,
                  background: checkStyle.bg,
                  border: `1px solid \${checkStyle.color}`,
                  borderRadius: 4,
                  padding: '1px 6px',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 4,
                }}
              >
                <strong>{checkStyle.label}</strong>
                <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
              </span>
            );
          })}
        </div>
      )}
      {missingFields.length > 0 && (
        <div style={{
          marginBottom: hasConcerns || hasSuggestions ? 6 : 0,
          paddingLeft: 20,
        }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 4 }}>
            Missing fields:
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {missingFields.map((field) => (
              <span
                key={field}
                style={{
                  fontSize: 10,
                  color: 'var(--accent-yellow)',
                  background: 'rgba(210, 153, 34, 0.18)',
                  border: '1px solid var(--accent-yellow)',
                  borderRadius: 4,
                  padding: '1px 6px',
                  fontFamily: 'monospace',
                }}
              >
                {missingFieldLabels[field] || field}
              </span>
            ))}
          </div>
        </div>
      )}
      {hasConcerns && (
        <div style={{ marginBottom: hasSuggestions ? 4 : 0 }}>
          {critique.concerns.map((c, i) => (
            <div key={i} style={{ color: 'var(--text-secondary)', paddingLeft: 20 }}>
              &bull; {c}
            </div>
          ))}
        </div>
      )}
      {hasSuggestions && (
        <div>
          {critique.suggestions.map((s, i) => (
            <div key={i} style={{ color: 'var(--text-muted)', paddingLeft: 20, fontStyle: 'italic' }}>
              &rarr; {s}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default HypothesisCritique;
