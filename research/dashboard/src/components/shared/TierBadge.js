import React from 'react';
import { TIER_COLORS, TIER_LABELS } from '../../utils/scoringEngine';
import { decisionGate as sharedDecisionGate } from '../../utils/candidateState';

export function decisionGate(entry) {
  return sharedDecisionGate(entry);
}

const CHECK_LABELS = {
  screeningEvidence: 'Screening evidence',
  investigationEvidence: 'Investigation evidence',
  robustnessFloor: 'Robustness \u2265 0.50',
  validationEvidence: 'Validation evidence',
  baselineBeatsReference: 'Baseline < 1.0',
  consistencyBounded: 'Multi-seed std \u2264 0.12',
  ckaArtifactBacked: 'CKA artifact-backed',
};

export default function TierBadge({ tier, entry }) {
  if (!tier) return null;

  const gate = decisionGate(entry || {});

  const tooltipLines = ['Promotion criteria:'];
  Object.entries(gate.checks).forEach(([name, ok]) => {
    tooltipLines.push(`${ok ? '\u2713' : '\u2717'} ${CHECK_LABELS[name] || name}`);
  });

  if (tier !== 'breakthrough' && gate.missing.length > 0) {
    tooltipLines.push('');
    tooltipLines.push(`Missing for breakthrough: ${gate.missing.map(m => CHECK_LABELS[m] || m).join(', ')}`);
  }

  const tooltip = tooltipLines.join('\n');
  const semanticWarning = entry?.semantic_warning || null;
  const semanticWarningTitle = semanticWarning
    ? [semanticWarning.message, ...(semanticWarning.evidence || [])].join('\n')
    : '';

  return (
    <span style={{ display: 'inline-flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
      <span
        title={tooltip}
        style={{
          padding: '2px 8px',
          borderRadius: 4,
          fontSize: 11,
          fontWeight: 600,
          color: TIER_COLORS[tier] || 'var(--text-muted)',
          background: `${TIER_COLORS[tier] || 'var(--text-muted)'}22`,
          border: `1px solid ${TIER_COLORS[tier] || 'var(--border)'}`,
          textTransform: 'uppercase',
          cursor: 'help',
        }}
      >
        {TIER_LABELS[tier] || tier}
      </span>
      {semanticWarning && (
        <span
          title={semanticWarningTitle}
          style={{
            padding: '2px 8px',
            borderRadius: 4,
            fontSize: 10,
            fontWeight: 700,
            color: 'var(--accent-yellow)',
            background: 'rgba(210, 153, 34, 0.12)',
            border: '1px solid rgba(210, 153, 34, 0.45)',
            whiteSpace: 'nowrap',
          }}
        >
          {semanticWarning.label || 'Warning'}
        </span>
      )}
    </span>
  );
}
