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

  return (
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
  );
}
