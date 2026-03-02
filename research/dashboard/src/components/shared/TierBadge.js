import React from 'react';
import { TIER_COLORS, TIER_LABELS } from '../../utils/scoringEngine';

export function decisionGate(entry) {
  const checks = {
    screeningEvidence: entry.screening_loss_ratio != null && entry.screening_novelty != null,
    investigationEvidence: entry.investigation_loss_ratio != null && entry.investigation_robustness != null,
    robustnessFloor: entry.investigation_robustness != null && entry.investigation_robustness >= 0.5,
    validationEvidence: entry.validation_loss_ratio != null
      && entry.validation_baseline_ratio != null
      && entry.validation_multi_seed_std != null,
    baselineBeatsReference: entry.validation_baseline_ratio != null && entry.validation_baseline_ratio < 1.0,
    consistencyBounded: entry.validation_multi_seed_std != null && entry.validation_multi_seed_std <= 0.12,
  };
  const decisionReady = Object.values(checks).every(Boolean);
  const missing = Object.entries(checks)
    .filter(([, ok]) => !ok)
    .map(([name]) => name);
  return {
    decisionReady,
    label: decisionReady ? 'Decision-Ready' : 'Exploratory',
    color: decisionReady ? 'var(--accent-green)' : 'var(--accent-yellow)',
    missing,
    checks,
  };
}

const CHECK_LABELS = {
  screeningEvidence: 'Screening evidence',
  investigationEvidence: 'Investigation evidence',
  robustnessFloor: 'Robustness \u2265 0.50',
  validationEvidence: 'Validation evidence',
  baselineBeatsReference: 'Baseline < 1.0',
  consistencyBounded: 'Multi-seed std \u2264 0.12',
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
