import { qkvUsageDescriptor } from '../../utils/architecture';
import { parseArchSpec } from '../report/reportUtils';

export function toRetentionPercent(value) {
  if (value == null) return null;
  const num = Number(value);
  if (!Number.isFinite(num)) return null;
  return num <= 1.0 ? num * 100 : num;
}

export { leaderboardMetricChips as metricChips } from '../../utils/metricChips';

export function qualityFlags(entry) {
  const flags = [];
  if (entry.cka_source === 'artifact') {
    flags.push({ label: 'CKA artifact-backed', tone: 'high' });
  } else {
    flags.push({ label: 'CKA fallback heuristic', tone: 'low' });
  }
  if (entry.validation_baseline_ratio != null) {
    flags.push({ label: 'Baseline measured', tone: 'medium' });
  } else {
    flags.push({ label: 'Baseline unavailable', tone: 'low' });
  }
  if (entry.routing_confidence_mean != null) {
    flags.push({ label: 'Routing telemetry', tone: 'medium' });
  }
  const qkv = qkvUsageDescriptor(entry);
  flags.push({ label: qkv.label, tone: qkv.tone, detail: qkv.detail });
  return flags;
}

export function candidateEligibility(entry) {
  const tier = typeof entry?.tier === 'string' ? entry.tier.toLowerCase() : '';
  const hasInvestigationEvidence = entry?.investigation_loss_ratio != null;
  const hasValidationEvidence = entry?.validation_loss_ratio != null || Boolean(entry?.validation_passed);

  const investigationEligible = tier === 'screening' && !hasInvestigationEvidence;
  const validationEligible = tier === 'investigation' && Boolean(entry?.investigation_passed) && !hasValidationEvidence;

  let queueReason = null;
  if (!investigationEligible && !validationEligible) {
    if (tier === 'screening' && hasInvestigationEvidence) {
      queueReason = 'already_investigated_unchanged';
    } else if (tier === 'investigation' && !entry?.investigation_passed) {
      queueReason = 'not_investigation_passed';
    } else if (tier === 'validation' || tier === 'breakthrough') {
      queueReason = 'already_promoted';
    } else {
      queueReason = 'not_progression_eligible';
    }
  }

  return {
    investigationEligible,
    validationEligible,
    queueEligible: investigationEligible || validationEligible,
    queueReason,
  };
}

export function reproducibilityPacketStatus(entry) {
  const spec = parseArchSpec(entry?.arch_spec_json);
  const checks = [
    { label: 'result_id', ok: !!entry?.result_id },
    { label: 'graph_fingerprint', ok: !!entry?.graph_fingerprint },
    { label: 'arch_spec', ok: !!spec },
    { label: 'baseline_ratio', ok: entry?.validation_baseline_ratio != null },
    { label: 'multi_seed_std', ok: entry?.validation_multi_seed_std != null },
    { label: 'cka_artifact', ok: entry?.cka_source === 'artifact' },
  ];
  const readyCount = checks.filter(check => check.ok).length;
  const totalChecks = checks.length;
  const label = readyCount === totalChecks ? 'Ready' : readyCount >= 4 ? 'Partial' : 'Sparse';
  const color = readyCount === totalChecks
    ? 'var(--accent-green)'
    : readyCount >= 4
      ? 'var(--accent-yellow)'
      : 'var(--accent-red)';
  return {
    label,
    color,
    readyCount,
    totalChecks,
    missing: checks.filter(check => !check.ok).map(check => check.label),
  };
}
