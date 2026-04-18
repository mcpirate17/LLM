import { qkvUsageDescriptor } from '../../utils/architecture';
import {
  candidateEligibility as sharedCandidateEligibility,
  reproducibilityPacketStatus as sharedReproducibilityPacketStatus,
} from '../../utils/candidateState';

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
    flags.push({ label: 'CKA heuristic estimate', tone: 'low' });
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
  return sharedCandidateEligibility(entry);
}

export function reproducibilityPacketStatus(entry) {
  return sharedReproducibilityPacketStatus(entry);
}
