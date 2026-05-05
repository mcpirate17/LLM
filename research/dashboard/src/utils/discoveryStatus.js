export function normalizeTier(value) {
  return String(value || '').trim().toLowerCase();
}

export const DISCOVERY_TIER_FILTERS = [
  'all',
  'screening',
  'failed',
  'investigation',
  'validation_pending',
  'validation',
  'breakthrough',
];

export const FAILED_DISCOVERY_TIERS = new Set([
  'screened_out',
  'investigation_failed',
  'investigation_fingerprint_incomplete',
  'validation_failed',
  'failed',
  'rejected',
]);

export const CAPABILITY_QUALITY_ORDER = {
  breakthrough: 5,
  qualified: 4,
  pending: 3,
  training_only: 2,
  investigated: 1,
  exploratory: 0,
};

export function getDiscoveryDisplayStatus(entry) {
  const tier = normalizeTier(entry?.tier);
  const validationPassed = Boolean(entry?.validation_passed);

  if (tier === 'screened_out') {
    return {
      tierKey: 'screened_out',
      label: 'Failed Screening',
      tabTier: 'screened_out',
      isFailure: true,
    };
  }
  if (tier === 'investigation_failed') {
    return {
      tierKey: 'investigation_failed',
      label: 'Failed Investigation',
      tabTier: 'investigation_failed',
      isFailure: true,
    };
  }
  if (tier === 'investigation_fingerprint_incomplete') {
    return {
      tierKey: 'investigation_fingerprint_incomplete',
      label: 'Fingerprint Failed',
      tabTier: 'investigation_fingerprint_incomplete',
      isFailure: true,
    };
  }
  if (tier === 'validation_failed') {
    return {
      tierKey: 'validation_failed',
      label: 'Failed Validation',
      tabTier: 'validation_failed',
      isFailure: true,
    };
  }
  if (tier === 'validation' && !validationPassed) {
    return {
      tierKey: 'validation_pending',
      label: 'Validation Pending',
      tabTier: 'validation_pending',
      isFailure: false,
    };
  }
  return {
    tierKey: tier || 'screening',
    label: null,
    tabTier: tier || 'screening',
    isFailure: false,
  };
}

export function capabilityQualityStatus(entry) {
  return String(entry?.capability_quality?.status || '').trim().toLowerCase();
}

export function capabilityQualityLabel(entry) {
  return String(entry?.capability_quality?.label || '').trim();
}

export function capabilityQualityRank(entry) {
  return CAPABILITY_QUALITY_ORDER[capabilityQualityStatus(entry)] ?? -1;
}

export function matchesActiveTier(entry, activeTier) {
  if (!activeTier || activeTier === 'all') return true;
  const normalized = normalizeTier(activeTier);
  const status = getDiscoveryDisplayStatus(entry);
  if (normalized === 'failed') {
    return status.isFailure || FAILED_DISCOVERY_TIERS.has(status.tierKey);
  }
  return status.tabTier === normalized;
}
