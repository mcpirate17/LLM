export const TIER_ORDER = {
  breakthrough: 4,
  validation: 3,
  validation_pending: 2.5,
  investigation: 2,
  validation_failed: 1.5,
  investigation_failed: 1,
  screening: 1,
  screened_out: 0,
};

export const TIER_COLORS = {
  screening: 'var(--accent-blue)',
  screened_out: 'var(--text-muted)',
  investigation_failed: 'var(--accent-red)',
  validation_failed: 'var(--accent-red)',
  validation_pending: 'var(--accent-purple)',
  investigation: 'var(--accent-yellow)',
  validation: 'var(--accent-purple)',
  breakthrough: 'var(--accent-green)',
};

export const TIER_LABELS = {
  screening: 'Screening',
  screened_out: 'Failed Screening',
  investigation_failed: 'Failed Investigation',
  validation_failed: 'Failed Validation',
  validation_pending: 'Validation Pending',
  investigation: 'Investigation',
  validation: 'Validation',
  breakthrough: 'Breakthrough',
};

export function bestLoss(entry) {
  const tier = String(entry?.tier || '').toLowerCase();
  if ((tier === 'validation' || tier === 'breakthrough') && entry?.validation_loss_ratio != null) {
    return Number(entry.validation_loss_ratio);
  }
  if ((tier === 'investigation' || tier === 'investigation_failed' || tier === 'validation' || tier === 'breakthrough') && entry?.investigation_loss_ratio != null) {
    return Number(entry.investigation_loss_ratio);
  }
  if (entry?.screening_loss_ratio != null) return Number(entry.screening_loss_ratio);
  if (entry?.loss_ratio != null) return Number(entry.loss_ratio);
  return null;
}

export function percentOfReference(entryLoss, refLoss) {
  const e = Number(entryLoss);
  const r = Number(refLoss);
  if (!Number.isFinite(e) || !Number.isFinite(r) || r <= 0) return null;
  return (e / r) * 100;
}
