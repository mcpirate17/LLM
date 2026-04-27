import { candidateEligibility } from './candidateState';

describe('candidateEligibility', () => {
  test('screening rows without investigation evidence remain investigation-eligible', () => {
    const eligibility = candidateEligibility({ tier: 'screening' });
    expect(eligibility.investigationEligible).toBe(true);
    expect(eligibility.validationEligible).toBe(false);
    expect(eligibility.queueReason).toBe(null);
  });

  test('screening rows with investigation evidence are no longer queued for investigation', () => {
    const eligibility = candidateEligibility({
      tier: 'screening',
      investigation_loss_ratio: 0.42,
    });
    expect(eligibility.investigationEligible).toBe(false);
    expect(eligibility.validationEligible).toBe(false);
    expect(eligibility.queueReason).toBe('already_investigated_unchanged');
  });

  test('training-only validation rows stay validation-eligible', () => {
    const eligibility = candidateEligibility({
      tier: 'validation',
      validation_passed: 1,
      validation_loss_ratio: 0.31,
      validation_baseline_ratio: 1.08,
      capability_quality: { status: 'training_only' },
    });
    expect(eligibility.investigationEligible).toBe(false);
    expect(eligibility.validationEligible).toBe(true);
    expect(eligibility.confirmationEligible).toBe(true);
    expect(eligibility.queueReason).toBe(null);
  });

  test('capability-qualified validation rows move to champion confirmation', () => {
    const eligibility = candidateEligibility({
      tier: 'validation',
      validation_passed: 1,
      validation_loss_ratio: 0.22,
      validation_baseline_ratio: 0.91,
      capability_quality: { status: 'qualified' },
    });
    expect(eligibility.investigationEligible).toBe(false);
    expect(eligibility.validationEligible).toBe(false);
    expect(eligibility.confirmationEligible).toBe(true);
    expect(eligibility.queueReason).toBe(null);
  });

  test('breakthrough rows are confirmation-eligible even if validation is complete', () => {
    const eligibility = candidateEligibility({
      tier: 'breakthrough',
      validation_passed: 1,
      capability_quality: { status: 'breakthrough' },
    });
    expect(eligibility.investigationEligible).toBe(false);
    expect(eligibility.validationEligible).toBe(false);
    expect(eligibility.confirmationEligible).toBe(true);
    expect(eligibility.queueReason).toBe(null);
  });

  test('failed validation rows do not enter confirmation', () => {
    const eligibility = candidateEligibility({
      tier: 'validation',
      validation_passed: 0,
      validation_loss_ratio: 1.22,
    });
    expect(eligibility.confirmationEligible).toBe(false);
    expect(eligibility.queueEligible).toBe(true);
    expect(eligibility.queueReason).toBe(null);
  });
});
