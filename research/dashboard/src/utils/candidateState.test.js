import { candidateEligibility } from './candidateState';

describe('candidateEligibility', () => {
  test('screening rows without investigation evidence remain investigation-eligible', () => {
    const eligibility = candidateEligibility({ tier: 'screening' });
    expect(eligibility.investigationEligible).toBe(true);
    expect(eligibility.validationEligible).toBe(false);
    expect(eligibility.queueReason).toBe(null);
  });

  test('known failed stage1 screening rows are not investigation-eligible', () => {
    const eligibility = candidateEligibility({
      tier: 'screening',
      stage1_passed: false,
    });
    expect(eligibility.investigationEligible).toBe(false);
    expect(eligibility.queueEligible).toBe(false);
    expect(eligibility.queueReason).toBe('not_progression_eligible');
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

  test('passed investigation rows without ranker evidence enter capability ranking', () => {
    const eligibility = candidateEligibility({
      tier: 'investigation',
      investigation_passed: 1,
      investigation_loss_ratio: 0.38,
    });
    expect(eligibility.investigationEligible).toBe(false);
    expect(eligibility.capabilityRankingEligible).toBe(true);
    expect(eligibility.validationEligible).toBe(true);
    expect(eligibility.queueEligible).toBe(true);
    expect(eligibility.queueReason).toBe(null);
  });

  test('capability-ranked rows stop requeueing rankers', () => {
    const eligibility = candidateEligibility({
      tier: 'capability_ranking',
      investigation_passed: 1,
      investigation_loss_ratio: 0.38,
      binding_intermediate_auc: 0.72,
    });
    expect(eligibility.capabilityRankingEligible).toBe(false);
    expect(eligibility.validationEligible).toBe(true);
    expect(eligibility.queueReason).toBe(null);
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
