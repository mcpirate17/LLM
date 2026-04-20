import { DISCOVERY_TIER_FILTERS, matchesActiveTier } from '../utils/discoveryStatus';

describe('matchesActiveTier', () => {
  test('allows all tiers when active tier is all', () => {
    expect(matchesActiveTier({ tier: 'screening' }, 'all')).toBe(true);
    expect(matchesActiveTier({ tier: 'validation' }, 'all')).toBe(true);
  });

  test('matches tier case-insensitively', () => {
    expect(matchesActiveTier({ tier: 'investigation' }, 'Investigation')).toBe(true);
    expect(matchesActiveTier({ tier: 'SCREENING' }, 'screening')).toBe(true);
  });

  test('rejects rows from other tiers', () => {
    expect(matchesActiveTier({ tier: 'screening' }, 'investigation')).toBe(false);
    expect(matchesActiveTier({ tier: 'validation' }, 'screening')).toBe(false);
  });

  test('rejects missing tier when a specific tier is selected', () => {
    expect(matchesActiveTier({}, 'investigation')).toBe(false);
  });

  test('treats uncompleted validation rows as validation pending, not validated', () => {
    expect(matchesActiveTier({ tier: 'validation', validation_passed: 0 }, 'validation')).toBe(false);
    expect(matchesActiveTier({ tier: 'validation', validation_passed: 0 }, 'validation_pending')).toBe(true);
  });

  test('exposes validation pending as a first-class tier filter', () => {
    expect(DISCOVERY_TIER_FILTERS).toContain('validation_pending');
  });
});
