import { SCORE_MAX, scoreScaleDomain, scoreScalePercent } from './format';

describe('score scale utilities', () => {
  test('uses the v12 scorer ceiling for empty and clamped chart domains', () => {
    expect(SCORE_MAX).toBe(850);
    expect(scoreScaleDomain([])).toEqual({ min: 0, max: 850 });
    expect(scoreScaleDomain([300, 849, 900]).max).toBe(850);
  });

  test('falls back to the v12 scorer ceiling when no explicit chart domain exists', () => {
    expect(scoreScalePercent(425, null, 0)).toBe(50);
  });
});
