import { canonicalScoreComponents } from './backendScore';

describe('canonicalScoreComponents', () => {
  test('maps canonical breakdown keys to labeled colored components', () => {
    const components = canonicalScoreComponents({
      score_breakdown: {
        binding: 85,
        early_convergence: 4.1,
        learning_efficiency: 11.2,
        perf_short: 21.2,
      },
    });

    expect(components).toEqual([
      expect.objectContaining({ key: 'binding', label: 'Binding Range', color: '#a371f7' }),
      expect.objectContaining({ key: 'early_convergence', label: 'Early Convergence', color: '#f0883e' }),
      expect.objectContaining({ key: 'learning_efficiency', label: 'Learning Efficiency', color: '#db61a2' }),
      expect.objectContaining({ key: 'perf_short', label: 'Screening Loss', color: 'var(--accent-blue)' }),
    ]);
  });

  test('drops penalties and non-positive values', () => {
    const components = canonicalScoreComponents({
      score_breakdown: {
        novelty: 5,
        binding_penalty: 7,
        robustness: 0,
      },
    });

    expect(components).toEqual([
      expect.objectContaining({ key: 'novelty' }),
    ]);
  });
});
