import { canonicalScoreComponents, evalMetricQuality } from './backendScore';

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
      expect.objectContaining({ key: 'perf_short', label: 'SCRN Loss', color: 'var(--accent-blue)' }),
    ]);
  });

  test('drops penalties and non-positive values', () => {
    const components = canonicalScoreComponents({
      score_breakdown: {
        novelty: 5,
        binding_penalty: 7,
        _v9_v8_1_raw: 100,
        robustness: 0,
      },
    });

    expect(components).toEqual([
      expect.objectContaining({ key: 'novelty' }),
    ]);
  });

  test('labels v10 capability and aux trajectory keys', () => {
    const components = canonicalScoreComponents({
      score_breakdown: {
        cap_ar: 25,
        cap_induction: 18,
        cap_erf_density: 22,
        cap_id_collapse: 24,
        aux_erf_variance: 8,
      },
    });
    const byKey = Object.fromEntries(components.map(c => [c.key, c]));
    expect(byKey.cap_ar.label).toBe('AR Gate Gate');
    expect(byKey.cap_induction.label).toBe('Induction Probe');
    expect(byKey.cap_erf_density.label).toBe('ERF Density');
    expect(byKey.cap_id_collapse.label).toBe('ID Collapse');
    expect(byKey.aux_erf_variance.label).toBe('ERF Variance');
  });

  test('splits v10 base total into loss, understanding, and other base totals', () => {
    const components = canonicalScoreComponents({
      score_breakdown: {
        _v10_aux_trajectory_total: 13.2,
        _v10_base_v8style_total: 106.0,
        _v10_capability_total: 100.6,
        _v14_language_control_total: 31.4,
        aux_erf_variance: 9.9,
        aux_icld: 3.3,
        cap_binding: 25,
        cap_induction: 25,
        perf_short: 20.5,
        learning_efficiency: 7.0,
        tinystories: 21.5,
        diagnostic: 10.0,
      },
    });

    expect(components).toEqual([
      expect.objectContaining({ key: '_loss_tier_total', label: 'Loss Total', weight: 27.5 }),
      expect.objectContaining({ key: '_understanding_tier_total', label: 'Understanding Total', weight: 31.5 }),
      expect.objectContaining({ key: '_other_base_total', label: 'Other Base', weight: 47.0 }),
      expect.objectContaining({ key: '_v10_capability_total', label: 'Capability Total', weight: 100.6 }),
      expect.objectContaining({ key: '_v10_aux_trajectory_total', label: 'Aux Trajectory Total', weight: 13.2 }),
      expect.objectContaining({ key: '_v14_language_control_total', label: 'Language Control', weight: 31.4 }),
    ]);
  });

  test('surfaces validation AR Validation as its own additive score component', () => {
    const components = canonicalScoreComponents({
      score_breakdown: {
        _v10_base_v8style_total: 100,
        _v10_capability_total: 80,
        cap_ar_validation_validation: 42,
      },
    });

    expect(components).toEqual([
      expect.objectContaining({ key: '_other_base_total', label: 'Other Base', weight: 100 }),
      expect.objectContaining({ key: '_v10_capability_total', label: 'Capability Total', weight: 80 }),
      expect.objectContaining({ key: 'cap_ar_validation_validation', label: 'AR VAL', weight: 42 }),
    ]);
  });
});

describe('evalMetricQuality', () => {
  test('marks complete post-BPE eval rows trusted', () => {
    expect(evalMetricQuality({
      screening_wikitext_metric_version: 'bpe_eval_v1',
      wikitext_perplexity: 12,
      tinystories_perplexity: 20,
      hellaswag_acc: 0.27,
      blimp_overall_accuracy: 0.52,
    })).toEqual(expect.objectContaining({ key: 'trusted_bpe', reliability: 'high' }));
  });

  test('marks post-BPE rows with missing evals partial', () => {
    const quality = evalMetricQuality({
      screening_wikitext_metric_version: 'bpe_eval_v1',
      wikitext_perplexity: 12,
      hellaswag_acc: 0.27,
    });
    expect(quality.key).toBe('partial_bpe');
    expect(quality.missing).toEqual(['TinyStories', 'BLiMP']);
  });

  test('keeps byte-era or unversioned eval rows out of the trusted bucket', () => {
    expect(evalMetricQuality({
      screening_wikitext_metric_version: 'screening_wikitext_v1',
      wikitext_perplexity: 12,
      hellaswag_acc: 0.27,
    })).toEqual(expect.objectContaining({ key: 'legacy_eval', reliability: 'low' }));
  });

  test('marks explicitly failed BPE rescoring rows failed even with partial metrics', () => {
    expect(evalMetricQuality({
      tags: 'bpe_eval_failed,manual_quarantine_20260426',
      wikitext_perplexity: 851.82,
      hellaswag_acc: 0.28,
    })).toEqual(expect.objectContaining({ key: 'failed_eval', reliability: 'low' }));
  });
});
