export const DEFAULT_CONFIG = {
  n_programs: 50,
  model_dim: 256,
  n_layers: 4,
  vocab_size: 32000,
  max_seq_len: 256,
  device: 'cuda',
  stage1_steps: 500,
  stage1_lr: 0.0003,
  max_depth: 10,
  max_ops: 16,
  math_space_weight: 2.0,
  residual_prob: 0.7,
  max_experiments: 100,
  max_time_minutes: 0,
  max_cost_dollars: 0,
  // Evolution/novelty
  population_size: 50,
  n_generations: 20,
  tournament_size: 5,
  mutation_rate: 0.7,
  crossover_rate: 0.3,
  elitism: 5,
  novelty_weight: 0.5,
  fitness_weight: 0.5,
  archive_size: 200,
  k_nearest: 15,
  archive_threshold: 0.3,
  // Automation
  auto_scale_up: true,
  auto_scale_up_min_survivors: 3,
  auto_scale_up_top_n: 5,
  auto_report: true,
  auto_report_every_n: 5,
  // Model source
  model_source: 'mixed',
  morph_ratio: 0.5,
  // Grammar probabilities
  grammar_split_prob: 0.2,
  grammar_merge_prob: 0.1,
  grammar_risky_op_prob: 0.05,
  grammar_freq_domain_prob: 0.05,
  structured_sparsity_bias: 0.0,
  // Category weights (higher = more likely to be sampled)
  category_weights: {
    elementwise_unary: 1.0,
    elementwise_binary: 1.0,
    reduction: 1.0,
    linear_algebra: 1.0,
    structural: 1.0,
    parameterized: 1.0,
    mixing: 1.0,
    sequence: 1.0,
    frequency: 1.0,
    math_space: 1.0,
    functional: 1.0,
  },
  // Op control
  excluded_ops: '',
  op_weights: '',
  // Training programs
  use_synthesized_training: false,
  n_training_programs: 3,
  // Auto-escalation
  auto_investigate: true,
  auto_investigate_min_survivors: 1,
  auto_investigate_top_n: 5,
  auto_validate: true,
  auto_validate_min_robustness: 0.5,
  auto_validate_top_n: 3,
  // Investigation/validation
  investigation_steps: 2500,
  investigation_batch_size: 4,
  validation_steps: 10000,
  validation_batch_size: 8,
  validation_seq_len: 512,
  validation_n_seeds: 3,
};

const CANARY_PREFS_STORAGE_KEY = 'aria.controlpanel.canaryPrefs.v1';

export const readCanaryPrefs = () => {
  try {
    if (typeof window === 'undefined' || !window.localStorage) return {};
    const raw = window.localStorage.getItem(CANARY_PREFS_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
};

export const writeCanaryPrefs = (prefs) => {
  try {
    if (typeof window === 'undefined' || !window.localStorage) return;
    window.localStorage.setItem(CANARY_PREFS_STORAGE_KEY, JSON.stringify(prefs || {}));
  } catch {
    // Ignore persistence failures
  }
};
