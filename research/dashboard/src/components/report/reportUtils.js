export const COMPRESSION_FACTORS = {
  low_rank: 0.55,
  shared_basis: 0.5,
  hash_trick: 0.35,
  structured_sparse: 0.4,
  kronecker: 0.5,
  polynomial: 0.6,
  residual_quantized: 0.3,
  compressed_attention: 0.7,
};

export function parseArchSpec(value) {
  if (!value || typeof value !== 'string') return null;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

export function resolveLossRatio(program) {
  if (!program) return null;
  const val = program.validation_loss_ratio;
  if (val != null && Number.isFinite(Number(val))) return Number(val);
  const lr = program.loss_ratio;
  return (lr != null && Number.isFinite(Number(lr))) ? Number(lr) : null;
}

export function compressionSummary(program) {
  const spec = parseArchSpec(program.arch_spec_json);
  const compressionKey = spec?.choices?.weight_storage || spec?.choices?.token_representation;
  const factor = COMPRESSION_FACTORS[compressionKey] || 1.0;
  const rawParams = program.param_count || program.graph_n_params_estimate || null;
  const compressedParams = rawParams != null ? Math.max(1, Math.round(rawParams * factor)) : null;
  const ratio = rawParams != null && compressedParams != null
    ? Math.max(0.01, Math.min(1.0, compressedParams / rawParams))
    : null;
  const memoryMb = compressedParams != null ? (compressedParams * 4) / (1024 * 1024) : null;
  const primaryLoss = resolveLossRatio(program);
  const qualityRetention = program.baseline_loss_ratio != null
    ? Math.max(0, Math.min(1, 1.25 - program.baseline_loss_ratio))
    : primaryLoss != null
      ? Math.max(0, Math.min(1, 1.0 - primaryLoss))
      : null;
  return {
    label: compressionKey || 'dense',
    ratio,
    memoryMb,
    qualityRetention,
  };
}

export { programMetricChips as metricChips } from '../../utils/metricChips';

const QKV_OPS = new Set(['local_window_attn', 'sliding_window_mask', 'multi_head_mix']);

export const TOKEN_MIXING_FAMILIES = {
  local_window_attn: 'attention',
  sliding_window_mask: 'attention',
  softmax_last: 'attention',
  multi_head_mix: 'attention',
  selective_scan: 'ssm',
  conv1d_seq: 'conv',
  rfft_seq: 'frequency',
  irfft_seq: 'frequency',
  argsort_seq: 'sorting',
  token_pool_restore: 'pooling',
  cumsum_seq: 'pooling',
  roll_seq: 'pooling',
  basis_expansion: 'functional',
  integral_kernel: 'functional',
  fixed_point_iter: 'functional',
};

export const FAMILY_LABELS = {
  attention: 'QKV-based',
  ssm: 'State Space',
  conv: 'Convolution',
  frequency: 'Frequency Domain',
  sorting: 'Sort-based',
  pooling: 'Pooling',
  functional: 'Functional/Operator',
};

export const FAMILY_COLORS = {
  attention: 'var(--accent-blue)',
  ssm: 'var(--accent-green)',
  conv: 'var(--accent-yellow)',
  frequency: 'var(--accent-purple)',
  sorting: 'var(--accent-red)',
  pooling: 'var(--text-muted)',
  functional: '#e0a060',
};

export function classifyTokenMixing(program) {
  const raw = program.graph_json || program._graph_json;
  if (!raw) return { families: new Set(), qkvFree: null, ops: [] };
  try {
    const graph = typeof raw === 'string' ? JSON.parse(raw) : raw;
    const nodes = graph.nodes || {};
    const ops = Object.values(nodes).map(n => n.op_name || n.op).filter(Boolean);
    const families = new Set();
    const detectedOps = [];
    for (const op of ops) {
      const family = TOKEN_MIXING_FAMILIES[op];
      if (family) {
        families.add(family);
        detectedOps.push(op);
      }
    }
    const qkvFree = !ops.some(op => QKV_OPS.has(op));
    return { families, qkvFree, ops: detectedOps };
  } catch {
    return { families: new Set(), qkvFree: null, ops: [] };
  }
}

// qkvUsageDescriptor consolidated into utils/architecture.js
export { qkvUsageDescriptor } from '../../utils/architecture';

export const WEIGHT_STORAGE_LABELS = {
  dense_matrix: 'Dense (baseline)',
  low_rank: 'Low-Rank (UV)',
  hypernetwork: 'Hypernetwork',
  shared_basis: 'Shared Basis',
  hash_trick: 'Hash Trick',
  kronecker: 'Kronecker',
  polynomial: 'Polynomial',
  structured_sparse: 'Structured Sparse',
};

export const TOKEN_REP_LABELS = {
  standard_float: 'Standard Float',
  binary_hash: 'Binary Hash',
  residual_quantized: 'Residual Quantized',
  complex_valued: 'Complex',
  quaternion: 'Quaternion',
  multi_resolution: 'Multi-Resolution',
  mixture_embedding: 'Mixture Embedding',
};

export function reliabilityBand(sampleSize) {
  if (sampleSize >= 30) return { label: 'high', color: 'var(--accent-green)' };
  if (sampleSize >= 12) return { label: 'medium', color: 'var(--accent-yellow)' };
  return { label: 'low', color: 'var(--accent-red)' };
}

export function wilsonInterval(successes, total, z = 1.96) {
  if (!Number.isFinite(successes) || !Number.isFinite(total) || total <= 0) {
    return null;
  }
  const p = successes / total;
  const z2 = z * z;
  const denom = 1 + z2 / total;
  const center = p + z2 / (2 * total);
  const margin = z * Math.sqrt((p * (1 - p) + z2 / (4 * total)) / total);
  const low = Math.max(0, (center - margin) / denom);
  const high = Math.min(1, (center + margin) / denom);
  return { low, high };
}

export function reproducibilityPacketStatus(program) {
  const spec = parseArchSpec(program?.arch_spec_json);
  const checks = [
    { label: 'result_id', ok: !!program?.result_id },
    { label: 'graph_fingerprint', ok: !!program?.graph_fingerprint },
    { label: 'arch_spec', ok: !!spec },
    { label: 'loss_ratio', ok: resolveLossRatio(program) != null },
    { label: 'baseline_ratio', ok: program?.baseline_loss_ratio != null },
    { label: 'cka_artifact', ok: program?.cka_source === 'artifact' },
  ];
  const readyCount = checks.filter(check => check.ok).length;
  const totalChecks = checks.length;
  const label = readyCount === totalChecks ? 'Ready' : readyCount >= 4 ? 'Partial' : 'Sparse';
  const color = readyCount === totalChecks
    ? 'var(--accent-green)'
    : readyCount >= 4
      ? 'var(--accent-yellow)'
      : 'var(--accent-red)';
  return { label, color, readyCount, totalChecks };
}

export function decisionGate(program) {
  const checks = {
    screeningEvidence: resolveLossRatio(program) != null && program.novelty_score != null,
    baselineEvidence: program.baseline_loss_ratio != null,
    baselineBeatsReference: program.baseline_loss_ratio != null && program.baseline_loss_ratio < 1.0,
    ckaArtifactBacked: program.cka_source === 'artifact',
  };
  const decisionReady = Object.values(checks).every(Boolean);
  const missing = Object.entries(checks)
    .filter(([, ok]) => !ok)
    .map(([name]) => name);
  return {
    decisionReady,
    label: decisionReady ? 'Decision-Ready' : 'Exploratory',
    color: decisionReady ? 'var(--accent-green)' : 'var(--accent-yellow)',
    missing,
  };
}

export const DISC_COLUMNS = [
  { key: '_score', label: 'Score' },
  { key: 'graph_fingerprint', label: 'Fingerprint' },
  { key: 'repeat_count', label: 'Repeats' },
  { key: 'loss_ratio', label: 'Loss Ratio (val)' },
  { key: 'novelty_score', label: 'Novelty' },
  { key: 'baseline_loss_ratio', label: 'Baseline' },
  { key: '_compressionRatio', label: 'Compression' },
  { key: '_metricQualityOrder', label: 'Metric Quality' },
  { key: 'cka_source', label: 'CKA Source' },
  { key: 'most_similar_to', label: 'Similar To' },
  { key: '_decisionGateOrder', label: 'Decision Gate' },
  { key: 'rating', label: 'Rating' },
];

export const DISC_RATING_ORDER = { 'S1 - Exceptional': 4, 'S1 - Strong': 3, 'S1 - Moderate': 2, 'S1 - Marginal': 1 };

export const REPORT_DISCOVERY_SORT_PREFS_KEY = 'dashboard.report.discovery-rankings.sort.v1';
export const REPORT_DISCOVERY_VIEW_PREFS_KEY = 'dashboard.report.discovery-rankings.view.v1';

export function reportQueueReasonLabel(reason) {
  if (reason === 'already_investigated_unchanged') return 'Already investigated (unchanged).';
  if (reason === 'not_investigation_passed') return 'Investigation did not pass robustness gate.';
  if (reason === 'already_validated') return 'Already validated.';
  if (reason === 'not_investigation_tier') return 'Validation requires investigation tier.';
  if (reason === 'not_screening_tier') return 'Investigation requires screening tier.';
  if (reason === 'not_stage1_survivor') return 'Candidate is not a Stage-1 survivor.';
  if (reason === 'not_in_leaderboard') return 'Candidate is not in progression leaderboard yet.';
  if (reason === 'result_not_found') return 'Result ID not found.';
  return 'Candidate is not currently eligible for progression actions.';
}
