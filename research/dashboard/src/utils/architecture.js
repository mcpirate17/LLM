/**
 * Shared architecture analysis utilities for dashboard components.
 */

const QKV_OPS = new Set(['local_window_attn', 'sliding_window_mask', 'multi_head_mix']);

/** Detect whether an entry uses QKV-based attention from its graph_json. Returns true/false/null. */
export function detectQkvFree(entry) {
  const raw = entry._graph_json || entry.graph_json;
  if (!raw) return null;
  try {
    const graph = typeof raw === 'string' ? JSON.parse(raw) : raw;
    const nodes = graph.nodes || {};
    const ops = Object.values(nodes).map(n => n.op_name || n.op).filter(Boolean);
    return !ops.some(op => QKV_OPS.has(op));
  } catch {
    return null;
  }
}

/** Classify QKV usage for display. Returns { label, detail, tone }. */
export function qkvUsageDescriptor(entry) {
  const usage = entry?.qkv_usage;
  if (usage === 'qkv_free') {
    return {
      label: 'QKV-free',
      detail: 'Non-attention token mixing path (SSM/conv/frequency/functional).',
      tone: 'high',
    };
  }
  if (usage === 'q_eq_k_eq_v') {
    return {
      label: 'Q=K=V',
      detail: 'Shared-projection attention variant (reduced attention parameterization).',
      tone: 'medium',
    };
  }
  if (usage === 'full_qkv') {
    return {
      label: 'Full QKV',
      detail: 'Standard Q/K/V attention primitives are present.',
      tone: 'medium',
    };
  }
  const qkvFree = detectQkvFree(entry);
  if (qkvFree === true) {
    return {
      label: 'QKV-free*',
      detail: 'Inferred from graph ops when qkv_usage enum is unavailable.',
      tone: 'high',
    };
  }
  if (qkvFree === false) {
    return {
      label: 'Uses QKV*',
      detail: 'Inferred from graph ops when qkv_usage enum is unavailable.',
      tone: 'medium',
    };
  }
  return {
    label: 'QKV unknown',
    detail: 'Insufficient graph/payload info to classify QKV usage.',
    tone: 'low',
  };
}
