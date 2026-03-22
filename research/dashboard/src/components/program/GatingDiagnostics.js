import React from 'react';
import RoutingHeatmap from './RoutingHeatmap';

const GATING_OPS = new Set(['topk_gate']);
const ACTIVATION_GATING_OPS = new Set(['sigmoid', 'silu', 'gelu']);

function detectGatingOps(graphJson) {
  if (!graphJson) return { hasGating: false, gatingOps: [], activationGates: [] };
  const nodes = graphJson.nodes || {};
  const nodeList = Array.isArray(nodes) ? nodes : Object.values(nodes);
  const gatingOps = [];
  const activationGates = [];
  const seen = new Set();
  for (const node of nodeList) {
    if (!node || typeof node !== 'object') continue;
    const op = node.op_name || node.op;
    if (!op || seen.has(op)) continue;
    seen.add(op);
    if (GATING_OPS.has(op)) gatingOps.push(op);
    if (ACTIVATION_GATING_OPS.has(op)) activationGates.push(op);
  }
  return { hasGating: gatingOps.length > 0, gatingOps, activationGates };
}

function entropyInterpretation(entropy, nExperts) {
  if (entropy == null || nExperts == null || nExperts < 2) return null;
  const maxEntropy = Math.log2(nExperts);
  const ratio = maxEntropy > 0 ? entropy / maxEntropy : 0;
  if (ratio > 0.85) return { label: 'Well-distributed', color: 'var(--accent-green)', risk: 'low' };
  if (ratio > 0.5) return { label: 'Moderate skew', color: 'var(--accent-yellow)', risk: 'medium' };
  return { label: 'Route collapse risk', color: 'var(--accent-red)', risk: 'high' };
}

export function GatingDiagnostics({ program }) {
  const graphJson = program.graph_json_parsed;
  const { hasGating, gatingOps, activationGates } = detectGatingOps(graphJson);

  const hasRouting = program.routing_drop_rate != null ||
    program.routing_utilization_entropy != null ||
    program.routing_confidence_mean != null;

  if (!hasGating && !hasRouting) return null;

  let expertUtil = null;
  let nExperts = null;
  if (program.routing_expert_utilization_json) {
    try {
      const parsed = typeof program.routing_expert_utilization_json === 'string'
        ? JSON.parse(program.routing_expert_utilization_json)
        : program.routing_expert_utilization_json;

      let normalized = null;
      if (Array.isArray(parsed)) {
        normalized = parsed;
      } else if (parsed && typeof parsed === 'object') {
        normalized = Object.values(parsed);
      }

      if (Array.isArray(normalized)) {
        expertUtil = normalized
          .map(v => Number(v))
          .filter(v => Number.isFinite(v));
        nExperts = expertUtil.length;
      }
    } catch { /* ignore */ }
  }

  const entropy = program.routing_utilization_entropy;
  const interpretation = entropyInterpretation(entropy, nExperts || 2);
  const dropRate = program.routing_drop_rate;
  const confMean = program.routing_confidence_mean;
  const confStd = program.routing_confidence_std;
  const tokProcessed = program.routing_tokens_processed;
  const tokSkipped = program.routing_tokens_skipped;
  const overflows = program.routing_capacity_overflow_count;

  const maxUtil = Array.isArray(expertUtil) && expertUtil.length > 0
    ? Math.max(...expertUtil)
    : 0;

  const tokenRetention = (tokProcessed != null && (tokProcessed + (tokSkipped || 0)) > 0)
    ? Number(tokProcessed) / Number(tokProcessed + (tokSkipped || 0))
    : (dropRate != null ? Math.max(0, Math.min(1, 1 - Number(dropRate))) : null);

  const tokenRetentionCurve = (() => {
    if (program.routing_expert_utilization_json) {
      try {
        const parsed = typeof program.routing_expert_utilization_json === 'string'
          ? JSON.parse(program.routing_expert_utilization_json)
          : program.routing_expert_utilization_json;
        if (Array.isArray(parsed) && parsed.length > 0 && typeof parsed[0] === 'object') {
          const points = parsed
            .map((point, idx) => ({
              step: Number(point.step ?? idx),
              retention: Number(point.retention ?? point.token_retention ?? point.value),
            }))
            .filter(point => Number.isFinite(point.step) && Number.isFinite(point.retention))
            .map(point => ({ step: point.step, retention: Math.max(0, Math.min(1, point.retention)) }));
          if (points.length >= 2) return points;
        }
      } catch {
        // ignore malformed payloads
      }
    }
    if (tokenRetention != null) {
      return [
        { step: 0, retention: 1.0 },
        { step: 1, retention: tokenRetention },
      ];
    }
    return [];
  })();

  const tokenCurvePath = (() => {
    if (!tokenRetentionCurve || tokenRetentionCurve.length < 2) return null;
    const maxStep = Math.max(...tokenRetentionCurve.map(point => point.step), 1);
    return tokenRetentionCurve
      .map((point, idx) => {
        const x = (point.step / maxStep) * 100;
        const y = (1 - point.retention) * 100;
        return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(' ');
  })();

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
        Gating & Routing Diagnostics
      </div>
      <div style={{
        padding: '10px 12px', background: 'var(--bg-tertiary)', borderRadius: 6,
        borderLeft: `3px solid ${interpretation ? interpretation.color : 'var(--accent-blue)'}`,
        display: 'flex', flexDirection: 'column', gap: 10,
      }}>
        {/* Detected ops */}
        {(gatingOps.length > 0 || activationGates.length > 0) && (
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600 }}>Gating ops:</span>
            {gatingOps.map(op => (
              <code key={op} style={{
                fontSize: 10, padding: '1px 6px', borderRadius: 3,
                background: 'rgba(0, 212, 255, 0.15)', color: 'var(--accent-purple)',
              }}>{op}</code>
            ))}
            {activationGates.map(op => (
              <code key={op} style={{
                fontSize: 10, padding: '1px 6px', borderRadius: 3,
                background: 'var(--bg-secondary)', color: 'var(--text-muted)',
              }}>{op}</code>
            ))}
          </div>
        )}

        {/* Routing mode */}
        {program.routing_mode && (
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            <span style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Mode:</span>{' '}
            <span style={{ color: 'var(--accent-blue)' }}>{program.routing_mode}</span>
          </div>
        )}

        {/* Key metrics row */}
        {hasRouting && (
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12 }}>
            {dropRate != null && (
              <div>
                <div style={{ color: 'var(--text-muted)', fontSize: 10, fontWeight: 600, marginBottom: 2 }}>Drop Rate</div>
                <span style={{
                  fontWeight: 600, fontSize: 14,
                  color: dropRate > 0.3 ? 'var(--accent-red)' : dropRate > 0.1 ? 'var(--accent-yellow)' : 'var(--accent-green)',
                }}>
                  {(dropRate * 100).toFixed(1)}%
                </span>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                  {dropRate > 0.3 ? 'High token loss' : dropRate > 0.1 ? 'Moderate' : 'Healthy'}
                </div>
              </div>
            )}
            {entropy != null && (
              <div>
                <div style={{ color: 'var(--text-muted)', fontSize: 10, fontWeight: 600, marginBottom: 2 }}>Utilization Entropy</div>
                <span style={{ fontWeight: 600, fontSize: 14, color: interpretation?.color || 'var(--text-secondary)' }}>
                  {Number(entropy).toFixed(3)}
                </span>
                {interpretation && (
                  <div style={{ fontSize: 10, color: interpretation.color }}>{interpretation.label}</div>
                )}
                {interpretation && (
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    Collapse risk: <strong style={{ color: interpretation.color }}>{interpretation.risk}</strong>
                  </div>
                )}
              </div>
            )}
            {confMean != null && (
              <div>
                <div style={{ color: 'var(--text-muted)', fontSize: 10, fontWeight: 600, marginBottom: 2 }}>Gate Confidence</div>
                <span style={{
                  fontWeight: 600, fontSize: 14,
                  color: confMean > 0.8 ? 'var(--accent-green)' : confMean > 0.5 ? 'var(--accent-yellow)' : 'var(--accent-red)',
                }}>
                  {Number(confMean).toFixed(3)}
                </span>
                {confStd != null && (
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 4 }}>
                    ±{Number(confStd).toFixed(3)}
                  </span>
                )}
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                  {confMean > 0.8 ? 'Decisive' : confMean > 0.5 ? 'Moderate' : 'Uncertain routing'}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Token-retention curve */}
        {tokenCurvePath && (
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>
              Token Retention Curve
            </div>
            <div style={{ border: '1px solid var(--border)', borderRadius: 4, background: 'var(--bg-secondary)', padding: '4px 6px' }}>
              <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: 54, display: 'block' }}>
                <path d={tokenCurvePath} fill="none" stroke="var(--accent-blue)" strokeWidth="2" />
              </svg>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
              Final retention: {tokenRetention != null ? `${(tokenRetention * 100).toFixed(1)}%` : 'not measured'}
            </div>
          </div>
        )}

        {/* Token flow */}
        {(tokProcessed != null || tokSkipped != null || overflows != null) && (
          <div style={{ display: 'flex', gap: 12, fontSize: 11, color: 'var(--text-secondary)', flexWrap: 'wrap' }}>
            {tokProcessed != null && <span>Processed: <strong>{Number(tokProcessed).toLocaleString()}</strong></span>}
            {tokSkipped != null && (
              <span>Skipped: <strong style={{ color: tokSkipped > 0 ? 'var(--accent-yellow)' : 'var(--text-secondary)' }}>
                {Number(tokSkipped).toLocaleString()}
              </strong></span>
            )}
            {overflows != null && overflows > 0 && (
              <span>Capacity overflows: <strong style={{ color: 'var(--accent-red)' }}>
                {Number(overflows).toLocaleString()}
              </strong></span>
            )}
          </div>
        )}

        {/* Expert utilization bars */}
        {expertUtil && expertUtil.length > 0 && (
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>
              Expert Utilization ({expertUtil.length} experts)
            </div>
            <div style={{ display: 'flex', gap: 2, alignItems: 'flex-end', height: 40 }}>
              {expertUtil.map((val, i) => {
                const v = Number(val);
                const pct = maxUtil > 0 ? (v / maxUtil) * 100 : 0;
                const isCollapsed = expertUtil.length > 2 && v < (maxUtil * 0.1);
                return (
                  <div
                    key={i}
                    title={`Expert ${i}: ${(v * 100).toFixed(1)}%`}
                    style={{
                      flex: 1,
                      height: `${Math.max(pct, 4)}%`,
                      background: isCollapsed ? 'var(--accent-red)' : 'var(--accent-blue)',
                      borderRadius: '2px 2px 0 0',
                      opacity: isCollapsed ? 0.7 : 0.6,
                      minWidth: 4,
                      maxWidth: 24,
                    }}
                  />
                );
              })}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
              <span>E0</span>
              <span>E{expertUtil.length - 1}</span>
            </div>
          </div>
        )}

        {/* Routing Heatmaps (from sparsity report) */}
        {program.sparsity_report_json_parsed?.routing_heatmaps && (
          <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12, marginTop: 4 }}>
            {Object.entries(program.sparsity_report_json_parsed.routing_heatmaps).map(([name, data]) => (
              <RoutingHeatmap key={name} data={data} nExperts={expertUtil?.length || 4} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default React.memo(GatingDiagnostics);
