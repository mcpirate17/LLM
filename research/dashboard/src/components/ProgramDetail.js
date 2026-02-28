import { apiCall } from "../services/apiService";
import React, { useState, useEffect, useRef } from 'react';
import { lossColor, noveltyColor } from '../utils/colors';
import useCopyToClipboard from '../hooks/useCopyToClipboard';
import apiService from '../services/apiService';
import { CHART_DEFAULTS, clampToScale, getFixedScale } from '../utils/chartScales';


/**
 * ProgramDetail — Modal showing computation graph, stage pipeline,
 * fingerprint radar chart, training metrics, similar architectures,
 * sandbox metrics, FLOPs, baseline comparison, training curve.
 */

function StagePipeline({ program }) {
  const stages = [
    { key: 'stage0_passed', label: 'Stage 0', sublabel: 'Compilation' },
    { key: 'stage05_passed', label: 'Stage 0.5', sublabel: 'Stability' },
    { key: 'stage1_passed', label: 'Stage 1', sublabel: 'Learning' },
  ];

  return (
    <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
      {stages.map((stage, i) => {
        const passed = program[stage.key];
        const color = passed ? 'var(--accent-green)' : 'var(--accent-red)';
        const bg = passed ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)';
        return (
          <React.Fragment key={stage.key}>
            {i > 0 && <span style={{ color: 'var(--text-muted)', fontSize: 16 }}>&rarr;</span>}
            <div style={{
              padding: '6px 12px',
              background: bg,
              border: `1px solid ${color}`,
              borderRadius: 6,
              textAlign: 'center',
              minWidth: 80,
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color }}>{passed ? 'PASS' : 'FAIL'}</div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{stage.label}</div>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}

function RadarChart({ program, size = 240 }) {
  // Extended radar with fingerprint metrics
  const axes = [
    { key: 'novelty_score', label: 'Novelty' },
    { key: 'structural_novelty', label: 'Structural' },
    { key: 'behavioral_novelty', label: 'Behavioral' },
    { key: 'fp_interaction_locality', label: 'Locality' },
    { key: 'fp_interaction_sparsity', label: 'Sparsity' },
    { key: 'fp_isotropy', label: 'Isotropy' },
    { key: 'fp_rank_ratio', label: 'Rank' },
    { key: 'fp_sensitivity_uniformity', label: 'Sensitivity' },
  ].filter(a => program[a.key] !== null && program[a.key] !== undefined);

  // Fall back to minimal if no fingerprint data
  if (axes.length < 3) {
    const fallback = [
      { key: 'novelty_score', label: 'Novelty' },
      { key: 'structural_novelty', label: 'Structural' },
      { key: 'behavioral_novelty', label: 'Behavioral' },
    ];
    axes.length = 0;
    axes.push(...fallback);
  }

  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 30;
  const n = axes.length;

  const getPoint = (i, val) => {
    const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
    const d = val * r;
    return { x: cx + d * Math.cos(angle), y: cy + d * Math.sin(angle) };
  };

  const rings = [0.25, 0.5, 0.75, 1.0];
  const values = axes.map(a => Math.min(program[a.key] || 0, 1));
  const points = values.map((v, i) => getPoint(i, v));
  const polygonPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z';

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {rings.map(ring => {
        const ringPoints = axes.map((_, i) => getPoint(i, ring));
        const d = ringPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z';
        return <path key={ring} d={d} fill="none" stroke="var(--border, #30363d)" strokeWidth={0.5} />;
      })}
      {axes.map((_, i) => {
        const end = getPoint(i, 1);
        return <line key={i} x1={cx} y1={cy} x2={end.x} y2={end.y}
          stroke="var(--border, #30363d)" strokeWidth={0.5} />;
      })}
      <path d={polygonPath} fill="rgba(188, 140, 255, 0.2)" stroke="var(--accent-purple, #bc8cff)" strokeWidth={2} />
      {points.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={3}
          fill="var(--accent-purple, #bc8cff)" stroke="var(--bg-secondary, #161b22)" strokeWidth={1.5} />
      ))}
      {axes.map((axis, i) => {
        const labelPt = getPoint(i, 1.25);
        return (
          <text key={i} x={labelPt.x} y={labelPt.y}
            textAnchor="middle" dominantBaseline="middle"
            fill="var(--text-secondary, #8b949e)" fontSize={9}>
            {axis.label}
          </text>
        );
      })}
    </svg>
  );
}

function RoutingHeatmap({ data, nExperts }) {
  if (!data || !Array.isArray(data)) return null;
  // data is [seq_len] or [seq_len, top_k]
  const seqLen = data.length;
  const experts = nExperts || 4;
  
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>
        Routing Heatmap (Token x Expert)
      </div>
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: `repeat(${experts}, 1fr)`,
        gap: 1,
        background: 'var(--border)',
        border: '1px solid var(--border)',
        padding: 1
      }}>
        {data.map((selected, t) => {
          const selectedList = Array.isArray(selected) ? selected : [selected];
          return Array.from({ length: experts }).map((_, e) => {
            const isActive = selectedList.includes(e);
            return (
              <div 
                key={`${t}-${e}`}
                title={`Token ${t}, Expert ${e}: ${isActive ? 'Active' : 'Inactive'}`}
                style={{
                  height: Math.max(2, 80 / seqLen),
                  background: isActive ? 'var(--accent-blue)' : 'var(--bg-secondary)',
                  opacity: isActive ? 0.8 : 1
                }}
              />
            );
          });
        })}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
        <span>T0 (start)</span>
        <span>T{seqLen - 1} (end)</span>
      </div>
    </div>
  );
}

function SparsityDiagnostics({ program }) {
  const report = program.sparsity_report_json_parsed;
  const ratio = program.sparsity_ratio;
  const deadCount = program.dead_neuron_count;

  if (!report && ratio == null) return null;

  return (
    <div className="card" style={{ padding: 12, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <div className="card-title" style={{ margin: 0 }}>Activation Sparsity</div>
        {report?.max_layer_collapse > 0.9 && (
          <span className="badge badge-error">COLLAPSED</span>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Overall Sparsity</div>
          <div style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)' }}>
            {ratio != null ? `${(ratio * 100).toFixed(1)}%` : 'N/A'}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Dead Neurons</div>
          <div style={{ fontSize: 18, fontWeight: 600, color: deadCount > 0 ? 'var(--accent-yellow)' : 'var(--text-primary)' }}>
            {deadCount != null ? deadCount.toLocaleString() : 'N/A'}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Max Collapse</div>
          <div style={{ fontSize: 18, fontWeight: 600, color: (report?.max_layer_collapse > 0.5) ? 'var(--accent-red)' : 'var(--text-primary)' }}>
            {report?.max_layer_collapse != null ? `${(report.max_layer_collapse * 100).toFixed(1)}%` : 'N/A'}
          </div>
        </div>
      </div>

      {report?.dead_neuron_ratio > 0.1 && (
        <div style={{ fontSize: 11, color: 'var(--text-secondary)', background: 'rgba(210, 153, 34, 0.1)', padding: '6px 10px', borderRadius: 4, border: '1px solid rgba(210, 153, 34, 0.2)' }}>
          <strong>Aria Note:</strong> High dead neuron ratio ({ (report.dead_neuron_ratio * 100).toFixed(1) }%) suggests architectural bottlenecking or poor initialization.
        </div>
      )}
    </div>
  );
}

function TrainingCurve({ resultId }) {
  const [curve, setCurve] = useState(null);
  const MAX_POINTS = 500; // Cap points to prevent memory leaks

  useEffect(() => {
    apiService.getTrainingCurve(resultId)
      .then(d => {
        if (Array.isArray(d)) {
          // If the data is massive, downsample or take recent window
          const recentData = d.slice(-MAX_POINTS);
          setCurve(recentData);
        }
      })
      .catch(() => {});
  }, [resultId]);

  if (!curve || curve.length === 0) return null;

  const W = 350, H = 120;
  const pad = { l: 45, r: 10, t: 10, b: 25 };

  const losses = curve.map(c => c.loss).filter(l => l != null && isFinite(l));
  if (losses.length < 2) return null;

  const lossDefaults = CHART_DEFAULTS.training_loss;
  const lossScale = getFixedScale('training.loss', losses, {
    defaultMin: lossDefaults.min,
    defaultMax: lossDefaults.max,
  });
  const minL = lossScale.min;
  const maxL = lossScale.max;
  const rangeL = maxL - minL || 1;

  const denom = Math.max(1, MAX_POINTS - 1);
  const xScale = i => pad.l + (i / denom) * (W - pad.l - pad.r);
  const yScale = v => H - pad.b - ((clampToScale(v, lossScale) - minL) / rangeL) * (H - pad.t - pad.b);

  const pathD = losses.map((l, i) => `${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(l)}`).join(' ');

  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
        Training Curve
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="var(--border)" />
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="var(--border)" />
        <text x={pad.l - 5} y={yScale(maxL)} textAnchor="end" fill="var(--text-muted)" fontSize={9}>{maxL.toFixed(2)}</text>
        <text x={pad.l - 5} y={yScale(minL)} textAnchor="end" fill="var(--text-muted)" fontSize={9}>{minL.toFixed(2)}</text>
        <text x={W / 2} y={H - 3} textAnchor="middle" fill="var(--text-muted)" fontSize={9}>Step</text>
        <path d={pathD} fill="none" stroke="var(--accent-green)" strokeWidth={1.5} />
      </svg>
    </div>
  );
}

function MetricRow({ label, value }) {
  if (value === null || value === undefined || value === '--') return null;
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: '1px solid var(--border)' }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <span>{value}</span>
    </div>
  );
}

function RobustnessProfile({ program, leaderboardEntry }) {
  const source = leaderboardEntry || program || {};
  const noise = source?.robustness_noise_score;
  const longCtx = source?.robustness_long_ctx_score;
  const initStd = source?.init_sensitivity_std;
  const quantRetentionRaw = source?.quant_int8_retention;
  const quantRetention = quantRetentionRaw == null
    ? null
    : (Number(quantRetentionRaw) <= 1 ? Number(quantRetentionRaw) * 100 : Number(quantRetentionRaw));
  const qualityPerByte = source?.quant_quality_per_byte;
  const spectralNorm = source?.jacobian_spectral_norm ?? source?.fp_jacobian_spectral_norm ?? source?.fp_spectral_norm ?? source?.spectral_norm ?? null;

  const hasAny = [noise, longCtx, initStd, quantRetention, qualityPerByte, spectralNorm]
    .some(v => v != null && Number.isFinite(Number(v)));
  if (!hasAny) return null;

  const gauge = (value, invert = false) => {
    if (value == null) return null;
    const v = Math.max(0, Math.min(1, Number(value)));
    const pct = (invert ? (1 - v) : v) * 100;
    return `${pct.toFixed(0)}%`;
  };

  return (
    <div style={{
      marginTop: 12,
      padding: 10,
      background: 'var(--bg-tertiary)',
      borderRadius: 6,
      border: '1px solid var(--border)',
    }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 8 }}>
        Robustness Profile
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12 }}>
        <MetricRow label="Noise sensitivity" value={noise != null ? `${Number(noise).toFixed(3)} (${gauge(noise, true)})` : null} />
        <MetricRow label="Long-context score" value={longCtx != null ? `${Number(longCtx).toFixed(3)} (${gauge(longCtx, false)})` : null} />
        <MetricRow label="Init sensitivity std" value={initStd != null ? Number(initStd).toFixed(4) : null} />
        <MetricRow label="INT8 retention" value={quantRetention != null ? `${quantRetention.toFixed(1)}%` : null} />
        <MetricRow label="Quality per byte" value={qualityPerByte != null ? Number(qualityPerByte).toFixed(4) : null} />
        <MetricRow label="Spectral norm" value={spectralNorm != null ? Number(spectralNorm).toFixed(4) : null} />
      </div>
    </div>
  );
}

function AriaAdvice({ analysis }) {
  const advice = analysis?.brittleness_advice;
  if (!advice) return null;

  return (
    <div style={{
      marginTop: 12,
      padding: 12,
      background: 'rgba(188, 140, 255, 0.05)',
      borderRadius: 8,
      border: '1px solid rgba(188, 140, 255, 0.3)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span style={{ fontSize: 16 }}>🧬</span>
        <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--accent-purple)', textTransform: 'uppercase' }}>
          Aria's Advice: Stabilisation
        </div>
      </div>
      
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6, color: 'var(--text-primary)' }}>
        {advice.summary}
      </div>
      
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10, lineHeight: 1.4 }}>
        {advice.diagnosis}
      </div>

      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 4 }}>
          Recommended Improvements
        </div>
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: 'var(--text-primary)', lineHeight: 1.5 }}>
          {advice.remedies.map((r, i) => (
            <li key={i} style={{ marginBottom: 2 }}>{r}</li>
          ))}
        </ul>
      </div>

      <div style={{ 
        padding: '8px 10px', 
        background: 'rgba(188, 140, 255, 0.1)', 
        borderRadius: 6,
        fontSize: 11,
        fontStyle: 'italic',
        color: 'var(--text-secondary)',
        lineHeight: 1.4,
        borderLeft: '2px solid var(--accent-purple)'
      }}>
        " {advice.aria_insight} "
      </div>
    </div>
  );
}

function ReferenceComparison({ program, leaderboardEntry }) {
  const [references, setReferences] = useState([]);
  const [selectedRefId, setSelectedRefId] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    apiService.getReferences()
      .then(d => {
        setReferences(d.entries || []);
        // Auto-select a reference from same family if possible
        const sameFamily = (d.entries || []).find(r => r.architecture_family === (leaderboardEntry?.architecture_family || program?.architecture_family));
        if (sameFamily) setSelectedRefId(sameFamily.result_id);
        else if (d.entries?.length > 0) setSelectedRefId(d.entries[0].result_id);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [program.result_id]);

  const selectedRef = references.find(r => r.result_id === selectedRefId);
  if (references.length === 0 && !loading) return null;

  const metrics = [
    { key: 'validation_loss_ratio', label: 'Loss Ratio', higherIsBetter: false },
    { key: 'param_efficiency', label: 'Param Efficiency', higherIsBetter: true },
    { key: 'quant_int8_retention', label: 'Quant Retention', higherIsBetter: true },
    { key: 'robustness_long_ctx_score', label: 'Long-Context', higherIsBetter: true },
    { key: 'robustness_noise_score', label: 'Noise Score', higherIsBetter: false },
  ];

  const getValue = (obj, key) => {
    let val = obj?.[key];
    if (val === undefined || val === null) return null;
    return Number(val);
  };

  return (
    <div style={{ marginTop: 16, padding: 12, background: 'var(--bg-tertiary)', borderRadius: 8, border: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
          Compare to Reference
        </div>
        <select
          value={selectedRefId}
          onChange={e => setSelectedRefId(e.target.value)}
          style={{ fontSize: 11, padding: '2px 4px', background: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border)', borderRadius: 4 }}
        >
          {references.map(r => (
            <option key={r.result_id} value={r.result_id}>{r.reference_name || r.display_name}</option>
          ))}
        </select>
      </div>

      {selectedRef ? (
        <div style={{ display: 'grid', gap: 8 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', borderBottom: '1px solid var(--border)', pb: 4 }}>
            <span>Metric</span>
            <span style={{ textAlign: 'right' }}>Candidate</span>
            <span style={{ textAlign: 'right' }}>{selectedRef.reference_name || 'Ref'}</span>
          </div>
          {metrics.map(m => {
            const vCan = getValue(leaderboardEntry || program, m.key);
            const vRef = getValue(selectedRef, m.key);
            if (vCan === null && vRef === null) return null;
            
            const diff = (vCan !== null && vRef !== null) ? (m.higherIsBetter ? vCan - vRef : vRef - vCan) : null;
            const diffPct = (diff !== null && vRef !== 0) ? (diff / Math.abs(vRef)) * 100 : null;
            const isBetter = diff !== null && diff > 0;

            return (
              <div key={m.key} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, fontSize: 12, alignItems: 'center' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{m.label}</span>
                <span style={{ textAlign: 'right', fontWeight: 600, color: isBetter ? 'var(--accent-green)' : 'var(--text-primary)' }}>
                  {vCan !== null ? vCan.toFixed(3) : '--'}
                  {diffPct !== null && (
                    <div style={{ fontSize: 9, opacity: 0.8 }}>
                      {isBetter ? '+' : ''}{diffPct.toFixed(1)}%
                    </div>
                  )}
                </span>
                <span style={{ textAlign: 'right', color: 'var(--text-muted)' }}>
                  {vRef !== null ? vRef.toFixed(3) : '--'}
                </span>
              </div>
            );
          })}
        </div>
      ) : (
        <div style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'center', py: 8 }}>
          {loading ? 'Loading references...' : 'No reference architectures pinned yet.'}
        </div>
      )}
    </div>
  );
}

const TIER_COLORS = {
  screening: 'var(--accent-blue)',
  investigation: 'var(--accent-yellow)',
  validation: 'var(--accent-purple)',
  breakthrough: 'var(--accent-green)',
};

const TIER_LABELS = {
  screening: 'Screening',
  investigation: 'Investigation',
  validation: 'Validation',
  breakthrough: 'Breakthrough',
};

function decisionGate(entry) {
  const checks = {
    screeningEvidence: entry.screening_loss_ratio != null && entry.screening_novelty != null,
    investigationEvidence: entry.investigation_loss_ratio != null && entry.investigation_robustness != null,
    robustnessFloor: entry.investigation_robustness != null && entry.investigation_robustness >= 0.5,
    validationEvidence: entry.validation_loss_ratio != null
      && entry.validation_baseline_ratio != null
      && entry.validation_multi_seed_std != null,
    baselineBeatsReference: entry.validation_baseline_ratio != null && entry.validation_baseline_ratio < 1.0,
    consistencyBounded: entry.validation_multi_seed_std != null && entry.validation_multi_seed_std <= 0.12,
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
    checks,
  };
}

function TierBadge({ tier, entry }) {
  if (!tier) return null;

  const gate = decisionGate(entry || {});
  const checkLabels = {
    screeningEvidence: 'Screening evidence',
    investigationEvidence: 'Investigation evidence',
    robustnessFloor: 'Robustness \u2265 0.50',
    validationEvidence: 'Validation evidence',
    baselineBeatsReference: 'Baseline < 1.0',
    consistencyBounded: 'Multi-seed std \u2264 0.12',
  };

  const tooltipLines = ['Promotion criteria:'];
  Object.entries(gate.checks).forEach(([name, ok]) => {
    tooltipLines.push(`${ok ? '\u2713' : '\u2717'} ${checkLabels[name] || name}`);
  });

  if (tier !== 'breakthrough' && gate.missing.length > 0) {
    tooltipLines.push('');
    tooltipLines.push(`Missing for breakthrough: ${gate.missing.map(m => checkLabels[m] || m).join(', ')}`);
  }

  const tooltip = tooltipLines.join('\n');

  return (
    <span
      title={tooltip}
      style={{
        padding: '2px 8px',
        borderRadius: 4,
        fontSize: 11,
        fontWeight: 600,
        color: TIER_COLORS[tier] || 'var(--text-muted)',
        background: `${TIER_COLORS[tier] || 'var(--text-muted)'}22`,
        border: `1px solid ${TIER_COLORS[tier] || 'var(--border)'}`,
        textTransform: 'uppercase',
        cursor: 'help',
      }}
    >
      {TIER_LABELS[tier] || tier}
    </span>
  );
}

function HypothesisInfo({ hypothesis }) {
  if (!hypothesis) return null;
  const colors = {
    confirmed: 'var(--accent-green)',
    refuted: 'var(--accent-red)',
    inconclusive: 'var(--accent-yellow)',
    pending: 'var(--text-muted)',
    testing: 'var(--accent-blue)',
  };
  return (
    <div style={{
      padding: 12, background: 'var(--bg-tertiary)', borderRadius: 4,
      borderLeft: `2px solid ${colors[hypothesis.status] || 'var(--border)'}`,
      fontSize: 13,
    }}>
      <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', marginBottom: 4,
        color: colors[hypothesis.status] || 'var(--text-muted)' }}>
        Hypothesis: {hypothesis.status}
      </div>
      <div style={{ marginBottom: 4 }}>{hypothesis.prediction}</div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
        <em>Metric:</em> {hypothesis.success_metric}
      </div>
      {hypothesis.outcome_summary && (
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4 }}>
          <em>Outcome:</em> {hypothesis.outcome_summary}
        </div>
      )}
    </div>
  );
}

function BenchmarkEvidenceSnapshot({ program, leaderboardEntry }) {
  const tier = leaderboardEntry?.tier;
  const isBreakthrough = tier === 'breakthrough';
  const ratio = Number(program?.baseline_loss_ratio);
  const hasRatio = Number.isFinite(ratio);
  const beatsBaseline = hasRatio && ratio < 1;

  if (!hasRatio && !isBreakthrough) return null;

  return (
    <div style={{
      marginTop: 12,
      padding: 10,
      background: 'var(--bg-tertiary)',
      borderRadius: 6,
      borderLeft: `3px solid ${beatsBaseline ? 'var(--accent-green)' : 'var(--accent-yellow)'}`,
    }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 6 }}>
        Benchmark Evidence Snapshot
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
        <div>
          <strong>Fixed-seed baseline ratio:</strong>{' '}
          {hasRatio ? ratio.toFixed(3) : 'Unavailable'}
          {hasRatio && (
            <span style={{ marginLeft: 6, color: beatsBaseline ? 'var(--accent-green)' : 'var(--accent-red)' }}>
              {beatsBaseline ? '(< 1.0, beats baseline)' : '(≥ 1.0, below baseline)'}
            </span>
          )}
        </div>
        <div>
          <strong>Interpretation:</strong>{' '}
          {hasRatio
            ? (beatsBaseline
              ? 'This architecture outperforms the fixed-seed transformer baseline on the same setup.'
              : 'This architecture does not yet beat the fixed-seed transformer baseline on this snapshot.')
            : 'Baseline comparison was not recorded for this result.'}
        </div>
        {isBreakthrough && (
          <div>
            <strong>Breakthrough note:</strong> tier promotion also requires multi-seed stability and robustness checks beyond this fixed-seed snapshot.
          </div>
        )}
      </div>
    </div>
  );
}

function ExternalBenchmarkCard({ program }) {
  const raw = program?.external_benchmarks || program?.benchmark_scores || program?.external_benchmark_scores;
  const entries = Array.isArray(raw) ? raw : raw && typeof raw === 'object' ? Object.entries(raw).map(([name, score]) => ({ name, score })) : [];
  const normalized = entries
    .map((entry) => {
      if (!entry || typeof entry !== 'object') return null;
      return {
        name: entry.name || entry.benchmark || entry.task || 'benchmark',
        score: entry.score ?? entry.value ?? null,
        unit: entry.unit || entry.metric || '',
        source: entry.source || entry.provider || '',
        date: entry.date || entry.timestamp || '',
      };
    })
    .filter(Boolean);

  return (
    <div style={{
      marginTop: 12,
      padding: 10,
      background: 'var(--bg-tertiary)',
      borderRadius: 6,
      border: '1px solid var(--border)',
    }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 6 }}>
        External Benchmarks
      </div>
      {normalized.length === 0 ? (
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          No external benchmark scores recorded yet. Recommended for early research: Open LLM Leaderboard v1
          (MMLU, ARC, HellaSwag, TruthfulQA, Winogrande, GSM8K) via lm-eval harness.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {normalized.map((entry, idx) => (
            <div key={`${entry.name}-${idx}`} style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              <strong>{entry.name}:</strong>{' '}
              {entry.score != null ? entry.score : 'n/a'}
              {entry.unit ? ` ${entry.unit}` : ''}
              {entry.source ? <span style={{ color: 'var(--text-muted)' }}> · {entry.source}</span> : null}
              {entry.date ? <span style={{ color: 'var(--text-muted)' }}> · {entry.date}</span> : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function EvidenceFlagChips({ flags }) {
  if (!flags) return null;
  const entries = [
    { key: 'has_baseline', label: 'Baseline' },
    { key: 'has_cka_artifact', label: 'CKA Artifact' },
    { key: 'has_multi_seed', label: 'Multi-Seed' },
    { key: 'has_hypothesis', label: 'Hypothesis' },
  ];
  return (
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
      {entries.map(({ key, label }) => {
        const ok = flags[key];
        return (
          <span key={key} style={{
            fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 4,
            color: ok ? 'var(--accent-green)' : 'var(--accent-red)',
            background: ok ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
            border: `1px solid ${ok ? 'var(--accent-green)' : 'var(--accent-red)'}44`,
          }}>
            {ok ? '\u2713' : '\u2717'} {label}
          </span>
        );
      })}
    </div>
  );
}

function HypothesisLineage({ chain }) {
  if (!chain || chain.length === 0) return <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No linked hypothesis</div>;
  const statusColors = {
    confirmed: 'var(--accent-green)',
    refuted: 'var(--accent-red)',
    inconclusive: 'var(--accent-yellow)',
    pending: 'var(--text-muted)',
    testing: 'var(--accent-blue)',
  };
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {chain.map((h, i) => (
        <div key={h.hypothesis_id || i} style={{
          fontSize: 12, padding: '4px 8px',
          borderLeft: `3px solid ${statusColors[h.status] || 'var(--border)'}`,
          color: 'var(--text-secondary)',
        }}>
          <span style={{ fontWeight: 600, color: statusColors[h.status] || 'var(--text-muted)', textTransform: 'uppercase', fontSize: 10, marginRight: 6 }}>
            [{h.status}]
          </span>
          {h.prediction || h.title || 'Untitled hypothesis'}
        </div>
      ))}
    </div>
  );
}

function OutcomesByPhase({ outcomes }) {
  if (!outcomes) return null;
  const phases = [
    { key: 'screening', label: 'Screening' },
    { key: 'investigation', label: 'Investigation' },
    { key: 'validation', label: 'Validation' },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {phases.map(({ key, label }) => {
        const data = outcomes[key];
        if (!data) return (
          <div key={key} style={{ fontSize: 12, color: 'var(--text-muted)', padding: '4px 0', borderBottom: '1px solid var(--border)' }}>
            {label}: --
          </div>
        );
        const passed = data.passed;
        return (
          <div key={key} style={{ fontSize: 12, padding: '4px 0', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
            <span style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              {data.discovery_loss_ratio != null && <span>D.LR: {Number(data.discovery_loss_ratio).toFixed(4)}</span>}
              {data.validation_loss_ratio != null && <span>V.LR: {Number(data.validation_loss_ratio).toFixed(4)}</span>}
              {data.loss_ratio != null && <span>LR: {Number(data.loss_ratio).toFixed(4)}</span>}
              {data.novelty != null && <span>Nov: {Number(data.novelty).toFixed(3)}</span>}
              {data.robustness != null && <span>Rob: {Number(data.robustness).toFixed(3)}</span>}
              {data.baseline_ratio != null && <span>BL: {Number(data.baseline_ratio).toFixed(3)}</span>}
              {data.multi_seed_std != null && <span>Std: {Number(data.multi_seed_std).toFixed(4)}</span>}
              {passed !== undefined && (
                <span style={{
                  fontSize: 10, fontWeight: 600, padding: '1px 6px', borderRadius: 3,
                  color: passed ? 'var(--accent-green)' : 'var(--accent-red)',
                  background: passed ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                }}>
                  {passed ? 'PASS' : 'FAIL'}
                </span>
              )}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function FailureContext({ context }) {
  if (!context || (!context.stage_at_death && !context.error_type)) return null;
  const topErrors = Object.entries(context.experiment_errors || {}).sort((a, b) => b[1] - a[1]).slice(0, 5);
  return (
    <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
      {context.stage_at_death && <div>Stage at death: <strong>{context.stage_at_death}</strong></div>}
      {context.error_type && <div>Error type: <strong>{context.error_type}</strong></div>}
      {topErrors.length > 0 && (
        <div style={{ marginTop: 6 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>Top Experiment Errors</div>
          {topErrors.map(([err, count]) => (
            <div key={err} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', borderBottom: '1px solid var(--border)' }}>
              <span style={{ fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 300 }}>{err}</span>
              <span style={{ fontSize: 11, color: 'var(--accent-red)', flexShrink: 0, marginLeft: 8 }}>{count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function RecommendationCard({ recommendation }) {
  if (!recommendation) return null;
  const actionColors = {
    'investigate': 'var(--accent-blue)',
    'validate': 'var(--accent-purple)',
    're-validate': 'var(--accent-purple)',
    'scale up or publish': 'var(--accent-green)',
    'publish': 'var(--accent-green)',
    're-investigate or archive': 'var(--accent-yellow)',
    'archive': 'var(--accent-red)',
  };
  const color = actionColors[recommendation.action] || 'var(--accent-blue)';
  return (
    <div style={{
      padding: 12, borderRadius: 6, border: `1px solid ${color}`,
      background: `${color}11`,
    }}>
      <div style={{ fontSize: 14, fontWeight: 700, color, textTransform: 'uppercase', marginBottom: 4 }}>
        {recommendation.action}
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
        {recommendation.rationale}
      </div>
      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
        Confidence: {recommendation.confidence}
      </div>
    </div>
  );
}

const TOKEN_MIXING_OPS = {
  local_window_attn: { label: 'Local Window Attention', desc: 'Windowed causal self-attention (Q=K=V)', family: 'attention', complexity: 'linear' },
  sliding_window_mask: { label: 'Sliding Window Mask', desc: 'Exponential distance decay for windowed mixing', family: 'attention', complexity: 'linear' },
  softmax_last: { label: 'Softmax (dim)', desc: 'Standard softmax along last dimension', family: 'attention', complexity: 'quadratic' },
  softmax_seq: { label: 'Softmax (seq)', desc: 'Softmax along sequence dimension', family: 'attention', complexity: 'quadratic' },
  causal_mask: { label: 'Causal Mask', desc: 'Lower-triangular causal masking', family: 'attention', complexity: 'quadratic' },
  multi_head_mix: { label: 'Multi-Head Mix', desc: 'Multi-head reshape + per-head normalization', family: 'attention', complexity: 'quadratic' },
  selective_scan: { label: 'Selective Scan (SSM)', desc: 'Input-dependent state scan (Mamba-style)', family: 'ssm', complexity: 'linear' },
  conv1d_seq: { label: '1D Conv (depthwise)', desc: 'Depthwise convolution along sequence', family: 'conv', complexity: 'linear' },
  rfft_seq: { label: 'FFT (forward)', desc: 'Real FFT along sequence dimension', family: 'frequency', complexity: 'linear' },
  irfft_seq: { label: 'FFT (inverse)', desc: 'Inverse real FFT along sequence', family: 'frequency', complexity: 'linear' },
  sort_seq: { label: 'Sort Mixing', desc: 'Sort along sequence by learned key', family: 'sorting', complexity: 'nlogn' },
  argsort_seq: { label: 'Argsort Mixing', desc: 'Argsort along sequence dimension', family: 'sorting', complexity: 'nlogn' },
  token_pool_restore: { label: 'Token Pooling', desc: 'Pool adjacent token pairs then restore', family: 'pooling', complexity: 'linear' },
  basis_expansion: { label: 'Basis Expansion', desc: 'Sinusoidal basis projection (neural operator)', family: 'functional', complexity: 'linear' },
  integral_kernel: { label: 'Integral Kernel', desc: 'Learned kernel mixing over positions', family: 'functional', complexity: 'quadratic' },
  fixed_point_iter: { label: 'Fixed-Point Iter', desc: 'Implicit fixed-point iteration (DEQ-style)', family: 'functional', complexity: 'linear' },
};

const FAMILY_COLORS = {
  attention: 'var(--accent-blue)',
  ssm: 'var(--accent-green)',
  conv: 'var(--accent-yellow)',
  frequency: 'var(--accent-purple)',
  sorting: 'var(--accent-red)',
  pooling: 'var(--text-muted)',
  functional: '#e0a060',
};

const FAMILY_LABELS = {
  attention: 'QKV-based',
  ssm: 'State Space',
  conv: 'Convolution',
  frequency: 'Frequency Domain',
  sorting: 'Sort-based',
  pooling: 'Pooling',
  functional: 'Functional/Operator',
};

function TokenMixingTaxonomy({ graphJson }) {
  if (!graphJson) return null;

  const rawNodes = graphJson.nodes || [];
  const nodes = Array.isArray(rawNodes) ? rawNodes : Object.values(rawNodes);
  const detected = [];
  const seen = new Set();
  for (const node of nodes) {
    if (!node || typeof node !== 'object') continue;
    const opName = node.op || node.op_name;
    if (opName && TOKEN_MIXING_OPS[opName] && !seen.has(opName)) {
      seen.add(opName);
      detected.push({ op: opName, ...TOKEN_MIXING_OPS[opName] });
    }
  }

  if (detected.length === 0) return null;

  const families = [...new Set(detected.map(d => d.family))];
  const hasQKV = families.includes('attention');
  const summary = hasQKV
    ? `Uses QKV-style attention${families.length > 1 ? ' + ' + families.filter(f => f !== 'attention').map(f => FAMILY_LABELS[f]).join(', ') : ''}`
    : `QKV-free: ${families.map(f => FAMILY_LABELS[f]).join(' + ')}`;

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
        Token Mixing Mechanism
      </div>
      <div style={{
        padding: '8px 12px', background: 'var(--bg-tertiary)', borderRadius: 6,
        borderLeft: `3px solid ${hasQKV ? 'var(--accent-blue)' : 'var(--accent-green)'}`,
      }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: hasQKV ? 'var(--accent-blue)' : 'var(--accent-green)', marginBottom: 6 }}>
          {summary}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {detected.map(d => (
            <div key={d.op} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
              <span style={{
                padding: '1px 6px', borderRadius: 3, fontSize: 10, fontWeight: 600,
                background: `${FAMILY_COLORS[d.family]}22`, color: FAMILY_COLORS[d.family],
              }}>
                {FAMILY_LABELS[d.family]}
              </span>
              <code style={{ color: 'var(--text-secondary)', fontSize: 11 }}>{d.op}</code>
              <span style={{ color: 'var(--text-muted)' }}>{d.desc}</span>
              <span style={{
                marginLeft: 'auto', fontSize: 10, color: 'var(--text-muted)',
                fontStyle: 'italic', flexShrink: 0,
              }}>
                {d.complexity === 'linear' ? 'O(n)' : d.complexity === 'nlogn' ? 'O(n log n)' : 'O(n²)'}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

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

function GatingDiagnostics({ program }) {
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
                background: 'rgba(188, 140, 255, 0.15)', color: 'var(--accent-purple)',
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
                const isCollapsed = nExperts > 2 && v < (maxUtil * 0.1);
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
              <RoutingHeatmap key={name} data={data} nExperts={nExperts} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function RefinementRationale({ program }) {
  const graphMetadata = program?.graph_json_parsed?.metadata;
  const refinement = graphMetadata?.refinement;
  if (!refinement || typeof refinement !== 'object') return null;

  const intent = refinement.intent || 'balanced';
  const intentScore = Number(refinement.intent_score);
  const hasIntentScore = Number.isFinite(intentScore);
  const sourceResultId = refinement.source_result_id;
  const seedFingerprint = refinement.seed_fingerprint;
  const fallback = Boolean(refinement.fallback);
  const scoreBreakdown = refinement.intent_score_breakdown || {};
  const weightedTerms = scoreBreakdown?.weighted_terms || {};
  const weightedEntries = Object.entries(weightedTerms)
    .filter(([, value]) => Number.isFinite(Number(value)))
    .sort((a, b) => Number(b[1]) - Number(a[1]));
  const scoreBreakdownTooltip = weightedEntries.length > 0
    ? weightedEntries.map(([name, value]) => `${name}: ${Number(value).toFixed(4)}`).join('\n')
    : 'No score components available';

  return (
    <div style={{
      padding: 12,
      background: 'var(--bg-tertiary)',
      borderRadius: 6,
      border: '1px solid var(--border)',
    }}>
      <div style={{
        fontSize: 12,
        color: 'var(--text-secondary)',
        fontWeight: 600,
        textTransform: 'uppercase',
        marginBottom: 8,
      }}>
        Refinement Rationale
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12 }}>
        <div style={{ color: 'var(--text-muted)' }}>Intent</div>
        <div style={{ fontWeight: 600 }}>{intent}</div>
        <div style={{ color: 'var(--text-muted)' }}>Intent Score</div>
        <div
          style={{ fontWeight: 600, cursor: weightedEntries.length > 0 ? 'help' : 'default' }}
          title={scoreBreakdownTooltip}
        >
          {hasIntentScore ? intentScore.toFixed(4) : '--'}
        </div>
        {sourceResultId && (
          <>
            <div style={{ color: 'var(--text-muted)' }}>Parent Result</div>
            <div style={{ fontFamily: 'monospace' }}>{String(sourceResultId).slice(0, 12)}</div>
          </>
        )}
        {seedFingerprint && (
          <>
            <div style={{ color: 'var(--text-muted)' }}>Parent Fingerprint</div>
            <div style={{ fontFamily: 'monospace' }}>{seedFingerprint}</div>
          </>
        )}
        <div style={{ color: 'var(--text-muted)' }}>Selection Path</div>
        <div style={{ color: fallback ? 'var(--accent-yellow)' : 'var(--accent-green)' }}>
          {fallback ? 'fallback generation' : 'learning-guided refinement'}
        </div>
      </div>
      {weightedEntries.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
          Components:{' '}
          {weightedEntries
            .slice(0, 3)
            .map(([name, value]) => `${name} ${Number(value).toFixed(3)}`)
            .join(' · ')}
        </div>
      )}
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
        Score combines learned op success and intent-specific objective weighting.
      </div>
    </div>
  );
}

function RefinementLineage({ program, onViewInLeaderboard }) {
  const lineage = Array.isArray(program?.lineage_chain) ? program.lineage_chain : [];
  if (lineage.length === 0) return null;

  const short = (value, n = 12) => {
    const s = String(value || '').trim();
    if (!s) return '--';
    return s.length > n ? s.slice(0, n) : s;
  };

  return (
    <div style={{
      padding: 12,
      background: 'var(--bg-tertiary)',
      borderRadius: 6,
      border: '1px solid var(--border)',
    }}>
      <div style={{
        fontSize: 12,
        color: 'var(--text-secondary)',
        fontWeight: 600,
        textTransform: 'uppercase',
        marginBottom: 8,
      }}>
        Refinement Lineage
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
        New refinements create new fingerprints. Lineage tracks each child back to its parent result so you can iteratively improve from a base fingerprint.
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {lineage.map((entry, idx) => (
          <div
            key={`${entry?.result_id || 'lineage'}-${idx}`}
            style={{
              display: 'grid',
              gridTemplateColumns: '36px 1fr auto',
              gap: 8,
              alignItems: 'center',
              padding: '6px 8px',
              borderRadius: 4,
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border)',
            }}
          >
            <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600 }}>
              L{idx}
            </span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <span style={{ fontFamily: 'monospace', fontSize: 11 }}>
                {short(entry?.graph_fingerprint, 20)}
              </span>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                result {short(entry?.result_id, 12)}
                {entry?.refinement?.intent ? ` · intent ${entry.refinement.intent}` : ''}
                {entry?.refinement?.analysis_driven && (
                  <span
                    style={{
                      marginLeft: 4, padding: '1px 5px', borderRadius: 3, fontSize: 9,
                      background: 'var(--accent-purple)', color: '#fff', fontWeight: 700,
                    }}
                    title={entry?.refinement?.analysis_recipe?.primary_target || 'Data-driven refinement'}
                  >
                    DATA-DRIVEN
                  </span>
                )}
              </span>
            </div>
            {entry?.result_id && onViewInLeaderboard && (
              <button
                className="refresh-btn"
                style={{ fontSize: 10, padding: '2px 8px' }}
                onClick={() => onViewInLeaderboard(entry.result_id)}
              >
                Open
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function RefinementAdvisor({ analysis, loading, error, onLaunchRefinement, actionStarting }) {
  if (loading) return <div style={{ fontSize: 12, color: 'var(--text-muted)', padding: 12 }}>Analyzing program...</div>;
  if (error) return <div style={{ fontSize: 12, color: 'var(--accent-red)', padding: 12 }}>Analysis error: {error}</div>;
  if (!analysis || analysis.analysis_quality === 'no_data') return null;

  const recipe = analysis.recipe || {};
  const opHealth = analysis.op_health || [];
  const additions = analysis.recommended_additions || [];
  const gaps = (analysis.behavioral_gaps || []).filter(g => g.severity !== 'low');
  const stats = analysis.population_stats || {};

  const intentColors = { quality: 'var(--accent-red)', novelty: 'var(--accent-blue)', compression: '#1f7a4f', balanced: 'var(--accent-yellow)' };
  const healthColors = { strong: 'var(--accent-green)', weak: 'var(--accent-red)', risky: 'var(--accent-yellow)', untested: 'var(--text-muted)', neutral: 'var(--text-secondary)' };

  return (
    <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6, border: '1px solid var(--border)' }}>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
        Refinement Advisor
      </div>

      {/* Recipe banner */}
      <div style={{
        padding: 10, borderRadius: 6, marginBottom: 10,
        background: 'var(--bg-secondary)', border: `1px solid ${intentColors[recipe.recommended_intent] || 'var(--border)'}`,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <span style={{
            fontSize: 10, fontWeight: 700, textTransform: 'uppercase', padding: '2px 8px', borderRadius: 3,
            background: intentColors[recipe.recommended_intent] || 'var(--border)', color: '#fff',
          }}>
            {recipe.recommended_intent || 'balanced'}
          </span>
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            confidence: {recipe.confidence || 'low'}
          </span>
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            {stats.n_stage1_passed || 0} S1 survivors analyzed
          </span>
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{recipe.human_summary}</div>
      </div>

      {/* Op health grid */}
      {opHealth.length > 0 && (
        <div style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Op Health</div>
          <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4 }}>
            {opHealth.map(op => (
              <div
                key={op.op_name}
                style={{
                  minWidth: 100, padding: '6px 8px', borderRadius: 4, fontSize: 11,
                  background: 'var(--bg-secondary)', border: `1px solid ${healthColors[op.health] || 'var(--border)'}`,
                }}
                title={op.swap_candidates?.length
                  ? `Swap candidates: ${op.swap_candidates.map(c => `${c.op_name} (${(c.s1_rate * 100).toFixed(0)}%)`).join(', ')}`
                  : `${op.recommendation} — S1 rate: ${(op.global_s1_rate * 100).toFixed(1)}%`}
              >
                <div style={{ fontFamily: 'monospace', fontWeight: 600, marginBottom: 2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {op.op_name}
                </div>
                <div style={{ color: healthColors[op.health], fontWeight: 600 }}>
                  {op.health}
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>
                  S1: {(op.global_s1_rate * 100).toFixed(0)}% ({op.n_used} uses)
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                  {op.recommendation}
                  {op.swap_candidates?.length > 0 && ` (${op.swap_candidates.length} alt)`}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommended additions */}
      {additions.length > 0 && (
        <details style={{ marginBottom: 10 }}>
          <summary style={{ fontSize: 11, color: 'var(--text-muted)', cursor: 'pointer', marginBottom: 4 }}>
            Recommended Additions ({additions.length})
          </summary>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 4 }}>
            {additions.map(a => (
              <div key={a.op_name} style={{
                display: 'flex', gap: 8, alignItems: 'center', fontSize: 11,
                padding: '4px 8px', background: 'var(--bg-secondary)', borderRadius: 4,
              }}>
                <span style={{ fontFamily: 'monospace', fontWeight: 600, minWidth: 120 }}>{a.op_name}</span>
                <span style={{ color: 'var(--accent-green)' }}>S1: {(a.global_s1_rate * 100).toFixed(0)}%</span>
                <span style={{ color: 'var(--text-muted)' }}>{a.top_performer_frequency} uses</span>
                <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>{a.rationale}</span>
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Behavioral gaps */}
      {gaps.length > 0 && (
        <div style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Behavioral Gaps</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
            {gaps.map(g => (
              <div key={g.metric} style={{
                padding: '6px 8px', background: 'var(--bg-secondary)', borderRadius: 4, fontSize: 11,
                borderLeft: `3px solid ${g.severity === 'high' ? 'var(--accent-red)' : 'var(--accent-yellow)'}`,
              }}>
                <div style={{ fontWeight: 600, marginBottom: 2 }}>{g.label}</div>
                <div style={{ color: 'var(--text-muted)' }}>
                  Program: {g.program_value?.toFixed(3)} vs Pop: {g.population_mean?.toFixed(3)} (z={g.z_score > 0 ? '+' : ''}{g.z_score?.toFixed(1)})
                </div>
                {g.improvement_ops?.length > 0 && (
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                    Try: {g.improvement_ops.join(', ')}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action button */}
      <button
        className="start-btn"
        disabled={actionStarting === 'refine_advisor'}
        onClick={() => onLaunchRefinement(recipe.recommended_intent || 'balanced', 'refine_advisor', 'Failed to start analysis-driven refinement')}
        style={{ padding: '6px 16px', fontSize: 12, background: 'var(--accent-purple)', borderColor: 'var(--accent-purple)' }}
        title={recipe.primary_target || 'Refine using data-driven analysis'}
      >
        {actionStarting === 'refine_advisor' ? 'Starting...' : 'Refine with Recommendation'}
      </button>
    </div>
  );
}

function ProgramDetail({ resultId, onClose, onActionComplete, onSelectExperiment, onViewInLeaderboard, onSelectCampaign, onOpenInDesigner, onAddToComparison, eligibilityByResultId }) {
  const [program, setProgram] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [scaleUpOpen, setScaleUpOpen] = useState(false);
  const [scaleUpConfig, setScaleUpConfig] = useState({ steps: 5000, batch_size: 8, seq_len: 512 });
  const [scaleUpStarting, setScaleUpStarting] = useState(false);
  const [manualRunOpen, setManualRunOpen] = useState(false);
  const [manualRunStarting, setManualRunStarting] = useState(false);
  const [manualRunConfig, setManualRunConfig] = useState({
    steps: 2500, batch_size: 4, n_training_programs: 3, seq_len: 256,
    data_source: 'corpus',
    hf_dataset: 'roneneldan/TinyStories', hf_subset: '',
  });
  const [backfillRunning, setBackfillRunning] = useState(false);
  const [backfillResult, setBackfillResult] = useState(null);
  const [lossBackfillRunning, setLossBackfillRunning] = useState(false);
  const [lossBackfillResult, setLossBackfillResult] = useState(null);
  const [leaderboardEntry, setLeaderboardEntry] = useState(null);
  const [actionStarting, setActionStarting] = useState(null);
  const [actionError, setActionError] = useState(null);
  const [linkedHypothesis, setLinkedHypothesis] = useState(null);
  const [linkedDecision, setLinkedDecision] = useState(null);
  const [linkedExperiment, setLinkedExperiment] = useState(null);
  const [linkedCampaign, setLinkedCampaign] = useState(null);
  const [provenanceOpen, setProvenanceOpen] = useState(true);
  const [decisionPacket, setDecisionPacket] = useState(null);
  const [decisionPacketLoading, setDecisionPacketLoading] = useState(false);
  const [decisionPacketError, setDecisionPacketError] = useState(null);
  const [decisionPacketOpen, setDecisionPacketOpen] = useState(true);
  const [manifestLoading, setManifestLoading] = useState(false);
  const [manifestCopied, copyManifest] = useCopyToClipboard();
  const [latestRefineLaunch, setLatestRefineLaunch] = useState(null);
  const [refineLaunchHistory, setRefineLaunchHistory] = useState([]);
  const [refineTrace, setRefineTrace] = useState(null);
  const [refineTraceLoading, setRefineTraceLoading] = useState(false);
  const [refineAnalysis, setRefineAnalysis] = useState(null);
  const [refineAnalysisLoading, setRefineAnalysisLoading] = useState(false);
  const [refineAnalysisError, setRefineAnalysisError] = useState(null);
  const [drawerWidthVw, setDrawerWidthVw] = useState(45);
  const [drawerMaximized, setDrawerMaximized] = useState(false);
  const [resizingDrawer, setResizingDrawer] = useState(false);
  const drawerResizeRef = useRef({ startX: 0, startVw: 45 });

  const fetchAndCopyManifest = () => {
    if (!resultId) return;
    setManifestLoading(true);
    apiService.getReproducibilityManifest(resultId)
      .then(d => {
        copyManifest(JSON.stringify(d, null, 2));
        setManifestLoading(false);
      })
      .catch(() => { setManifestLoading(false); });
  };

  const fetchDecisionPacket = () => {
    if (!resultId) return;
    setDecisionPacketLoading(true);
    setDecisionPacketError(null);
    apiService.getDecisionPacket(resultId)
      .then(d => { setDecisionPacket(d); setDecisionPacketLoading(false); })
      .catch(e => { setDecisionPacketError('Failed: ' + e.message); setDecisionPacketLoading(false); });
  };

  useEffect(() => {
    if (!resultId) return;
    setLoading(true);
    setError(null);
    setLatestRefineLaunch(null);
    setRefineLaunchHistory([]);
    setRefineTrace(null);
    setRefineTraceLoading(false);
    setLinkedHypothesis(null);
    setLinkedDecision(null);
    setLinkedExperiment(null);
    setLinkedCampaign(null);
    
    apiService.getProgram(resultId)
      .then(d => {
        setProgram(d);
        setLoading(false);
        // Fetch linked hypothesis via experiment
        if (d?.experiment_id) {
          apiService.getExperiment(d.experiment_id)
            .then(expData => {
              if (expData?.experiment) {
                setLinkedExperiment(expData.experiment);
                if (expData.experiment.campaign_id) {
                  setLinkedCampaign({ campaign_id: expData.experiment.campaign_id, title: expData.experiment.campaign_title || expData.experiment.campaign_id });
                  // Find hypothesis linked to this experiment
                  apiService.getCampaignHypotheses(expData.experiment.campaign_id)
                    .then(hyps => {
                      const linked = (Array.isArray(hyps) ? hyps : []).find(
                        h => h.experiment_id === d.experiment_id
                      );
                      if (linked) setLinkedHypothesis(linked);
                    })
                    .catch(() => {});
                  // Find decisions mentioning this result
                  apiService.getCampaignDecisions(expData.experiment.campaign_id)
                    .then(decs => {
                      const linked = (Array.isArray(decs) ? decs : []).find(d => {
                        const evidenceIds = d.evidence_ids || [];
                        return Array.isArray(evidenceIds) && evidenceIds.includes(resultId);
                      });
                      if (linked) setLinkedDecision(linked);
                    })
                    .catch(() => {});
                }
              }
            })
            .catch(() => {});
        }
      })
      .catch(e => { setError('Failed to load program: ' + e.message); setLoading(false); });
    // Fetch leaderboard entry for this result
    apiService.getLeaderboard('?limit=200')
      .then(data => {
        if (data?.entries) {
          const entry = data.entries.find(e => e.result_id === resultId);
          setLeaderboardEntry(entry || null);
        }
      })
      .catch(() => {});
  }, [resultId]);

  // Auto-fetch refinement analysis for S1 survivors
  useEffect(() => {
    if (!resultId || !program?.stage1_passed) return;
    setRefineAnalysisLoading(true);
    setRefineAnalysisError(null);
    apiCall(`/api/programs/${encodeURIComponent(resultId)}/refine-analysis`)
      .then(r => r.ok ? r.json() : r.json().then(d => Promise.reject(new Error(d.error || 'Failed'))))
      .then(data => { setRefineAnalysis(data); setRefineAnalysisLoading(false); })
      .catch(e => { setRefineAnalysisError(e.message); setRefineAnalysisLoading(false); });
  }, [resultId, program?.stage1_passed]);

  useEffect(() => {
    if (!latestRefineLaunch?.experimentId || !resultId) return;

    let cancelled = false;
    let intervalId = null;

    const summarizeTrace = (payload) => {
      const experiment = payload?.experiment || {};
      const programs = Array.isArray(payload?.programs) ? payload.programs : [];

      const withRefinementMeta = programs.map(row => {
        let refinement = null;
        try {
          const raw = row?.graph_json;
          if (raw && typeof raw === 'string') {
            const parsed = JSON.parse(raw);
            refinement = parsed?.metadata?.refinement || null;
          }
        } catch (_) {
          refinement = null;
        }
        return { ...row, _refinement: refinement };
      });

      const lineage = withRefinementMeta.filter(
        row => String(row?._refinement?.source_result_id || '') === String(resultId),
      );
      const scoped = lineage.length > 0 ? lineage : withRefinementMeta;

      const finiteLosses = scoped
        .map(row => Number(row?.loss_ratio))
        .filter(value => Number.isFinite(value));
      const bestLoss = finiteLosses.length > 0 ? Math.min(...finiteLosses) : null;
      const stage1Survivors = scoped.filter(row => Boolean(row?.stage1_passed)).length;

      const uniqueFingerprints = [];
      const uniqueResultIds = [];
      const newCandidates = [];
      for (const row of scoped) {
        const fp = String(row?.graph_fingerprint || '').trim();
        const rid = String(row?.result_id || '').trim();
        if (fp && fp !== String(program?.graph_fingerprint || '') && !uniqueFingerprints.includes(fp)) {
          uniqueFingerprints.push(fp);
        }
        if (rid && rid !== String(resultId) && !uniqueResultIds.includes(rid)) {
          uniqueResultIds.push(rid);
        }
        if (rid && fp && rid !== String(resultId) && !newCandidates.some(c => c.resultId === rid)) {
          newCandidates.push({ resultId: rid, fingerprint: fp });
        }
      }

      const status = String(experiment?.status || '').toLowerCase();
      const completed = Boolean(experiment?.completed_at) || status === 'completed' || status === 'failed' || status === 'cancelled';

      return {
        status: status || 'running',
        completed,
        experiment,
        totals: {
          programs: programs.length,
          scopedPrograms: scoped.length,
          stage1Survivors,
          bestLoss,
        },
        newFingerprints: uniqueFingerprints.slice(0, 6),
        newResultIds: uniqueResultIds.slice(0, 6),
        newCandidates: newCandidates.slice(0, 6),
      };
    };

    const pollTrace = async () => {
      if (cancelled) return;
      setRefineTraceLoading(true);
      try {
        const response = await apiCall(`/api/experiments/${latestRefineLaunch.experimentId}`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        if (cancelled) return;
        const tracePayload = summarizeTrace(payload);
        setRefineTrace(tracePayload);
        setRefineLaunchHistory(prev => prev.map(item => (
          item.experimentId === latestRefineLaunch.experimentId
            ? {
                ...item,
                status: tracePayload.status,
                topCandidate: tracePayload.newCandidates?.[0] || null,
              }
            : item
        )));
        if (tracePayload.completed && intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
      } catch (e) {
        if (!cancelled) {
          setRefineTrace({ error: e?.message || 'Failed to load refinement trace' });
        }
      } finally {
        if (!cancelled) {
          setRefineTraceLoading(false);
        }
      }
    };

    pollTrace();
    intervalId = setInterval(pollTrace, 4000);

    return () => {
      cancelled = true;
      if (intervalId) clearInterval(intervalId);
    };
  }, [latestRefineLaunch, resultId, program?.graph_fingerprint]);

  useEffect(() => {
    if (!resizingDrawer) return undefined;
    const onMouseMove = (event) => {
      const viewportWidth = window.innerWidth || 1;
      const deltaPx = drawerResizeRef.current.startX - event.clientX;
      const deltaVw = (deltaPx / viewportWidth) * 100;
      const nextVw = drawerResizeRef.current.startVw + deltaVw;
      setDrawerWidthVw(Math.max(35, Math.min(90, nextVw)));
    };
    const onMouseUp = () => {
      setResizingDrawer(false);
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [resizingDrawer]);

  if (!resultId) return null;

  const fmt = (v, d = 4) => v != null ? Number(v).toFixed(d) : '--';
  const fmtMs = v => v != null ? `${Number(v).toFixed(1)}ms` : '--';
  const fmtMem = v => v != null ? `${Number(v).toFixed(1)}MB` : '--';
  const fmtInt = v => v != null ? Number(v).toLocaleString() : '--';
  const shortId = (v, n = 12) => {
    const s = String(v || '').trim();
    if (!s) return '--';
    return s.length > n ? s.slice(0, n) : s;
  };

  const tier = typeof leaderboardEntry?.tier === 'string' ? leaderboardEntry.tier.toLowerCase() : '';
  const hasInvestigationEvidence = leaderboardEntry?.investigation_loss_ratio != null;
  const hasValidationEvidence = leaderboardEntry?.validation_loss_ratio != null || Boolean(leaderboardEntry?.validation_passed);
  const fallbackEligibility = {
    investigationEligible: program?.stage1_passed && ((!leaderboardEntry && true) || (tier === 'screening' && !hasInvestigationEvidence)),
    validationEligible: tier === 'investigation' && Boolean(leaderboardEntry?.investigation_passed) && !hasValidationEvidence,
  };
  const resolvedEligibility = eligibilityByResultId?.[resultId] || fallbackEligibility;
  const lastRefinedCandidate =
    refineTrace?.newCandidates?.[0]
    || refineLaunchHistory.find(item => item?.topCandidate)?.topCandidate
    || null;

  const handleLaunchRefinement = async (intent, actionKey, failureLabel) => {
    setActionStarting(actionKey);
    try {
      setActionError(null);
      const res = await apiCall(`/api/experiments/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'refine_fingerprint',
          graph_fingerprints: [program.graph_fingerprint],
          n_programs: 24,
          model_source: 'fingerprint_refine',
          refine_intent: intent,
          mutation_rate: 0.85,
          ...(refineAnalysis ? { refine_analysis_json: refineAnalysis } : {}),
        }),
      });

      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        setActionError(payload.error || failureLabel);
      } else {
        const resolved = payload?.refine_resolution || {};
        setLatestRefineLaunch({
          experimentId: payload?.experiment_id,
          intent,
          startedAt: Date.now(),
          sourceResultId: resultId,
          sourceFingerprint: program?.graph_fingerprint,
          resolvedResultIds: Array.isArray(resolved?.result_ids) ? resolved.result_ids : [],
          resolvedFingerprints: Array.isArray(resolved?.resolved_fingerprints) ? resolved.resolved_fingerprints : [],
          unresolvedFingerprints: Array.isArray(resolved?.unresolved_fingerprints) ? resolved.unresolved_fingerprints : [],
        });
        setRefineLaunchHistory(prev => {
          const nextItem = {
            experimentId: payload?.experiment_id,
            intent,
            startedAt: Date.now(),
            sourceResultId: resultId,
            sourceFingerprint: program?.graph_fingerprint,
            status: 'running',
            topCandidate: null,
          };
          const deduped = prev.filter(item => item.experimentId !== nextItem.experimentId);
          return [nextItem, ...deduped].slice(0, 3);
        });
        if (onActionComplete) onActionComplete();
      }
    } catch (e) {
      setActionError('Error: ' + e.message);
    }
    setActionStarting(null);
  };

  return (
    <div className="program-drawer-backdrop" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div
        className="program-drawer"
        onClick={e => e.stopPropagation()}
        onMouseDown={e => e.stopPropagation()}
        style={drawerMaximized
          ? { width: '100%', minWidth: 0, maxWidth: '100%' }
          : { width: `${drawerWidthVw}vw`, minWidth: 460, maxWidth: '95vw' }}
      >
        {!drawerMaximized && (
          <div
            onMouseDown={(event) => {
              event.preventDefault();
              event.stopPropagation();
              drawerResizeRef.current = { startX: event.clientX, startVw: drawerWidthVw };
              setResizingDrawer(true);
            }}
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: 0,
              width: 8,
              cursor: 'col-resize',
              background: resizingDrawer ? 'rgba(88, 166, 255, 0.2)' : 'transparent',
              borderLeft: '1px solid rgba(88, 166, 255, 0.35)',
              zIndex: 2,
            }}
            title="Drag to resize"
            aria-hidden="true"
          />
        )}
        <div className="program-drawer-header">
          <span>Program Detail</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <button
              className="refresh-btn"
              aria-pressed={drawerMaximized}
              onClick={() => setDrawerMaximized(v => !v)}
              style={{ fontSize: 12, padding: '5px 10px' }}
              title={drawerMaximized ? 'Restore panel size' : 'Maximize panel'}
            >
              {drawerMaximized ? 'Restore' : 'Maximize'}
            </button>
            <button className="refresh-btn" onClick={onClose} style={{ fontSize: 18, lineHeight: 1, padding: '4px 8px' }}>&times;</button>
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: 24 }}>
        {loading ? (
          <p style={{ color: 'var(--text-muted)' }}>Loading...</p>
        ) : error ? (
          <p style={{ color: 'var(--accent-red)' }}>{error}</p>
        ) : !program ? (
          <p style={{ color: 'var(--accent-red)' }}>Program not found</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Header info */}
            <div>
              <div style={{ fontFamily: 'monospace', fontSize: 13, color: 'var(--accent-blue)', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
                <span>{program.graph_fingerprint}</span>
                {program.stage_at_death && program.stage_at_death !== 'survived' && (
                  <span style={{ fontSize: 11, color: 'var(--accent-red)' }}>
                    died at {program.stage_at_death}
                  </span>
                )}
                {leaderboardEntry && (
                  <>
                    <TierBadge tier={leaderboardEntry.tier} entry={leaderboardEntry} />
                    <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-green)' }}>
                      Score: {Number(leaderboardEntry.composite_score).toFixed(3)}
                    </span>
                  </>
                )}
              </div>
              <StagePipeline program={program} />
            </div>

            {/* Provenance & Context */}
            {(program.experiment_id || linkedHypothesis || leaderboardEntry || linkedCampaign) && (
              <div style={{
                background: 'var(--bg-tertiary)',
                borderRadius: 6,
                border: '1px solid var(--border)',
                overflow: 'hidden',
              }}>
                <div
                  onClick={() => setProvenanceOpen(!provenanceOpen)}
                  style={{
                    padding: '8px 12px',
                    cursor: 'pointer',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    userSelect: 'none',
                  }}
                >
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                    Provenance & Context
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {provenanceOpen ? '▾ collapse' : '▸ expand'}
                  </span>
                </div>
                {provenanceOpen && (
                  <div style={{ padding: '0 12px 12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {/* Source experiment */}
                    {program.experiment_id && (
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90 }}>Experiment:</span>
                        <span style={{ fontFamily: 'monospace', fontSize: 11 }}>{program.experiment_id.slice(0, 12)}</span>
                        {linkedExperiment?.started_at && (
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            {new Date(linkedExperiment.started_at).toLocaleString()}
                          </span>
                        )}
                        {linkedExperiment?.experiment_type && (
                          <span style={{ fontSize: 10, color: 'var(--accent-blue)', border: '1px solid var(--accent-blue)', borderRadius: 3, padding: '0 4px' }}>
                            {linkedExperiment.experiment_type}
                          </span>
                        )}
                      </div>
                    )}

                    {/* Hypothesis 1-liner */}
                    {linkedHypothesis && (
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'baseline', gap: 8 }}>
                        <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90, flexShrink: 0 }}>Hypothesis:</span>
                        <span style={{
                          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 400,
                          color: linkedHypothesis.status === 'confirmed' ? 'var(--accent-green)' :
                                 linkedHypothesis.status === 'refuted' ? 'var(--accent-red)' : 'var(--text-secondary)',
                        }} title={linkedHypothesis.prediction}>
                          [{linkedHypothesis.status}] {linkedHypothesis.prediction}
                        </span>
                      </div>
                    )}

                    {/* Campaign */}
                    {linkedCampaign && (
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90 }}>Campaign:</span>
                        <span>{linkedCampaign.title || linkedCampaign.campaign_id}</span>
                      </div>
                    )}

                    {/* Leaderboard status */}
                    {leaderboardEntry && (
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90 }}>Leaderboard:</span>
                        <TierBadge tier={leaderboardEntry.tier} entry={leaderboardEntry} />
                        <span style={{ fontWeight: 600, color: 'var(--accent-green)' }}>
                          {Number(leaderboardEntry.composite_score).toFixed(3)}
                        </span>
                        {leaderboardEntry.tier === 'screening' && (
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            needs investigation to advance
                          </span>
                        )}
                        {leaderboardEntry.tier === 'investigation' && !leaderboardEntry.investigation_passed && (
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            investigation completed (below threshold)
                          </span>
                        )}
                        {leaderboardEntry.tier === 'investigation' && leaderboardEntry.investigation_passed && (
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            ready for validation
                          </span>
                        )}
                      </div>
                    )}

                    {/* Quick nav links */}
                    <div style={{ display: 'flex', gap: 6, marginTop: 4, flexWrap: 'wrap' }}>
                      {program.experiment_id && onSelectExperiment && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 11, padding: '3px 10px' }}
                          onClick={() => { onClose(); onSelectExperiment(program.experiment_id); }}
                        >
                          Open Experiment
                        </button>
                      )}
                      {onAddToComparison && (
                        <button
                          className="refresh-btn"
                          title="Add to side-by-side comparison"
                          onClick={() => onAddToComparison(resultId)}
                        >
                          Compare
                        </button>
                      )}
                      {onOpenInDesigner && (
                        <button
                          className="refresh-btn"
                          title="Open this architecture in Aria Designer"
                          onClick={() => onOpenInDesigner(resultId)}
                        >
                          Open in Designer
                        </button>
                      )}
                      {leaderboardEntry && onViewInLeaderboard && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 11, padding: '3px 10px' }}
                          onClick={() => { onClose(); onViewInLeaderboard(resultId); }}
                        >
                          View in Leaderboard
                        </button>
                      )}
                      {linkedCampaign && onSelectCampaign && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 11, padding: '3px 10px' }}
                          onClick={() => { onClose(); onSelectCampaign(linkedCampaign.campaign_id); }}
                        >
                          Open Campaign
                        </button>
                      )}
                      <button
                        className="refresh-btn"
                        style={{
                          fontSize: 11, padding: '3px 10px',
                          background: decisionPacket ? 'rgba(188, 140, 255, 0.15)' : undefined,
                          borderColor: 'var(--accent-purple)',
                          color: 'var(--accent-purple)',
                        }}
                        disabled={decisionPacketLoading}
                        onClick={fetchDecisionPacket}
                      >
                        {decisionPacketLoading ? 'Loading...' : 'Decision Packet'}
                      </button>
                      <button
                        className="refresh-btn"
                        style={{ fontSize: 11, padding: '3px 10px' }}
                        disabled={manifestLoading}
                        onClick={fetchAndCopyManifest}
                      >
                        {manifestLoading ? 'Loading...' : manifestCopied ? 'Copied!' : 'Copy Manifest'}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Decision Packet */}
            {decisionPacketError && (
              <div style={{ padding: 8, background: 'rgba(248, 81, 73, 0.1)', border: '1px solid var(--accent-red)', borderRadius: 4, fontSize: 12, color: 'var(--accent-red)' }}>
                {decisionPacketError}
              </div>
            )}
            {decisionPacket && (
              <div style={{
                background: 'var(--bg-tertiary)', borderRadius: 6,
                border: '1px solid var(--accent-purple)', overflow: 'hidden',
              }}>
                <div
                  onClick={() => setDecisionPacketOpen(!decisionPacketOpen)}
                  style={{
                    padding: '8px 12px', cursor: 'pointer',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    userSelect: 'none', background: 'rgba(188, 140, 255, 0.08)',
                  }}
                >
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-purple)', textTransform: 'uppercase' }}>
                    Decision Packet
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {decisionPacketOpen ? '\u25BE collapse' : '\u25B8 expand'}
                  </span>
                </div>
                {decisionPacketOpen && (
                  <div style={{ padding: '8px 12px 12px', display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {/* Evidence Flags */}
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Evidence Flags</div>
                      <EvidenceFlagChips flags={decisionPacket.evidence_flags} />
                    </div>
                    {/* Hypothesis Lineage */}
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Hypothesis Lineage</div>
                      <HypothesisLineage chain={decisionPacket.hypothesis_chain} />
                    </div>
                    {/* Outcomes by Phase */}
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Outcomes by Phase</div>
                      <OutcomesByPhase outcomes={decisionPacket.outcomes} />
                    </div>
                    {/* Failure Context */}
                    {(decisionPacket.failure_context?.stage_at_death || decisionPacket.failure_context?.error_type) && (
                      <div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Failure Context</div>
                        <FailureContext context={decisionPacket.failure_context} />
                      </div>
                    )}
                    {/* Recommendation */}
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Recommendation</div>
                      <RecommendationCard recommendation={decisionPacket.recommendation} />
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Error if failed */}
            {(program.error_message || program.stage0_error) && (
              <div style={{
                padding: 8,
                background: 'rgba(248, 81, 73, 0.1)',
                border: '1px solid var(--accent-red)',
                borderRadius: 4,
                fontSize: 12,
                fontFamily: 'monospace',
                color: 'var(--accent-red)',
              }}>
                {program.error_type && (
                  <span style={{ fontWeight: 600 }}>[{program.error_type}] </span>
                )}
                {program.error_message || program.stage0_error}
              </div>
            )}
            {actionError && (
              <div style={{
                padding: 8,
                background: 'rgba(248, 81, 73, 0.1)',
                border: '1px solid var(--accent-red)',
                borderRadius: 4,
                fontSize: 12,
                color: 'var(--accent-red)',
              }}>
                {actionError}
              </div>
            )}
            {latestRefineLaunch && (
              <div style={{
                padding: 10,
                background: 'var(--bg-tertiary)',
                borderRadius: 6,
                border: '1px solid var(--border)',
                display: 'flex',
                flexDirection: 'column',
                gap: 8,
              }}>
                <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', color: 'var(--accent-purple)' }}>
                  Refinement Trace
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'grid', gap: 4 }}>
                  <div><strong>Intent:</strong> {latestRefineLaunch.intent}</div>
                  <div><strong>Experiment:</strong> <span style={{ fontFamily: 'monospace' }}>{shortId(latestRefineLaunch.experimentId, 16)}</span></div>
                  <div><strong>Source:</strong> <span style={{ fontFamily: 'monospace' }}>{shortId(latestRefineLaunch.sourceResultId, 12)}</span> · {shortId(latestRefineLaunch.sourceFingerprint, 18)}</div>
                  <div><strong>Resolved IDs:</strong> {latestRefineLaunch.resolvedResultIds.length > 0 ? latestRefineLaunch.resolvedResultIds.map(v => shortId(v, 10)).join(', ') : 'none'}</div>
                  {latestRefineLaunch.unresolvedFingerprints.length > 0 && (
                    <div><strong>Unresolved fingerprints:</strong> {latestRefineLaunch.unresolvedFingerprints.join(', ')}</div>
                  )}
                </div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {latestRefineLaunch.experimentId && onSelectExperiment && (
                    <button
                      className="refresh-btn"
                      style={{ fontSize: 11, padding: '3px 10px' }}
                      onClick={() => { onClose(); onSelectExperiment(latestRefineLaunch.experimentId); }}
                    >
                      Open Refinement Run
                    </button>
                  )}
                  {refineTrace?.newResultIds?.[0] && onViewInLeaderboard && (
                    <button
                      className="refresh-btn"
                      style={{ fontSize: 11, padding: '3px 10px' }}
                      onClick={() => { onClose(); onViewInLeaderboard(refineTrace.newResultIds[0]); }}
                    >
                      View Top Refined Result
                    </button>
                  )}
                </div>
                {refineTraceLoading && (
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Collecting live refinement outcomes…</div>
                )}
                {refineTrace?.error && (
                  <div style={{ fontSize: 11, color: 'var(--accent-red)' }}>{refineTrace.error}</div>
                )}
                {refineTrace && !refineTrace.error && (
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
                    <div><strong>Status:</strong> {refineTrace.status}</div>
                    <div>
                      <strong>Outcomes:</strong> {fmtInt(refineTrace.totals?.programs)} programs,
                      {' '}{fmtInt(refineTrace.totals?.stage1Survivors)} S1 survivors,
                      {' '}best loss {fmt(refineTrace.totals?.bestLoss)}
                    </div>
                    {refineTrace.newFingerprints?.length > 0 && (
                      <div>
                        <strong>New Fingerprints:</strong> {refineTrace.newFingerprints.map(fp => shortId(fp, 18)).join(', ')}
                      </div>
                    )}
                    {refineTrace.newCandidates?.length > 0 && (
                      <div>
                        <strong>Open Fingerprint:</strong>{' '}
                        {onViewInLeaderboard ? (
                          <span style={{ display: 'inline-flex', gap: 6, flexWrap: 'wrap', marginTop: 4 }}>
                            {refineTrace.newCandidates.map(candidate => (
                              <button
                                key={candidate.resultId}
                                className="refresh-btn"
                                style={{ fontSize: 10, padding: '2px 8px', fontFamily: 'monospace' }}
                                onClick={() => { onClose(); onViewInLeaderboard(candidate.resultId); }}
                                title={`Open ${candidate.fingerprint}`}
                              >
                                {shortId(candidate.fingerprint, 18)}
                              </button>
                            ))}
                          </span>
                        ) : (
                          refineTrace.newCandidates.map(candidate => shortId(candidate.fingerprint, 18)).join(', ')
                        )}
                      </div>
                    )}
                    {refineTrace.newResultIds?.length > 0 && (
                      <div>
                        <strong>New Result IDs:</strong> {refineTrace.newResultIds.map(rid => shortId(rid, 10)).join(', ')}
                      </div>
                    )}
                  </div>
                )}
                {refineLaunchHistory.length > 0 && (
                  <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>
                      Recent Refinement Launches
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {refineLaunchHistory.map(item => (
                        <div key={item.experimentId} style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', fontSize: 11 }}>
                          <span style={{ color: 'var(--text-secondary)' }}>{item.intent}</span>
                          <span style={{ fontFamily: 'monospace', color: 'var(--text-muted)' }}>{shortId(item.experimentId, 12)}</span>
                          <span style={{ color: 'var(--text-muted)' }}>{item.status || 'running'}</span>
                          {onSelectExperiment && (
                            <button
                              className="refresh-btn"
                              style={{ fontSize: 10, padding: '2px 8px' }}
                              onClick={() => { onClose(); onSelectExperiment(item.experimentId); }}
                            >
                              Open Run
                            </button>
                          )}
                          {item.topCandidate?.resultId && onViewInLeaderboard && (
                            <button
                              className="refresh-btn"
                              style={{ fontSize: 10, padding: '2px 8px', fontFamily: 'monospace' }}
                              onClick={() => { onClose(); onViewInLeaderboard(item.topCandidate.resultId); }}
                              title={`Open ${item.topCandidate.fingerprint}`}
                            >
                              Open Fingerprint
                            </button>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Metrics + Radar side by side */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
                  Core Metrics
                </div>
                <div style={{ fontSize: 13 }}>
                  <MetricRow label="Parameters" value={program.param_count ? `${(program.param_count / 1e6).toFixed(2)}M` : null} />
                  <MetricRow label="Loss Ratio" value={program.loss_ratio != null ?
                    <span style={{
                      color: lossColor(program.loss_ratio),
                      fontWeight: program.loss_ratio < 0.5 ? 600 : 'normal',
                    }} title={program.loss_ratio < 0.5 ? 'Learned quickly — strong candidate' : program.loss_ratio < 0.7 ? 'Moderate learning' : 'Slow learning'}>
                      {fmt(program.loss_ratio)}
                    </span> : null} />
                  <MetricRow label="Final Loss" value={fmt(program.final_loss)} />
                  <MetricRow label="Discovery Loss" value={program.discovery_loss != null ? fmt(program.discovery_loss) : null} />
                  <MetricRow label="Discovery LR" value={program.discovery_loss_ratio != null ? fmt(program.discovery_loss_ratio) : null} />
                  <MetricRow label="Validation Loss" value={program.validation_loss != null ? fmt(program.validation_loss) : null} />
                  <MetricRow label="Validation LR" value={program.validation_loss_ratio != null ? fmt(program.validation_loss_ratio) : null} />
                  <MetricRow label="Gen Gap" value={program.generalization_gap != null ? fmt(program.generalization_gap) : null} />
                  <MetricRow label="Baseline Ratio" value={program.baseline_loss_ratio != null ?
                    <span style={{
                      color: program.baseline_loss_ratio < 1 ? 'var(--accent-green)' : 'var(--accent-red)',
                      fontWeight: program.baseline_loss_ratio < 1 ? 600 : 'normal',
                    }} title={program.baseline_loss_ratio < 1 ? 'Beats a standard transformer!' : 'Underperforms a transformer of same size'}>
                      {fmt(program.baseline_loss_ratio)} {program.baseline_loss_ratio < 1 ? '(beats transformer)' : ''}
                    </span> : null} />
                  <MetricRow label="Throughput" value={program.throughput_tok_s != null ? `${Number(program.throughput_tok_s).toFixed(0)} tok/s` : null} />
                  <MetricRow label="Novelty" value={program.novelty_score != null ?
                    <span style={{
                      color: noveltyColor(program.novelty_score),
                    }} title={program.novelty_score > 0.8 ? 'Very different from known architectures' : program.novelty_score > 0.5 ? 'Moderately novel' : 'Similar to existing architectures'}>
                      {fmt(program.novelty_score, 3)}
                    </span> : null} />
                  <MetricRow label="Similar To" value={program.most_similar_to} />
                </div>

                <BenchmarkEvidenceSnapshot program={program} leaderboardEntry={leaderboardEntry} />
                <RobustnessProfile program={program} leaderboardEntry={leaderboardEntry} />
                <AriaAdvice analysis={refineAnalysis} />
                <ReferenceComparison program={program} leaderboardEntry={leaderboardEntry} />
                <ExternalBenchmarkCard program={program} />

                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8, marginTop: 12 }}>
                  Sandbox Timing
                </div>
                <div style={{ fontSize: 13 }}>
                  <MetricRow label="Compile" value={fmtMs(program.compile_time_ms)} />
                  <MetricRow label="Forward" value={fmtMs(program.forward_time_ms)} />
                  <MetricRow label="Backward" value={fmtMs(program.backward_time_ms)} />
                  <MetricRow label="Peak Memory" value={fmtMem(program.peak_memory_mb)} />
                  <MetricRow label="FLOPs (fwd)" value={program.flops_forward ? fmtInt(program.flops_forward) : null} />
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
                  Fingerprint Radar
                </div>
                <RadarChart program={program} size={260} />
              </div>
            </div>

            {/* CKA Similarity bars */}
            {(program.fp_cka_vs_transformer != null || program.fp_cka_vs_ssm != null) && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
                    CKA Similarity to Known Architectures
                  </div>
                  {(program.cka_source || program.cka_artifact_version) && (
                    <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                      {program.cka_source && (
                        <span style={{
                          fontSize: 10,
                          fontWeight: 600,
                          padding: '2px 6px',
                          borderRadius: 4,
                          background: program.cka_source === 'artifact' ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                          color: program.cka_source === 'artifact' ? 'var(--accent-green)' : 'var(--accent-red)',
                        }}>
                          {program.cka_source === 'artifact' ? 'Artifact CKA' : 'Fallback CKA'}
                        </span>
                      )}
                      {program.cka_artifact_version && (
                        <span style={{
                          fontSize: 10,
                          padding: '2px 6px',
                          borderRadius: 4,
                          background: 'var(--bg-tertiary)',
                          color: 'var(--text-muted)',
                        }}>
                          {program.cka_artifact_version}
                        </span>
                      )}
                    </div>
                  )}
                </div>
                {[
                  { label: 'Transformer', value: program.fp_cka_vs_transformer, color: 'var(--accent-blue)' },
                  { label: 'SSM', value: program.fp_cka_vs_ssm, color: 'var(--accent-green)' },
                  { label: 'Conv', value: program.fp_cka_vs_conv, color: 'var(--accent-yellow)' },
                ].map(({ label, value, color }) => value != null && (
                  <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: 12, color: 'var(--text-secondary)', minWidth: 80 }}>{label}</span>
                    <div style={{ flex: 1, height: 12, background: 'var(--bg-tertiary)', borderRadius: 3 }}>
                      <div style={{
                        width: `${Math.min(value, 1) * 100}%`, height: '100%',
                        background: color, borderRadius: 3, opacity: 0.6,
                      }} />
                    </div>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', minWidth: 40 }}>{(Number(value) * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            )}

            {/* Training metrics */}
            {program.initial_loss != null && (
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
                  Training Metrics
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 13 }}>
                  <MetricRow label="Initial Loss" value={fmt(program.initial_loss)} />
                  <MetricRow label="Min Loss" value={fmt(program.min_loss)} />
                  <MetricRow label="Steps" value={program.n_train_steps} />
                  <MetricRow label="Avg Step Time" value={fmtMs(program.avg_step_time_ms)} />
                  <MetricRow label="Mean Grad Norm" value={fmt(program.mean_grad_norm, 3)} />
                  <MetricRow label="Max Grad Norm" value={fmt(program.max_grad_norm, 3)} />
                </div>
              </div>
            )}

            {/* Training Curve */}
            {program.has_training_curve && (
              <TrainingCurve resultId={resultId} />
            )}

            {/* LLM Explanation */}
            {program.llm_explanation && (
              <div style={{
                padding: 12,
                background: 'var(--bg-tertiary)',
                borderRadius: 4,
                borderLeft: '2px solid var(--accent-purple)',
                fontSize: 13,
                color: 'var(--text-secondary)',
                fontStyle: 'italic',
              }}>
                <div style={{ fontSize: 11, color: 'var(--accent-purple)', marginBottom: 4, fontWeight: 600, fontStyle: 'normal' }}>
                  ARIA'S ANALYSIS
                </div>
                {program.llm_explanation}
              </div>
            )}

            {/* Linked Hypothesis */}
            {linkedHypothesis && (
              <HypothesisInfo hypothesis={linkedHypothesis} />
            )}

            {/* Linked Decision */}
            {linkedDecision && (
              <div style={{
                padding: 12, background: 'var(--bg-tertiary)', borderRadius: 4,
                borderLeft: `2px solid ${
                  linkedDecision.decision_type === 'go' ? 'var(--accent-green)' :
                  linkedDecision.decision_type === 'no_go' ? 'var(--accent-red)' : 'var(--accent-yellow)'
                }`,
                fontSize: 13,
              }}>
                <div style={{
                  fontSize: 11, fontWeight: 600, textTransform: 'uppercase', marginBottom: 4,
                  color: linkedDecision.decision_type === 'go' ? 'var(--accent-green)' :
                         linkedDecision.decision_type === 'no_go' ? 'var(--accent-red)' : 'var(--accent-yellow)',
                }}>
                  Decision: {linkedDecision.decision_type?.replace('_', ' ')}
                </div>
                <div>{linkedDecision.rationale}</div>
              </div>
            )}

            <RefinementRationale program={program} />
            <RefinementLineage program={program} onViewInLeaderboard={onViewInLeaderboard} />

            {program.stage1_passed && (
              <RefinementAdvisor
                analysis={refineAnalysis}
                loading={refineAnalysisLoading}
                error={refineAnalysisError}
                onLaunchRefinement={handleLaunchRefinement}
                actionStarting={actionStarting}
              />
            )}

            {/* Scale Up Button (only for S1 survivors) */}
            {program.stage1_passed && (
              <div style={{
                padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6,
                border: '1px solid var(--border)',
              }}>
                {!scaleUpOpen ? (
                  <button
                    className="start-btn"
                    onClick={() => setScaleUpOpen(true)}
                    style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(88, 166, 255, 0.15)', border: '1px solid rgba(88, 166, 255, 0.4)', color: 'var(--accent-blue)' }}
                  >
                    Scale Up This Architecture
                  </button>
                ) : (
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--text-secondary)' }}>
                      Scale-Up Configuration
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 8 }}>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Steps</label>
                        <input type="number" min="1000" max="50000" step="1000"
                          value={scaleUpConfig.steps}
                          onChange={e => setScaleUpConfig(c => ({ ...c, steps: parseInt(e.target.value) || 5000 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Batch Size</label>
                        <input type="number" min="4" max="16" step="1"
                          value={scaleUpConfig.batch_size}
                          onChange={e => setScaleUpConfig(c => ({ ...c, batch_size: parseInt(e.target.value) || 8 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Seq Length</label>
                        <input type="number" min="256" max="1024" step="128"
                          value={scaleUpConfig.seq_len}
                          onChange={e => setScaleUpConfig(c => ({ ...c, seq_len: parseInt(e.target.value) || 512 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button
                        className="start-btn"
                        disabled={scaleUpStarting}
                        onClick={async () => {
                          setScaleUpStarting(true);
                          try {
                            setActionError(null);
                            const res = await apiCall(`/api/experiments/start`, {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({
                                mode: 'scale_up',
                                result_ids: [resultId],
                                scale_up_steps: scaleUpConfig.steps,
                                scale_up_batch_size: scaleUpConfig.batch_size,
                                scale_up_seq_len: scaleUpConfig.seq_len,
                              }),
                            });
                            if (!res.ok) {
                              const err = await res.json();
                              setActionError(err.error || 'Failed to start scale-up');
                            } else {
                              setScaleUpOpen(false);
                              if (onActionComplete) onActionComplete();
                              onClose();
                            }
                          } catch (e) {
                            setActionError('Error: ' + e.message);
                          }
                          setScaleUpStarting(false);
                        }}
                        style={{ padding: '6px 16px', fontSize: 12 }}
                      >
                        {scaleUpStarting ? 'Starting...' : 'Start Scale-Up'}
                      </button>
                      <button
                        className="refresh-btn"
                        onClick={() => setScaleUpOpen(false)}
                        style={{ padding: '6px 12px', fontSize: 12 }}
                      >
                        Cancel
                      </button>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
                      Trains for {scaleUpConfig.steps} steps with batch={scaleUpConfig.batch_size}, seq={scaleUpConfig.seq_len}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Manual Training Run (power-user override) */}
            {program.stage1_passed && (
              <div style={{
                padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6,
                border: '1px solid var(--border)',
              }}>
                {!manualRunOpen ? (
                  <button
                    className="start-btn"
                    onClick={() => setManualRunOpen(true)}
                    style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(210, 153, 34, 0.15)', border: '1px solid rgba(210, 153, 34, 0.4)', color: 'var(--accent-yellow)' }}
                  >
                    Manual Training Run
                  </button>
                ) : (
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--text-secondary)' }}>
                      Manual Training Configuration
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 8, marginBottom: 8 }}>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Steps</label>
                        <input type="number" min="500" max="50000" step="500"
                          value={manualRunConfig.steps}
                          onChange={e => setManualRunConfig(c => ({ ...c, steps: parseInt(e.target.value) || 2500 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Batch Size</label>
                        <input type="number" min="1" max="32" step="1"
                          value={manualRunConfig.batch_size}
                          onChange={e => setManualRunConfig(c => ({ ...c, batch_size: parseInt(e.target.value) || 4 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Seq Length</label>
                        <input type="number" min="64" max="2048" step="64"
                          value={manualRunConfig.seq_len}
                          onChange={e => setManualRunConfig(c => ({ ...c, seq_len: parseInt(e.target.value) || 256 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Training Programs</label>
                        <input type="number" min="1" max="10" step="1"
                          value={manualRunConfig.n_training_programs}
                          onChange={e => setManualRunConfig(c => ({ ...c, n_training_programs: parseInt(e.target.value) || 3 }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        />
                      </div>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 8 }}>
                      <div>
                        <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Data Source</label>
                        <select
                          value={manualRunConfig.data_source}
                          onChange={e => setManualRunConfig(c => ({ ...c, data_source: e.target.value }))}
                          style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                        >
                          <option value="corpus">Corpus</option>
                          <option value="random">Random</option>
                          <option value="huggingface">HuggingFace</option>
                        </select>
                      </div>
                      {manualRunConfig.data_source === 'huggingface' && (
                        <>
                          <div>
                            <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>HF Dataset</label>
                            <input type="text"
                              value={manualRunConfig.hf_dataset}
                              onChange={e => setManualRunConfig(c => ({ ...c, hf_dataset: e.target.value }))}
                              placeholder="roneneldan/TinyStories"
                              style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                            />
                          </div>
                          <div>
                            <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>HF Subset</label>
                            <input type="text"
                              value={manualRunConfig.hf_subset}
                              onChange={e => setManualRunConfig(c => ({ ...c, hf_subset: e.target.value }))}
                              placeholder="(optional)"
                              style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                            />
                          </div>
                        </>
                      )}
                    </div>
                    <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
                      <button className="refresh-btn" style={{ padding: '3px 8px', fontSize: 11 }}
                        onClick={() => setManualRunConfig(c => ({ ...c, steps: 1000, batch_size: 4, n_training_programs: 1, seq_len: 256 }))}>
                        Quick
                      </button>
                      <button className="refresh-btn" style={{ padding: '3px 8px', fontSize: 11 }}
                        onClick={() => setManualRunConfig(c => ({ ...c, steps: 2500, batch_size: 4, n_training_programs: 3, seq_len: 256 }))}>
                        Standard
                      </button>
                      <button className="refresh-btn" style={{ padding: '3px 8px', fontSize: 11 }}
                        onClick={() => setManualRunConfig(c => ({ ...c, steps: 5000, batch_size: 8, n_training_programs: 5, seq_len: 512 }))}>
                        Deep
                      </button>
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button
                        className="start-btn"
                        disabled={manualRunStarting}
                        onClick={async () => {
                          setManualRunStarting(true);
                          try {
                            setActionError(null);
                            const body = {
                              mode: 'investigation',
                              force: true,
                              result_ids: [resultId],
                              n_training_programs: manualRunConfig.n_training_programs,
                              investigation_steps: manualRunConfig.steps,
                              investigation_batch_size: manualRunConfig.batch_size,
                              max_seq_len: manualRunConfig.seq_len,
                              data_mode: manualRunConfig.data_source,
                            };
                            if (manualRunConfig.data_source === 'huggingface') {
                              body.hf_dataset = manualRunConfig.hf_dataset;
                              body.hf_subset = manualRunConfig.hf_subset;
                            }
                            const res = await apiCall(`/api/experiments/start`, {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify(body),
                            });
                            if (!res.ok) {
                              const err = await res.json();
                              setActionError(err.error || 'Failed to start manual run');
                            } else {
                              setManualRunOpen(false);
                              if (onActionComplete) onActionComplete();
                              onClose();
                            }
                          } catch (e) {
                            setActionError('Error: ' + e.message);
                          }
                          setManualRunStarting(false);
                        }}
                        style={{ padding: '6px 16px', fontSize: 12 }}
                      >
                        {manualRunStarting ? 'Starting...' : 'Launch Manual Run'}
                      </button>
                      <button
                        className="refresh-btn"
                        onClick={() => setManualRunOpen(false)}
                        style={{ padding: '6px 12px', fontSize: 12 }}
                      >
                        Cancel
                      </button>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
                      {manualRunConfig.n_training_programs} program(s), {manualRunConfig.steps} steps, batch={manualRunConfig.batch_size}, seq={manualRunConfig.seq_len}, data={manualRunConfig.data_source}
                      {manualRunConfig.data_source === 'huggingface' && manualRunConfig.hf_dataset && ` (${manualRunConfig.hf_dataset})`}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Recompute Missing Metrics */}
            {program.stage1_passed && (() => {
              const metrics = [
                { key: 'novelty_score', label: 'Novelty' },
                { key: 'fp_jacobian_spectral_norm', label: 'Spectral Norm' },
                { key: 'fp_interaction_locality', label: 'Locality' },
                { key: 'fp_interaction_sparsity', label: 'Sparsity' },
                { key: 'fp_isotropy', label: 'Isotropy' },
                { key: 'fp_rank_ratio', label: 'Rank Ratio' },
                { key: 'fp_sensitivity_uniformity', label: 'Sensitivity' },
              ];
              const missing = metrics.filter(m => program[m.key] == null);
              const lbMissing = leaderboardEntry ? [
                { key: 'robustness_noise_score', label: 'Noise Robustness' },
                { key: 'quant_int8_retention', label: 'INT8 Quantization' },
                { key: 'init_sensitivity_std', label: 'Init Sensitivity' },
                { key: 'param_efficiency', label: 'Param Efficiency' },
              ].filter(m => leaderboardEntry[m.key] == null) : [];
              const allMissing = [...missing, ...lbMissing];
              if (allMissing.length === 0 && !backfillResult) return null;
              return (
                <div style={{
                  padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6,
                  border: '1px solid var(--border)',
                }}>
                  {allMissing.length > 0 && (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                      Missing: {allMissing.map(m => m.label).join(', ')}
                    </div>
                  )}
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <button
                      className="start-btn"
                      disabled={backfillRunning}
                      onClick={async () => {
                        setBackfillRunning(true);
                        setBackfillResult(null);
                        try {
                          setActionError(null);
                          const res = await apiCall(`/api/programs/${resultId}/backfill-metrics`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ device: 'cpu' }),
                          });
                          if (!res.ok) {
                            const err = await res.json();
                            setActionError(err.error || 'Backfill failed');
                            setBackfillResult({ status: 'error' });
                          } else {
                            const data = await res.json();
                            setBackfillResult(data.backfill || { status: 'ok' });
                          }
                        } catch (e) {
                          setActionError('Error: ' + e.message);
                          setBackfillResult({ status: 'error' });
                        }
                        setBackfillRunning(false);
                      }}
                      style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(139, 92, 246, 0.15)', border: '1px solid rgba(139, 92, 246, 0.4)', color: '#a78bfa' }}
                    >
                      {backfillRunning ? 'Computing...' : 'Recompute Missing Metrics'}
                    </button>
                    {backfillResult && backfillResult.status === 'ok' && (
                      <span style={{ fontSize: 11, color: 'var(--accent-green)' }}>Done — reload to see updates</span>
                    )}
                    {backfillResult && backfillResult.status === 'error' && (
                      <span style={{ fontSize: 11, color: 'var(--accent-red)' }}>Failed</span>
                    )}
                  </div>
                  {(program.discovery_loss_ratio == null || program.validation_loss_ratio == null) && (
                    <div style={{ marginTop: 8 }}>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
                        Missing loss:{' '}
                        {[
                          program.discovery_loss_ratio == null && 'Discovery',
                          program.validation_loss_ratio == null && 'Validation',
                        ].filter(Boolean).join(', ')}
                      </div>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        <button
                          className="start-btn"
                          disabled={lossBackfillRunning}
                          onClick={async () => {
                            setLossBackfillRunning(true);
                            setLossBackfillResult(null);
                            try {
                              setActionError(null);
                              const res = await apiCall(`/api/programs/${resultId}/backfill-loss`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ device: 'cpu' }),
                              });
                              if (!res.ok) {
                                const err = await res.json();
                                setActionError(err.error || 'Loss backfill failed');
                                setLossBackfillResult({ status: 'error' });
                              } else {
                                const data = await res.json();
                                setLossBackfillResult(data.updates || { status: 'ok' });
                              }
                            } catch (e) {
                              setActionError('Error: ' + e.message);
                              setLossBackfillResult({ status: 'error' });
                            }
                            setLossBackfillRunning(false);
                          }}
                          style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(139, 92, 246, 0.15)', border: '1px solid rgba(139, 92, 246, 0.4)', color: '#a78bfa' }}
                        >
                          {lossBackfillRunning ? 'Evaluating...' : 'Compute Discovery & Validation Loss'}
                        </button>
                        {lossBackfillResult && !lossBackfillResult.status && (
                          <span style={{ fontSize: 11, color: 'var(--accent-green)' }}>
                            {lossBackfillResult.discovery_loss_ratio != null && `D.LR: ${Number(lossBackfillResult.discovery_loss_ratio).toFixed(4)}`}
                            {lossBackfillResult.discovery_loss_ratio != null && lossBackfillResult.validation_loss_ratio != null && ' | '}
                            {lossBackfillResult.validation_loss_ratio != null && `V.LR: ${Number(lossBackfillResult.validation_loss_ratio).toFixed(4)}`}
                          </span>
                        )}
                        {lossBackfillResult && lossBackfillResult.status === 'error' && (
                          <span style={{ fontSize: 11, color: 'var(--accent-red)' }}>Failed</span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}

            {/* Investigate / Validate actions */}
            {program.stage1_passed && (
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {resolvedEligibility.investigationEligible && (
                  <button
                    className="start-btn"
                    disabled={actionStarting === 'investigate'}
                    onClick={async () => {
                      setActionStarting('investigate');
                      try {
                        setActionError(null);
                        const res = await apiCall(`/api/experiments/start`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ mode: 'investigation', result_ids: [resultId] }),
                        });
                        if (!res.ok) {
                          const err = await res.json();
                          setActionError(err.error || 'Failed to start investigation');
                        } else {
                          if (onActionComplete) onActionComplete();
                          onClose();
                        }
                      } catch (e) {
                        setActionError('Error: ' + e.message);
                      }
                      setActionStarting(null);
                    }}
                    style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(63, 185, 80, 0.15)', border: '1px solid rgba(63, 185, 80, 0.4)', color: 'var(--accent-green)' }}
                    title="Deep study with multiple training programs"
                  >
                    {actionStarting === 'investigate' ? 'Starting...' : 'Investigate'}
                  </button>
                )}
                {!resolvedEligibility.investigationEligible && (leaderboardEntry?.tier === 'screening') && (
                  <span style={{
                    fontSize: 11,
                    padding: '4px 8px',
                    borderRadius: 4,
                    background: 'rgba(210,153,34,0.12)',
                    color: 'var(--accent-yellow)',
                  }} title="Candidate already has investigation evidence; wait for changed conditions before re-investigating">
                    Already investigated
                  </span>
                )}
                {resolvedEligibility.validationEligible && (
                  <button
                    className="start-btn"
                    disabled={actionStarting === 'validate'}
                    onClick={async () => {
                      setActionStarting('validate');
                      try {
                        setActionError(null);
                        const res = await apiCall(`/api/experiments/start`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ mode: 'validation', result_ids: [resultId] }),
                        });
                        if (!res.ok) {
                          const err = await res.json();
                          setActionError(err.error || 'Failed to start validation');
                        } else {
                          if (onActionComplete) onActionComplete();
                          onClose();
                        }
                      } catch (e) {
                        setActionError('Error: ' + e.message);
                      }
                      setActionStarting(null);
                    }}
                    style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(188, 140, 255, 0.15)', border: '1px solid rgba(188, 140, 255, 0.4)', color: 'var(--accent-purple)' }}
                    title="Publication-grade multi-seed validation"
                  >
                    {actionStarting === 'validate' ? 'Starting...' : 'Validate'}
                  </button>
                )}
              </div>
            )}

            {/* Leaderboard training program details */}
            {leaderboardEntry?.investigation_best_training && (
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
                  Best Training Program (from Investigation)
                </div>
                <pre style={{
                  fontSize: 11, padding: 8, background: 'var(--bg-tertiary)',
                  borderRadius: 4, overflow: 'auto', maxHeight: 120,
                  color: 'var(--text-secondary)',
                }}>
                  {typeof leaderboardEntry.investigation_best_training === 'string'
                    ? leaderboardEntry.investigation_best_training
                    : JSON.stringify(leaderboardEntry.investigation_best_training, null, 2)}
                </pre>
              </div>
            )}

            <TokenMixingTaxonomy graphJson={program.graph_json_parsed} />
            <GatingDiagnostics program={program} />
            <SparsityDiagnostics program={program} />

            {/* Open in Designer action */}
            {program.graph_json_parsed && onOpenInDesigner && (
              <div style={{ marginTop: 8 }}>
                <button
                  onClick={() => {
                    onClose();
                    onOpenInDesigner(resultId);
                  }}
                  style={{
                    background: 'rgba(188, 140, 255, 0.15)',
                    border: '1px solid rgba(188, 140, 255, 0.4)',
                    color: 'var(--accent-purple)',
                    fontSize: 12,
                    fontWeight: 600,
                    padding: '8px 20px',
                    borderRadius: 6,
                    cursor: 'pointer',
                    width: '100%',
                  }}
                  title="Open this architecture in the visual graph designer"
                >
                  Open in Designer
                </button>
              </div>
            )}
          </div>
        )}
        </div>
      </div>
    </div>
  );
}

export default ProgramDetail;
