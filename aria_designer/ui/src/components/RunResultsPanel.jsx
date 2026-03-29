import { useState, memo } from 'react'
import { CheckCircle2, Loader, Circle, XCircle, ChevronRight } from 'lucide-react'
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import ChartActionRail from './ChartActionRail.jsx'
import OpProfileTable from './RunResults/OpProfileTable.jsx'
import BenchmarkTargets from './RunResults/BenchmarkTargets.jsx'
import { formatNum } from '../utils/format'

const STAGE_ORDER = ['conversion', 'profiling', 'compilation', 'sandbox', 'compression', 'fingerprint', 'novelty']

const STAGE_LABELS = {
  conversion: 'Conversion',
  profiling: 'Profiling',
  compilation: 'Compilation',
  sandbox: 'Sandbox Eval',
  compression: 'Compression',
  fingerprint: 'Fingerprint',
  novelty: 'Novelty',
}

const CATEGORY_COLORS = [
  '#17a3ff', '#24d1a0', '#a060ff', '#f0a020', '#ff6090',
  '#20c0f0', '#c060c0', '#ff8040', '#e0c040', '#60c060',
]

function scoreTone(score) {
  if (score == null) return 'muted'
  if (score >= 120) return 'strong'
  if (score >= 70) return 'promising'
  return 'weak'
}

function StageIcon({ status }) {
  if (status === 'done') return <CheckCircle2 size={14} color="#24d1a0" />
  if (status === 'running') return <Loader size={14} />
  if (status === 'error') return <XCircle size={14} color="#ff5050" />
  return <Circle size={14} />
}

function CollapsibleSection({ title, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div style={{ marginBottom: 8 }}>
      <button type="button" className="section-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className={`chevron ${open ? 'open' : ''}`}><ChevronRight size={12} /></span>
        {title}
      </button>
      {open && children}
    </div>
  )
}

function ProgressBar({ value, color = 'var(--accent)' }) {
  const pct = Math.max(0, Math.min(100, (value || 0) * 100))
  return (
    <div className="fp-bar-bg">
      <div className="fp-bar" style={{ width: `${pct}%`, background: color }} />
    </div>
  )
}

function RunResultsPanel({
  evalState,
  baseline = null,
  benchmarkObserved = {},
  onBenchmarkObservedChange = null,
}) {
  const { stages = [], totalTimeMs, error, compositeScore, graphFingerprint, discoveryUrl } = evalState || {}

  const stageMap = {}
  for (const s of stages) stageMap[s.stage] = s

  const sandboxMetrics = stageMap.sandbox?.metrics
  const profilingMetrics = stageMap.profiling?.metrics
  const compressionMetrics = stageMap.compression?.metrics
  const fingerprintMetrics = stageMap.fingerprint?.metrics
  const noveltyMetrics = stageMap.novelty?.metrics
  const benchmarkMetrics = evalState?.benchmarking || stageMap.benchmarking?.metrics || null
  const notMeasuredTargets = Array.isArray(benchmarkMetrics?.targets)
    ? benchmarkMetrics.targets.filter((t) => t.status === 'not_measured')
    : []
  const abiProbe = sandboxMetrics?.native_abi_probe || null
  const abiParityAttempted = Boolean(abiProbe?.parity_attempted)
  const abiParityPass = abiProbe?.parity_pass
  const abiPrimaryUsed = Boolean(abiProbe?.primary_used)
  const abiMode = abiProbe?.mode || (abiPrimaryUsed ? 'primary_forward_only' : 'probe_only')
  const abiParityMaxAbs = Number(abiProbe?.parity_max_abs_diff)
  const abiParityThreshold = Number(abiProbe?.parity_max_abs_threshold)
  const abiSampleRate = Number(abiProbe?.parity_sample_rate)
  const abiParityMaxAbsText = Number.isFinite(abiParityMaxAbs) ? abiParityMaxAbs.toExponential(2) : '-'
  const abiParityThresholdText = Number.isFinite(abiParityThreshold) ? abiParityThreshold.toExponential(2) : '-'
  const abiSampleRateText = Number.isFinite(abiSampleRate) ? `${Math.round(abiSampleRate * 100)}%` : '-'
  const abiParityState = abiParityAttempted
    ? (abiParityPass ? 'pass' : 'fail')
    : (abiPrimaryUsed ? 'primary' : 'probe')

  // FLOPs by category chart data
  const chartData = Object.entries(profilingMetrics?.flops_by_category || {})
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)

  // Sorted op profiles
  const opProfiles = profilingMetrics?.op_profiles || []
  const topBottleneck = opProfiles[0] || null
  const benchmarkSrc = benchmarkObserved && typeof benchmarkObserved === 'object' ? benchmarkObserved : {}
  const benchmarkInputCount = Object.entries(benchmarkSrc).filter(([, v]) => Number.isFinite(Number(v))).length

  if (!evalState || stages.length === 0) {
    return (
      <div className="eval-results" style={{ color: 'var(--muted)', padding: 16, textAlign: 'center' }}>
        Click <strong>Deep Run</strong> to stream evaluation results.
      </div>
    )
  }

  return (
    <div className="eval-results">
      {(compositeScore != null || graphFingerprint) && (
        <div className="score-discovery-section">
          <div>
            <div className="score-discovery-label">Score & Discovery</div>
            <div className={`composite-score-badge tone-${scoreTone(compositeScore)}`}>
              {compositeScore != null ? Number(compositeScore).toFixed(1) : '-'}
            </div>
          </div>
          <div className="score-discovery-actions">
            {graphFingerprint && (
              <a
                className="run-fingerprint-link"
                href={discoveryUrl || `http://localhost:5000/?search=${encodeURIComponent(graphFingerprint)}`}
                target="_blank"
                rel="noreferrer"
              >
                {graphFingerprint}
              </a>
            )}
            {graphFingerprint && (
              <button
                type="button"
                className="compare-run-button"
                onClick={() => window.open(discoveryUrl || `http://localhost:5000/?search=${encodeURIComponent(graphFingerprint)}`, '_blank', 'noopener,noreferrer')}
              >
                Compare
              </button>
            )}
          </div>
        </div>
      )}

      {/* Progress Stepper */}
      <div className="eval-stepper">
        {STAGE_ORDER.map((name) => {
          const s = stageMap[name]
          const st = s?.status || 'pending'
          return (
            <div key={name} className={`eval-step stage-${st}`}>
              <span className="step-icon"><StageIcon status={st} /></span>
              <span className="step-name">{STAGE_LABELS[name]}</span>
              {s?.elapsed_ms != null && <span className="step-time">{s.elapsed_ms.toFixed(0)}ms</span>}
            </div>
          )
        })}
      </div>

      {totalTimeMs != null && (
        <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 10, textAlign: 'right' }}>
          Total: {(totalTimeMs / 1000).toFixed(2)}s
        </div>
      )}

      {error && (
        <div style={{ color: 'var(--danger)', fontSize: 12, marginBottom: 10, padding: '6px 8px', background: 'rgba(255,80,80,0.1)', borderRadius: 6 }}>
          {error}
        </div>
      )}

      {baseline && (
        <CollapsibleSection title="Original Baseline" defaultOpen>
          <div style={{ marginBottom: 8, fontSize: 11, color: 'var(--muted)' }}>
            Imported from result <strong style={{ color: 'var(--text)' }}>{baseline.resultId}</strong>
          </div>
          <div className="metrics-grid">
            <div className="stat">
              <div className="stat-val">{baseline.lossRatio != null ? baseline.lossRatio.toFixed(4) : '-'}</div>
              <div className="stat-label">Loss Ratio</div>
            </div>
            <div className="stat">
              <div className="stat-val">{baseline.validationLossRatio != null ? baseline.validationLossRatio.toFixed(4) : '-'}</div>
              <div className="stat-label">Validation LR</div>
            </div>
            <div className="stat">
              <div className="stat-val">{baseline.discoveryLossRatio != null ? baseline.discoveryLossRatio.toFixed(4) : '-'}</div>
              <div className="stat-label">Discovery LR</div>
            </div>
            <div className="stat">
              <div className="stat-val">{baseline.noveltyScore != null ? baseline.noveltyScore.toFixed(3) : '-'}</div>
              <div className="stat-label">Novelty</div>
            </div>
          </div>
          {baseline.benchmarkScore == null && (
            <div style={{ marginTop: 6, fontSize: 10, color: 'var(--muted)' }}>
              Benchmark score was not stored for the imported baseline, so direct score-delta is unavailable.
            </div>
          )}
        </CollapsibleSection>
      )}

      {/* Summary Metrics Grid */}
      {sandboxMetrics && (
        <CollapsibleSection title="Summary" defaultOpen>
          <div className="metrics-grid">
            <div className="stat">
              <div className="stat-val">{formatNum(sandboxMetrics.param_count)}</div>
              <div className="stat-label">Params</div>
            </div>
            <div className="stat">
              <div className="stat-val">{formatNum(profilingMetrics?.total_flops_per_token)}</div>
              <div className="stat-label">FLOPs/tok</div>
            </div>
            <div className="stat">
              <div className="stat-val">{
                sandboxMetrics.peak_memory_mb
                  ? formatNum(sandboxMetrics.peak_memory_mb) + 'MB'
                  : profilingMetrics?.total_memory_bytes
                    ? formatNum(profilingMetrics.total_memory_bytes / (1024 * 1024)) + 'MB'
                    : '-'
              }</div>
              <div className="stat-label">Memory</div>
            </div>
            <div className="stat">
              <div className="stat-val">{sandboxMetrics.forward_ms?.toFixed(1)}</div>
              <div className="stat-label">Fwd ms</div>
            </div>
            <div className="stat">
              <div className="stat-val">{sandboxMetrics.backward_ms?.toFixed(1)}</div>
              <div className="stat-label">Bwd ms</div>
            </div>
            <div className="stat">
              <div className="stat-val">{sandboxMetrics.stability_score?.toFixed(2)}</div>
              <div className="stat-label">Stability</div>
            </div>
          </div>
          {abiProbe && (
            <div style={{ marginTop: 8, padding: '8px 10px', border: '1px solid #1f3147', borderRadius: 8, background: '#0f1928' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                <strong style={{ fontSize: 12, color: '#d8e6f5' }}>ABI Gate</strong>
                <span
                  style={{
                    fontSize: 10,
                    padding: '2px 8px',
                    borderRadius: 999,
                    border: `1px solid ${
                      abiParityState === 'pass' ? '#24d1a0'
                        : abiParityState === 'fail' ? '#ff5050'
                          : '#17a3ff'
                    }`,
                    color: abiParityState === 'pass' ? '#24d1a0'
                      : abiParityState === 'fail' ? '#ff5050'
                        : '#17a3ff',
                    background: abiParityState === 'pass' ? 'rgba(36,209,160,0.12)'
                      : abiParityState === 'fail' ? 'rgba(255,80,80,0.12)'
                        : 'rgba(23,163,255,0.12)',
                  }}
                >
                  {abiParityState === 'pass'
                    ? 'parity pass'
                    : abiParityState === 'fail'
                      ? 'parity fail'
                      : abiParityState === 'primary'
                        ? 'primary (no parity sample)'
                        : 'probe only'}
                </span>
              </div>
              <div
                style={{ marginTop: 4, fontSize: 10, color: '#8fa8c2' }}
                title="Parity checks compare ABI logits vs torch forward logits on sampled runs. Pass when max abs drift is <= threshold."
              >
                Legend: pass if sampled parity max_abs &le; threshold; fail otherwise.
              </div>
              <div style={{ marginTop: 6, fontSize: 11, color: '#8fa8c2', lineHeight: 1.5 }}>
                <div>Mode: <strong style={{ color: '#d8e6f5' }}>{abiMode}</strong></div>
                <div>Sample rate: <strong style={{ color: '#d8e6f5' }}>{abiSampleRateText}</strong></div>
                <div>Max abs drift: <strong style={{ color: '#d8e6f5' }}>{abiParityMaxAbsText}</strong></div>
                <div>Threshold: <strong style={{ color: '#d8e6f5' }}>{abiParityThresholdText}</strong></div>
                {abiProbe?.parity_reason ? (
                  <div>Reason: <strong style={{ color: '#d8e6f5' }}>{String(abiProbe.parity_reason)}</strong></div>
                ) : null}
              </div>
            </div>
          )}
        </CollapsibleSection>
      )}

      {/* FLOPs by Category Chart */}
      {chartData.length > 0 && (
        <CollapsibleSection title="FLOPs by Category">
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={140}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 60, right: 10, top: 4, bottom: 4 }}>
                <XAxis type="number" tick={{ fill: '#8fa8c2', fontSize: 10 }} tickFormatter={formatNum} />
                <YAxis type="category" dataKey="name" tick={{ fill: '#d8e6f5', fontSize: 11 }} width={55} />
                <Tooltip
                  contentStyle={{ background: '#101b2b', border: '1px solid #1f3147', borderRadius: 6, fontSize: 12 }}
                  formatter={(v) => formatNum(v)}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {chartData.map((_, i) => (
                    <Cell key={i} fill={CATEGORY_COLORS[i % CATEGORY_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <ChartActionRail
              insight={topBottleneck ? `Highest-cost op: ${topBottleneck.op_name}` : null}
              recommendation={topBottleneck ? 'Next best action: inspect the top FLOP consumer and compare its native-kernel coverage before re-running.' : 'Next best action: review the heaviest compute category before re-running.'}
            />
          </div>
        </CollapsibleSection>
      )}

      {/* Per-Op Profile Table */}
      {opProfiles.length > 0 && (
        <CollapsibleSection title={`Per-Op Profile (${opProfiles.length})`}>
          <OpProfileTable
            opProfiles={opProfiles}
            actionHint={topBottleneck ? `Next best action: inspect ${topBottleneck.op_name} first; it is currently the top bottleneck candidate.` : null}
          />
        </CollapsibleSection>
      )}

      {/* Bottleneck Highlights */}
      {profilingMetrics?.bottleneck_ops?.length > 0 && (
        <CollapsibleSection title="Bottlenecks">
          <div>
            {profilingMetrics.bottleneck_ops.slice(0, 3).map((op, i) => (
              <span key={i} className="bottleneck-badge">{op}</span>
            ))}
          </div>
        </CollapsibleSection>
      )}

      {/* Compression & Efficiency */}
      {compressionMetrics && (
        <CollapsibleSection title="Compression & Efficiency">
          {/* Efficiency Score Badge */}
          <div className="efficiency-score-badge">
            <div className="eff-score-value">{Math.round((compressionMetrics.efficiency_score || 0) * 100)}</div>
            <div className="eff-score-label">Efficiency</div>
          </div>

          {/* Score Breakdown */}
          <div className="compression-breakdown">
            {[
              { label: 'Prune Tol.', val: compressionMetrics.pruning_tolerance, color: '#24d1a0' },
              { label: 'Compression', val: Math.min((compressionMetrics.compression_ratio || 1) / 4, 1), color: '#17a3ff' },
              { label: 'Sparse Ops', val: compressionMetrics.sparse_op_coverage, color: '#a060ff' },
              { label: 'Mem Eff.', val: compressionMetrics.memory_efficiency_score, color: '#f0a020' },
            ].map(({ label, val, color }) => (
              <div className="fp-row" key={label}>
                <span className="fp-label">{label}</span>
                <ProgressBar value={val} color={color} />
                <span className="fp-val">{val != null ? (val * 100).toFixed(0) + '%' : '-'}</span>
              </div>
            ))}
          </div>

          {/* Pruning Tolerance Curve */}
          {compressionMetrics.pruning_curve?.length > 0 && (
            <div className="chart-container" style={{ marginTop: 8 }}>
              <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 4 }}>Pruning Curve</div>
            <ResponsiveContainer width="100%" height={120}>
                <LineChart data={compressionMetrics.pruning_curve} margin={{ left: 10, right: 10, top: 4, bottom: 4 }}>
                  <XAxis dataKey="sparsity" tick={{ fill: '#8fa8c2', fontSize: 10 }} tickFormatter={v => (v * 100) + '%'} />
                  <YAxis tick={{ fill: '#8fa8c2', fontSize: 10 }} domain={[0, 'auto']} tickFormatter={v => v.toFixed(1) + 'x'} />
                  <Tooltip
                    contentStyle={{ background: '#101b2b', border: '1px solid #1f3147', borderRadius: 6, fontSize: 12 }}
                    formatter={(v) => v.toFixed(3) + 'x'}
                    labelFormatter={(v) => 'Sparsity: ' + (v * 100) + '%'}
                  />
                  <Line type="monotone" dataKey="loss_ratio" stroke="#a060ff" strokeWidth={2} dot={{ r: 3, fill: '#a060ff' }} />
                </LineChart>
            </ResponsiveContainer>
            <ChartActionRail
              insight={compressionMetrics.compression_ratio != null ? `Compression ratio: ${Number(compressionMetrics.compression_ratio).toFixed(2)}x` : null}
              recommendation="Next best action: compare loss-vs-sparsity tradeoffs, then keep only sparse ops that preserve quality before another deep run."
            />
          </div>
          )}

          {/* Compression Metrics Grid */}
          <div className="metrics-grid" style={{ marginTop: 8 }}>
            <div className="stat">
              <div className="stat-val">{formatNum(compressionMetrics.compression_ratio?.toFixed(2))}x</div>
              <div className="stat-label">Compression</div>
            </div>
            <div className="stat">
              <div className="stat-val">{compressionMetrics.sparse_ops || 0}</div>
              <div className="stat-label">Sparse Ops</div>
            </div>
            <div className="stat">
              <div className="stat-val">{compressionMetrics.theoretical_size_int8_mb?.toFixed(1)}MB</div>
              <div className="stat-label">INT8 Size</div>
            </div>
            <div className="stat">
              <div className="stat-val">{compressionMetrics.theoretical_size_int4_mb?.toFixed(1)}MB</div>
              <div className="stat-label">INT4 Size</div>
            </div>
          </div>

          {/* Sparse Op Badges */}
          {compressionMetrics.sparse_op_names?.length > 0 && (
            <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {compressionMetrics.sparse_op_names.map(name => (
                <span key={name} className="sparse-op-badge">{name}</span>
              ))}
            </div>
          )}
        </CollapsibleSection>
      )}

      {/* Fingerprint Summary */}
      {fingerprintMetrics && !fingerprintMetrics.skipped && (
        <CollapsibleSection title="Fingerprint">
          <div className="fingerprint-grid">
            {[
              { label: 'Transformer', val: fingerprintMetrics.cka_vs_transformer, color: '#17a3ff' },
              { label: 'SSM', val: fingerprintMetrics.cka_vs_ssm, color: '#a060ff' },
              { label: 'Conv', val: fingerprintMetrics.cka_vs_conv, color: '#ff6090' },
              { label: 'Locality', val: fingerprintMetrics.locality, color: '#24d1a0' },
              { label: 'Sparsity', val: fingerprintMetrics.sparsity, color: '#f0a020' },
              { label: 'Isotropy', val: fingerprintMetrics.isotropy, color: '#20c0f0' },
            ].map(({ label, val, color }) => (
              <div className="fp-row" key={label}>
                <span className="fp-label">{label}</span>
                <ProgressBar value={val} color={color} />
                <span className="fp-val">{val != null ? val.toFixed(2) : '-'}</span>
              </div>
            ))}
          </div>
        </CollapsibleSection>
      )}

      {/* Novelty Scores */}
      {noveltyMetrics && !noveltyMetrics.skipped && (
        <CollapsibleSection title="Novelty">
          <div className="novelty-bars">
            {[
              { label: 'Structural', val: noveltyMetrics.structural_novelty },
              { label: 'Behavioral', val: noveltyMetrics.behavioral_novelty },
              { label: 'Overall', val: noveltyMetrics.overall_novelty },
            ].map(({ label, val }) => (
              <div className="novelty-row" key={label}>
                <span className="nov-label">{label}</span>
                <ProgressBar value={val} color="var(--accent)" />
                <span className="fp-val">{val != null ? val.toFixed(2) : '-'}</span>
              </div>
            ))}
            {noveltyMetrics.most_similar_to && (
              <div style={{ fontSize: 11, color: 'var(--muted)', paddingLeft: 88 }}>
                Most similar to: <strong style={{ color: 'var(--text)' }}>{noveltyMetrics.most_similar_to}</strong>
              </div>
            )}
          </div>
        </CollapsibleSection>
      )}

      {benchmarkMetrics && (
        <CollapsibleSection title="Benchmark Targets" defaultOpen>
          <BenchmarkTargets
            benchmarkMetrics={benchmarkMetrics}
            benchmarkObserved={benchmarkObserved}
            onBenchmarkObservedChange={onBenchmarkObservedChange}
          />

          {notMeasuredTargets.length > 0 && (
            <div style={{ marginTop: 8, fontSize: 11, color: 'var(--muted)', lineHeight: 1.55 }}>
              {notMeasuredTargets.slice(0, 4).map((t) => (
                <div key={t.id}>
                  <strong style={{ color: 'var(--text)' }}>{t.label}:</strong> {t.measurement || 'External benchmark input required.'}
                </div>
              ))}
            </div>
          )}

          {benchmarkMetrics.scaling_projection && (
            <div style={{ marginTop: 8, fontSize: 11, color: 'var(--muted)', lineHeight: 1.5 }}>
              <div>
                Projected Mamba-scale avg accuracy at current params:
                <strong style={{ color: 'var(--text)' }}> {
                  benchmarkMetrics.scaling_projection.projected_mamba_avg_accuracy == null
                    ? '-'
                    : benchmarkMetrics.scaling_projection.projected_mamba_avg_accuracy.toFixed(2)
                }</strong>
              </div>
              <div>
                Delta vs Mamba-2.8B reference:
                <strong style={{ color: 'var(--text)' }}> {
                  benchmarkMetrics.scaling_projection.delta_vs_mamba_2p8b_avg == null
                    ? '-'
                    : benchmarkMetrics.scaling_projection.delta_vs_mamba_2p8b_avg.toFixed(2)
                }</strong>
              </div>
            </div>
          )}
          <ChartActionRail
            insight={benchmarkInputCount > 0 ? `${benchmarkInputCount} external benchmark values loaded` : null}
            recommendation={notMeasuredTargets.length > 0 ? 'Next best action: fill the remaining benchmark inputs, then rerun Deep Run to convert “not measured” rows into actionable evidence.' : 'Next best action: use these targets to decide whether another deep run or a benchmark-focused comparison is warranted.'}
          />
        </CollapsibleSection>
      )}
    </div>
  )
}

export default memo(RunResultsPanel)
