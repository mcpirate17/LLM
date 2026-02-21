import { useState, useMemo } from 'react'
import { CheckCircle2, Loader, Circle, XCircle, ChevronRight } from 'lucide-react'
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

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

function formatNum(n) {
  if (n == null) return '-'
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B'
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K'
  return typeof n === 'number' ? n.toFixed(n % 1 ? 2 : 0) : String(n)
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
      <button className="section-toggle" onClick={() => setOpen(!open)}>
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

export default function RunResultsPanel({ evalState }) {
  const { stages = [], status, totalTimeMs, error } = evalState || {}
  const [sortCol, setSortCol] = useState('flops')
  const [sortAsc, setSortAsc] = useState(false)

  const stageMap = useMemo(() => {
    const m = {}
    for (const s of stages) m[s.stage] = s
    return m
  }, [stages])

  const sandboxMetrics = stageMap.sandbox?.metrics
  const profilingMetrics = stageMap.profiling?.metrics
  const compressionMetrics = stageMap.compression?.metrics
  const fingerprintMetrics = stageMap.fingerprint?.metrics
  const noveltyMetrics = stageMap.novelty?.metrics

  // FLOPs by category chart data
  const chartData = useMemo(() => {
    const raw = profilingMetrics?.flops_by_category || {}
    return Object.entries(raw)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
  }, [profilingMetrics])

  // Sorted op profiles
  const opProfiles = useMemo(() => {
    const ops = profilingMetrics?.op_profiles || []
    const sorted = [...ops].sort((a, b) => {
      const av = a[sortCol] ?? 0
      const bv = b[sortCol] ?? 0
      return sortAsc ? av - bv : bv - av
    })
    return sorted
  }, [profilingMetrics, sortCol, sortAsc])

  const handleSort = (col) => {
    if (sortCol === col) setSortAsc(!sortAsc)
    else { setSortCol(col); setSortAsc(false) }
  }

  if (!evalState || stages.length === 0) {
    return (
      <div className="eval-results" style={{ color: 'var(--muted)', padding: 16, textAlign: 'center' }}>
        Click <strong>Deep Run</strong> to stream evaluation results.
      </div>
    )
  }

  return (
    <div className="eval-results">
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
          </div>
        </CollapsibleSection>
      )}

      {/* Per-Op Profile Table */}
      {opProfiles.length > 0 && (
        <CollapsibleSection title={`Per-Op Profile (${opProfiles.length})`}>
          <div style={{ maxHeight: 200, overflowY: 'auto' }}>
            <table className="op-profile-table">
              <thead>
                <tr>
                  <th onClick={() => handleSort('op_name')}>Op</th>
                  <th onClick={() => handleSort('flops')}>FLOPs</th>
                  <th onClick={() => handleSort('params')}>Params</th>
                  <th onClick={() => handleSort('memory_bytes')}>Mem</th>
                  <th>K</th>
                </tr>
              </thead>
              <tbody>
                {opProfiles.map((op, i) => (
                  <tr key={i}>
                    <td>{op.op_name}</td>
                    <td>{formatNum(op.flops)}</td>
                    <td>{formatNum(op.params)}</td>
                    <td>{formatNum(op.memory_bytes)}</td>
                    <td>{op.has_native_kernel ? <span className="native-badge">C</span> : null}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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
    </div>
  )
}
