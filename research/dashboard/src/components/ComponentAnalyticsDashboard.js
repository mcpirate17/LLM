import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { apiCall } from '../services/apiService';
import MiniChart from './charts/MiniChart';

const STATUS_COLORS = {
  healthy: '#22c55e',
  structural: '#6b7280',
  degraded: '#eab308',
  broken: '#ef4444',
};

const SOURCE_COLORS = {
  search: '#3b82f6',
  'search+profiling': '#8b5cf6',
  profiling_only: '#6b7280',
};

const TIME_WINDOWS = [
  { value: '1h', label: '1h' },
  { value: '6h', label: '6h' },
  { value: '24h', label: '24h' },
  { value: '7d', label: '7d' },
  { value: 'all', label: 'All' },
];

function fmtPct(value) {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function fmtLoss(value) {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  return Number(value).toFixed(3);
}

// ─── Health Summary ───
function HealthSummary({ health }) {
  if (!health) return null;
  const items = [
    { label: 'Total Ops', value: health.total, color: 'var(--text-primary)' },
    { label: 'Healthy', value: health.healthy, color: STATUS_COLORS.healthy },
    { label: 'Degraded', value: health.degraded, color: STATUS_COLORS.degraded },
    { label: 'Broken', value: health.broken, color: STATUS_COLORS.broken },
  ];
  return (
    <div style={{ display: 'flex', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
      {items.map(item => (
        <div key={item.label} style={{
          flex: '1 1 100px', padding: '10px 14px', borderRadius: 8,
          background: 'var(--bg-secondary)',
          border: `1px solid ${item.value > 0 && item.label !== 'Total Ops' && item.label !== 'Healthy' ? item.color + '44' : 'var(--border-color)'}`,
        }}>
          <div style={{ fontSize: 22, fontWeight: 700, color: item.color }}>{item.value}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{item.label}</div>
        </div>
      ))}
    </div>
  );
}

// ─── Sortable Column Header ───
const SORT_COLUMNS = [
  { key: 'status', label: 'Status' },
  { key: 'op', label: 'Op Name' },
  { key: 'n_used', label: 'Used' },
  { key: 's0_rate', label: 'S0 Rate' },
  { key: 's1_rate', label: 'S1 Rate' },
  { key: 'grad_norm', label: 'Grad Norm' },
  { key: 'fwd_us', label: 'Fwd (us)' },
  { key: 'reasons', label: 'Issues' },
];

const STATUS_ORDER = { broken: 0, degraded: 1, structural: 2, healthy: 3 };

function sortComponents(list, sortKey, sortDir) {
  if (!sortKey) return list;
  return [...list].sort((a, b) => {
    let va = a[sortKey];
    let vb = b[sortKey];
    // Status sorts by severity
    if (sortKey === 'status') {
      va = STATUS_ORDER[va] ?? 3;
      vb = STATUS_ORDER[vb] ?? 3;
    }
    // Issues sorts by count
    if (sortKey === 'reasons') {
      va = Array.isArray(va) ? va.length : 0;
      vb = Array.isArray(vb) ? vb.length : 0;
    }
    // Nulls always sort last
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === 'string') {
      const cmp = va.localeCompare(vb);
      return sortDir === 'asc' ? cmp : -cmp;
    }
    return sortDir === 'asc' ? va - vb : vb - va;
  });
}

// ─── Component Health Grid ───
function ComponentGrid({ components, filter, searchTerm, sourceFilter }) {
  const [sortKey, setSortKey] = useState('n_used');
  const [sortDir, setSortDir] = useState('desc');

  const handleSort = useCallback((key) => {
    setSortKey(prev => {
      if (prev === key) {
        setSortDir(d => d === 'asc' ? 'desc' : 'asc');
        return key;
      }
      setSortDir(key === 'op' ? 'asc' : 'desc');
      return key;
    });
  }, []);

  const filtered = useMemo(() => {
    let list = components || [];
    // Deduplicate by op name (keep first occurrence — highest priority from backend sort)
    const seen = new Set();
    list = list.filter(c => {
      if (seen.has(c.op)) return false;
      seen.add(c.op);
      return true;
    });
    if (filter !== 'all') list = list.filter(c => c.status === filter);
    if (sourceFilter !== 'all') list = list.filter(c => c.data_source === sourceFilter);
    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      list = list.filter(c => c.op.toLowerCase().includes(q));
    }
    return sortComponents(list, sortKey, sortDir);
  }, [components, filter, searchTerm, sourceFilter, sortKey, sortDir]);

  if (filtered.length === 0) {
    return <p className="ux-state ux-state-empty">No components match the current filter.</p>;
  }

  const sortArrow = (key) => {
    if (sortKey !== key) return <span style={{ opacity: 0.25, marginLeft: 2 }}>↕</span>;
    return <span style={{ marginLeft: 2 }}>{sortDir === 'asc' ? '↑' : '↓'}</span>;
  };

  return (
    <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
      <table className="data-table" style={{ fontSize: 12 }}>
        <thead style={{ position: 'sticky', top: 0, zIndex: 1, background: 'var(--bg-primary)' }}>
          <tr>
            {SORT_COLUMNS.map(col => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap', background: 'var(--bg-primary)' }}
              >
                {col.label}{sortArrow(col.key)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {filtered.map(c => (
            <tr key={c.op} style={{
              background: c.status === 'broken' ? '#ef444408' : c.status === 'degraded' ? '#eab30808' : undefined,
            }}>
              <td>
                <span style={{
                  display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                  background: STATUS_COLORS[c.status],
                  boxShadow: c.status !== 'healthy' ? `0 0 4px ${STATUS_COLORS[c.status]}66` : 'none',
                }} />
              </td>
              <td style={{ fontFamily: 'monospace', fontWeight: c.status !== 'healthy' ? 600 : 400 }}>
                {c.op}
                {c.data_source && (
                  <span style={{
                    marginLeft: 6, padding: '1px 5px', borderRadius: 4, fontSize: 9,
                    background: (SOURCE_COLORS[c.data_source] || '#666') + '22',
                    color: SOURCE_COLORS[c.data_source] || '#666',
                  }}>
                    {c.data_source === 'profiling_only' ? 'prof' : c.data_source === 'search+profiling' ? 's+p' : 'src'}
                  </span>
                )}
              </td>
              <td style={{ textAlign: 'right' }}>{c.n_used || 0}</td>
              <td style={{
                textAlign: 'right',
                color: c.s0_rate !== null ? (c.s0_rate < 0.3 ? STATUS_COLORS.broken : c.s0_rate < 0.6 ? STATUS_COLORS.degraded : 'var(--text-primary)') : 'var(--text-muted)',
              }}>
                {c.s0_rate !== null ? `${(c.s0_rate * 100).toFixed(0)}%` : '-'}
              </td>
              <td style={{
                textAlign: 'right',
                color: c.s1_rate !== null ? (c.s1_rate < 0.05 ? STATUS_COLORS.broken : c.s1_rate < 0.15 ? STATUS_COLORS.degraded : 'var(--text-primary)') : 'var(--text-muted)',
              }}>
                {c.s1_rate !== null ? `${(c.s1_rate * 100).toFixed(1)}%` : '-'}
              </td>
              <td style={{
                textAlign: 'right', fontFamily: 'monospace', fontSize: 11,
                color: c.grad_norm !== null ? (c.grad_norm > 50000 ? STATUS_COLORS.broken : c.grad_norm > 3000 ? STATUS_COLORS.degraded : 'var(--text-muted)') : 'var(--text-muted)',
              }}>
                {c.grad_norm !== null ? (c.grad_norm > 1e6 ? `${(c.grad_norm / 1e6).toFixed(1)}M` : c.grad_norm > 1e3 ? `${(c.grad_norm / 1e3).toFixed(1)}K` : c.grad_norm.toFixed(0)) : '-'}
              </td>
              <td style={{ textAlign: 'right', fontFamily: 'monospace', fontSize: 11, color: 'var(--text-muted)' }}>
                {c.fwd_us != null ? c.fwd_us.toFixed(1) : '-'}
              </td>
              <td style={{ fontSize: 11, color: 'var(--text-muted)', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {c.reasons && c.reasons.length > 0 ? c.reasons.join('; ') : ''}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Failure Blocklist ───
function FailureBlocklist({ blocklist }) {
  if (!blocklist || Object.keys(blocklist).length === 0) return null;
  const entries = Object.entries(blocklist).sort((a, b) => a[1] - b[1]).slice(0, 20);
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Failure Penalties (auto-deweighted op pairs)</div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table" style={{ fontSize: 12 }}>
          <thead><tr><th>Op Pair Signature</th><th>Penalty</th></tr></thead>
          <tbody>
            {entries.map(([sig, penalty]) => (
              <tr key={sig}>
                <td style={{ fontFamily: 'monospace' }}>{sig}</td>
                <td style={{
                  textAlign: 'right', fontWeight: 600,
                  color: penalty <= 0.05 ? STATUS_COLORS.broken : STATUS_COLORS.degraded,
                }}>
                  {`${(penalty * 100).toFixed(0)}%`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Op Pair Heatmap ───
const PAIR_COLUMNS = [
  { key: 'op_a', label: 'Op A' },
  { key: 'op_b', label: 'Op B' },
  { key: 'n', label: 'Count' },
  { key: 's0_rate', label: 'S0 Rate' },
  { key: 's1_rate', label: 'S1 Rate' },
];

function OpPairHeatmap({ pairs }) {
  const [sortKey, setSortKey] = useState('n');
  const [sortDir, setSortDir] = useState('desc');

  const handleSort = useCallback((key) => {
    setSortKey(prev => {
      if (prev === key) {
        setSortDir(d => d === 'asc' ? 'desc' : 'asc');
        return key;
      }
      setSortDir(key === 'op_a' || key === 'op_b' ? 'asc' : 'desc');
      return key;
    });
  }, []);

  const sorted = useMemo(() => {
    return sortComponents(pairs || [], sortKey, sortDir);
  }, [pairs, sortKey, sortDir]);

  if (!sorted || sorted.length === 0) return null;

  const sortArrow = (key) => {
    if (sortKey !== key) return <span style={{ opacity: 0.25, marginLeft: 2 }}>↕</span>;
    return <span style={{ marginLeft: 2 }}>{sortDir === 'asc' ? '↑' : '↓'}</span>;
  };

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Top Op Pairs (co-occurrence)</div>
      <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
        <table className="data-table" style={{ fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0, zIndex: 1, background: 'var(--bg-primary)' }}>
            <tr>
              {PAIR_COLUMNS.map(col => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap', background: 'var(--bg-primary)' }}
                >
                  {col.label}{sortArrow(col.key)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map(p => {
              const s0Color = p.s0_rate < 0.3 ? STATUS_COLORS.broken : p.s0_rate < 0.6 ? STATUS_COLORS.degraded : '#22c55e';
              return (
                <tr key={`${p.op_a}-${p.op_b}`}>
                  <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{p.op_a}</td>
                  <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{p.op_b}</td>
                  <td style={{ textAlign: 'right' }}>{p.n}</td>
                  <td style={{ textAlign: 'right', color: s0Color }}>{(p.s0_rate * 100).toFixed(0)}%</td>
                  <td style={{ textAlign: 'right', color: p.s1_rate < 0.05 ? STATUS_COLORS.broken : 'var(--text-primary)' }}>
                    {(p.s1_rate * 100).toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Loss Distribution Panel (CSS box plots) ───
function LossDistributionPanel({ distributions }) {
  if (!distributions || distributions.length === 0) return null;
  const globalMax = Math.max(...distributions.map(d => d.max), 1.5);

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Loss Distribution by Op</div>
      <div style={{ maxHeight: 400, overflowY: 'auto' }}>
        {distributions.slice(0, 30).map(d => {
          const scale = (v) => `${Math.min((v / globalMax) * 100, 100)}%`;
          return (
            <div key={d.op} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ width: 120, fontSize: 11, fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{d.op}</span>
              <div style={{ flex: 1, position: 'relative', height: 16, background: 'var(--bg-tertiary)', borderRadius: 4 }}>
                {/* Whiskers min-max */}
                <div style={{
                  position: 'absolute', top: 7, height: 2, background: 'var(--text-muted)',
                  left: scale(d.min), width: `calc(${scale(d.max)} - ${scale(d.min)})`,
                  opacity: 0.4,
                }} />
                {/* Box q1-q3 */}
                <div style={{
                  position: 'absolute', top: 2, height: 12, borderRadius: 2,
                  background: 'var(--accent-blue)', opacity: 0.5,
                  left: scale(d.q1), width: `calc(${scale(d.q3)} - ${scale(d.q1)})`,
                }} />
                {/* Median line */}
                <div style={{
                  position: 'absolute', top: 1, height: 14, width: 2,
                  background: '#fff', borderRadius: 1, left: scale(d.median),
                }} />
              </div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)', minWidth: 50, textAlign: 'right' }}>
                n={d.n}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Grammar Evolution Panel ───
function GrammarEvolutionPanel({ events }) {
  if (!events || events.length === 0) return null;
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Grammar Evolution</div>
      <div style={{ maxHeight: 300, overflowY: 'auto' }}>
        {events.map(e => (
          <div key={e.id} style={{ padding: '6px 0', borderBottom: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              {new Date(e.timestamp * 1000).toLocaleString()}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-primary)', marginTop: 2 }}>
              {e.description?.slice(0, 100)}
            </div>
            {e.changes && Object.keys(e.changes).length > 0 && (
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 4 }}>
                {Object.entries(e.changes).slice(0, 8).map(([op, ch]) => (
                  <span key={op} style={{
                    padding: '1px 6px', borderRadius: 4, fontSize: 10,
                    background: ch.new > ch.old ? '#22c55e22' : '#ef444422',
                    color: ch.new > ch.old ? '#22c55e' : '#ef4444',
                  }}>
                    {op}: {Number(ch.old).toFixed(2)} &rarr; {Number(ch.new).toFixed(2)}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Failure Pattern Panel ───
function FailurePatternPanel({ patterns }) {
  if (!patterns || patterns.length === 0) return null;
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Failure Patterns</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {patterns.slice(0, 10).map(p => (
          <div key={p.error_type} style={{
            padding: '8px 12px', borderRadius: 6, background: '#ef444408',
            borderLeft: '3px solid #ef4444',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>{p.error_type}</span>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{p.count} occurrences</span>
            </div>
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 4 }}>
              {p.top_ops.map(op => (
                <span key={op.op} style={{
                  padding: '1px 6px', borderRadius: 4, fontSize: 10,
                  background: 'var(--bg-tertiary)', color: 'var(--text-muted)',
                }}>
                  {op.op} ({op.occurrences})
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Leaderboard Dynamics Panel ───
function LeaderboardDynamicsPanel({ daily, recentPromotions }) {
  if ((!daily || Object.keys(daily).length === 0) && (!recentPromotions || recentPromotions.length === 0)) return null;

  const tierColors = { screening: '#3b82f6', investigation: '#eab308', validation: '#22c55e', breakthrough: '#a855f7' };

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Leaderboard Dynamics</div>
      {daily && Object.keys(daily).length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Daily tier counts</div>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table" style={{ fontSize: 11 }}>
              <thead>
                <tr><th>Date</th><th>Screening</th><th>Investigation</th><th>Validation</th><th>Breakthrough</th></tr>
              </thead>
              <tbody>
                {Object.entries(daily).slice(-10).map(([day, tiers]) => (
                  <tr key={day}>
                    <td>{day}</td>
                    {['screening', 'investigation', 'validation', 'breakthrough'].map(t => (
                      <td key={t} style={{ textAlign: 'right', color: tierColors[t] || 'var(--text-primary)' }}>{tiers[t] || 0}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {recentPromotions && recentPromotions.length > 0 && (
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Recent entries</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {recentPromotions.slice(0, 8).map(p => (
              <div key={p.entry_id} style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 12 }}>
                <span style={{
                  padding: '1px 6px', borderRadius: 4, fontSize: 10, fontWeight: 600,
                  background: (tierColors[p.tier] || '#888') + '22',
                  color: tierColors[p.tier] || '#888',
                }}>{p.tier}</span>
                <span style={{ fontFamily: 'monospace', fontSize: 11, color: 'var(--text-muted)' }}>{p.result_id?.slice(0, 12)}</span>
                {p.composite_score != null && (
                  <span style={{ fontSize: 11 }}>score: {p.composite_score.toFixed(3)}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Insight Effectiveness Panel ───
function InsightEffectivenessPanel({ insights }) {
  if (!insights || insights.length === 0) return null;
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Insight Effectiveness</div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table" style={{ fontSize: 12 }}>
          <thead>
            <tr><th>Type</th><th>Subject</th><th>Predictions</th><th>Accuracy</th><th>Bayesian Mean</th><th>Status</th></tr>
          </thead>
          <tbody>
            {insights.slice(0, 20).map(ins => (
              <tr key={ins.insight_id}>
                <td style={{ fontSize: 11 }}>{ins.insight_type || ins.category}</td>
                <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{ins.subject_key?.slice(0, 20) || '-'}</td>
                <td style={{ textAlign: 'right' }}>{ins.n_predictions}</td>
                <td style={{
                  textAlign: 'right', fontWeight: 600,
                  color: ins.accuracy > 0.6 ? '#22c55e' : ins.accuracy > 0.3 ? '#eab308' : '#ef4444',
                }}>
                  {(ins.accuracy * 100).toFixed(0)}%
                </td>
                <td style={{ textAlign: 'right' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4, justifyContent: 'flex-end' }}>
                    <div style={{ width: 40, height: 6, borderRadius: 3, background: 'var(--bg-tertiary)', overflow: 'hidden' }}>
                      <div style={{
                        width: `${(ins.bayesian_mean * 100).toFixed(0)}%`, height: '100%',
                        background: ins.bayesian_mean > 0.5 ? '#22c55e' : '#eab308', borderRadius: 3,
                      }} />
                    </div>
                    <span style={{ fontSize: 11 }}>{ins.bayesian_mean.toFixed(2)}</span>
                  </div>
                </td>
                <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{ins.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StructuralDiagnosticsPanel({ data }) {
  if (!data) return null;
  const topTemplates = Array.isArray(data.top_templates) ? data.top_templates.slice(0, 5) : [];
  const strugglingTemplates = Array.isArray(data.struggling_templates) ? data.struggling_templates.slice(0, 5) : [];
  const weakSlots = Array.isArray(data.slot_observability) ? data.slot_observability.slice(0, 6) : [];
  const templateTrends = Array.isArray(data.template_trends) ? data.template_trends.slice(0, 3) : [];
  const slotTrends = Array.isArray(data.slot_trends) ? data.slot_trends.slice(0, 3) : [];
  const lossTrends = Array.isArray(data.loss_trends) ? data.loss_trends : [];
  const recommendations = Array.isArray(data.recommendations) ? data.recommendations : [];
  const loss = data.loss_distribution || {};
  const summary = data.summary || {};

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Structural Diagnostics</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Template and slot-level observability for the search grammar. This highlights which structural recipes survive, which slots are collapsing quality, and whether validation loss is drifting above training loss.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 10, marginBottom: 16 }}>
        <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }}>{Number(summary.templates_tracked || 0)}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Templates tracked</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>{Number(summary.avg_templates_per_graph || 0).toFixed(2)} / graph</div>
        </div>
        <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }}>{Number(summary.avg_motifs_per_graph || 0).toFixed(2)}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Avg motifs / graph</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>{Number(summary.motifs_tracked || 0)} motifs tracked</div>
        </div>
        <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent-blue)' }}>{fmtLoss(loss.training?.median)}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Median train LR</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>P75 {fmtLoss(loss.training?.p75)}</div>
        </div>
        <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent-yellow)' }}>{fmtLoss(loss.validation?.median)}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Median val LR</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>P75 {fmtLoss(loss.validation?.p75)}</div>
        </div>
        <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent-blue)' }}>{Number(summary.routing_fast_lane_templates || 0)}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Routing fast-lane templates</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>
            {Number(summary.routing_fast_lane_positive_templates || 0)} positive slow starters
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>Highest Success Templates</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {topTemplates.length > 0 ? topTemplates.map((row) => (
              <div key={row.name} style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>{row.name}</span>
                  <span style={{ fontSize: 12, color: '#22c55e', fontWeight: 700 }}>{fmtPct(row.s1_rate)}</span>
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>
                  {row.n_used} runs · val LR {fmtLoss(row.avg_validation_loss_ratio ?? row.avg_loss_ratio)} · best {fmtLoss(row.best_loss_ratio)}
                  {row.routing_fast_lane_runs ? ` · fast lane ${fmtPct(row.routing_fast_lane_positive_rate)}` : ''}
                </div>
              </div>
            )) : <div className="ux-state ux-state-empty">No template diagnostics yet.</div>}
          </div>
        </div>

        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>Templates To Fix</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {strugglingTemplates.length > 0 ? strugglingTemplates.map((row) => (
              <div key={row.name} style={{ padding: '8px 10px', borderRadius: 6, background: '#ef444408', border: '1px solid #ef444422' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>{row.name}</span>
                  <span style={{ fontSize: 12, color: '#ef4444', fontWeight: 700 }}>{fmtPct(row.s1_rate)}</span>
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>
                  {row.n_used} runs · slots {row.slot_count || 0} · fail {row.top_failure_reason || 'unknown'} · avg LR {fmtLoss(row.avg_validation_loss_ratio ?? row.avg_loss_ratio)}
                  {row.routing_fast_lane_runs ? ` · fast lane ${fmtPct(row.routing_fast_lane_positive_rate)}` : ''}
                </div>
              </div>
            )) : <div className="ux-state ux-state-empty">No struggling templates identified.</div>}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.15fr 0.85fr', gap: 16 }}>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>Weakest Slots</div>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table" style={{ fontSize: 11 }}>
              <thead>
                <tr><th>Slot</th><th>Template</th><th>S1</th><th>Avg LR</th><th>Motif</th><th>Failure</th></tr>
              </thead>
              <tbody>
                {weakSlots.map((row) => (
                  <tr key={row.slot_key}>
                    <td style={{ fontFamily: 'monospace' }}>{row.slot_key}</td>
                    <td>{row.template_name}</td>
                    <td style={{ textAlign: 'right', color: (row.s1_rate || 0) < 0.15 ? '#ef4444' : '#eab308' }}>{fmtPct(row.s1_rate)}</td>
                    <td style={{ textAlign: 'right' }}>{fmtLoss(row.avg_loss_ratio)}</td>
                    <td style={{ fontFamily: 'monospace' }}>{row.top_selected_motif || '-'}</td>
                    <td>{row.top_failure_reason || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {Array.isArray(summary.zero_slot_templates) && summary.zero_slot_templates.length > 0 && (
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 8 }}>
              Zero-slot templates: {summary.zero_slot_templates.slice(0, 6).join(', ')}
              {summary.zero_slot_templates.length > 6 ? ` +${summary.zero_slot_templates.length - 6}` : ''}
            </div>
          )}
        </div>

        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>Recommended Fixes</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {recommendations.length > 0 ? recommendations.map((item, idx) => (
              <div key={idx} style={{ padding: '9px 10px', borderRadius: 6, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', fontSize: 12, color: 'var(--text-primary)', lineHeight: 1.5 }}>
                {item}
              </div>
            )) : <div className="ux-state ux-state-empty">No structural recommendations yet.</div>}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 18 }}>
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>Template Success Trends</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {templateTrends.length > 0 ? templateTrends.map((series) => (
              <div key={series.name} style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
                <MiniChart
                  data={series.points}
                  valueKey="s1_rate"
                  label={`${series.name} S1`}
                  color="#22c55e"
                  formatValue={(v) => `${(Number(v) * 100).toFixed(1)}%`}
                  scaleKey="pass_rate"
                  windowSize={12}
                />
              </div>
            )) : <div className="ux-state ux-state-empty">Need more experiments for template trends.</div>}
          </div>
        </div>

        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>Weak Slot Trends</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {slotTrends.length > 0 ? slotTrends.map((series) => (
              <div key={series.slot_key} style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
                <MiniChart
                  data={series.points}
                  valueKey="s1_rate"
                  label={`${series.slot_key} S1`}
                  color="#ef4444"
                  formatValue={(v) => `${(Number(v) * 100).toFixed(1)}%`}
                  scaleKey="pass_rate"
                  windowSize={12}
                />
              </div>
            )) : <div className="ux-state ux-state-empty">Need more experiments for slot trends.</div>}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 18 }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>Loss Drift Trends</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
          <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
            <MiniChart
              data={lossTrends}
              valueKey="training_median"
              label="Median Train LR"
              color="#58a6ff"
              formatValue={(v) => Number(v).toFixed(3)}
              scaleKey="loss_ratio"
              windowSize={12}
            />
          </div>
          <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
            <MiniChart
              data={lossTrends}
              valueKey="validation_median"
              label="Median Val LR"
              color="#d29922"
              formatValue={(v) => Number(v).toFixed(3)}
              scaleKey="loss_ratio"
              windowSize={12}
            />
          </div>
          <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
            <MiniChart
              data={lossTrends}
              valueKey="discovery_median"
              label="Median Discovery LR"
              color="#a855f7"
              formatValue={(v) => Number(v).toFixed(3)}
              scaleKey="loss_ratio"
              windowSize={12}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Main Component Analytics Dashboard ───
export default function ComponentAnalyticsDashboard() {
  const [health, setHealth] = useState(null);
  const [blocklist, setBlocklist] = useState({});
  const [opPairs, setOpPairs] = useState([]);
  const [lossDist, setLossDist] = useState([]);
  const [grammarEvents, setGrammarEvents] = useState([]);
  const [failurePatterns, setFailurePatterns] = useState([]);
  const [leaderboardData, setLeaderboardData] = useState({ daily: {}, recent_promotions: [] });
  const [insightData, setInsightData] = useState([]);
  const [filter, setFilter] = useState('all');
  const [sourceFilter, setSourceFilter] = useState('all');
  const [timeWindow, setTimeWindow] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const windowParam = timeWindow !== 'all' ? `?window=${timeWindow}` : '';
      const [healthRes, blockRes, pairRes, lossRes, gramRes, failRes, lbRes, insRes] = await Promise.all([
        apiCall(`/api/observability/health${windowParam}`),
        apiCall('/api/observability/failure-blocklist'),
        apiCall('/api/observability/op-pairs'),
        apiCall('/api/observability/loss-distribution'),
        apiCall('/api/observability/grammar-evolution'),
        apiCall('/api/observability/failure-patterns'),
        apiCall('/api/observability/leaderboard-dynamics'),
        apiCall('/api/observability/insight-effectiveness'),
      ]);
      if (healthRes.ok) setHealth(await healthRes.json());
      if (blockRes.ok) { const d = await blockRes.json(); setBlocklist(d.blocklist || {}); }
      if (pairRes.ok) { const d = await pairRes.json(); setOpPairs(d.pairs || []); }
      if (lossRes.ok) { const d = await lossRes.json(); setLossDist(d.distributions || []); }
      if (gramRes.ok) { const d = await gramRes.json(); setGrammarEvents(d.events || []); }
      if (failRes.ok) { const d = await failRes.json(); setFailurePatterns(d.patterns || []); }
      if (lbRes.ok) setLeaderboardData(await lbRes.json());
      if (insRes.ok) { const d = await insRes.json(); setInsightData(d.insights || []); }
    } catch (err) {
      console.error('ComponentAnalytics fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [timeWindow]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await apiCall('/api/observability/health/refresh', { method: 'POST' });
      await fetchData();
    } finally {
      setRefreshing(false);
    }
  }, [fetchData]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (loading) {
    return <div className="card"><p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Loading component analytics...</p></div>;
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>Component Analytics</h2>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <select value={timeWindow} onChange={e => setTimeWindow(e.target.value)} style={{
            padding: '4px 8px', fontSize: 11, borderRadius: 6,
            background: 'var(--bg-tertiary)', color: 'var(--text-primary)',
            border: '1px solid var(--border-color)', cursor: 'pointer',
          }}>
            {TIME_WINDOWS.map(w => (
              <option key={w.value} value={w.value}>{w.label}</option>
            ))}
          </select>
          <select value={sourceFilter} onChange={e => setSourceFilter(e.target.value)} style={{
            padding: '4px 8px', fontSize: 11, borderRadius: 6,
            background: 'var(--bg-tertiary)', color: 'var(--text-primary)',
            border: '1px solid var(--border-color)', cursor: 'pointer',
          }}>
            <option value="all">All sources</option>
            <option value="search">Search only</option>
            <option value="search+profiling">Search+Profiling</option>
            <option value="profiling_only">Profiling only</option>
          </select>
          <button onClick={handleRefresh} disabled={refreshing} style={{
            padding: '4px 12px', fontSize: 12, borderRadius: 6,
            background: 'var(--bg-secondary)', color: 'var(--text-primary)',
            border: '1px solid var(--border-color)', cursor: 'pointer',
            opacity: refreshing ? 0.5 : 1,
          }}>
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </div>

      <HealthSummary health={health} />

      {/* Component Grid with filters */}
      <div className="card" style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
          <div className="card-title" style={{ margin: 0 }}>Component Health Grid</div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <input type="text" placeholder="Search ops..." value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              style={{
                padding: '4px 10px', fontSize: 12, borderRadius: 6, width: 160,
                background: 'var(--bg-tertiary)', color: 'var(--text-primary)',
                border: '1px solid var(--border-color)', outline: 'none',
              }}
            />
            {['all', 'broken', 'degraded', 'healthy'].map(f => (
              <button key={f} onClick={() => setFilter(f)} style={{
                padding: '3px 10px', fontSize: 11, borderRadius: 12, cursor: 'pointer',
                background: filter === f ? (STATUS_COLORS[f] || 'var(--accent-blue)') + '22' : 'var(--bg-tertiary)',
                color: filter === f ? (STATUS_COLORS[f] || 'var(--accent-blue)') : 'var(--text-muted)',
                border: `1px solid ${filter === f ? (STATUS_COLORS[f] || 'var(--accent-blue)') + '44' : 'var(--border-color)'}`,
                fontWeight: filter === f ? 600 : 400, textTransform: 'capitalize',
              }}>
                {f}{f !== 'all' && health ? ` (${health[f] || 0})` : ''}
              </button>
            ))}
          </div>
        </div>
        <ComponentGrid components={health?.components} filter={filter} searchTerm={searchTerm} sourceFilter={sourceFilter} />
      </div>

      <OpPairHeatmap pairs={opPairs} />
      <LossDistributionPanel distributions={lossDist} />
      <FailureBlocklist blocklist={blocklist} />
      <FailurePatternPanel patterns={failurePatterns} />
      <GrammarEvolutionPanel events={grammarEvents} />
      <LeaderboardDynamicsPanel daily={leaderboardData.daily} recentPromotions={leaderboardData.recent_promotions} />
      <InsightEffectivenessPanel insights={insightData} />
    </div>
  );
}
