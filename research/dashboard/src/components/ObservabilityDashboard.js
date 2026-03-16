import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { apiCall } from '../services/apiService';

/**
 * ObservabilityDashboard — Real-time pipeline health, component grid, alerts, SSE stream.
 *
 * P0: Component health grid (all ops color-coded by health)
 * P1: SSE training stream (replaces polling)
 * P2: Alert framework with thresholds
 * P3: Failure blocklist view
 */

const STATUS_COLORS = {
  healthy: '#22c55e',
  degraded: '#eab308',
  broken: '#ef4444',
};

const SEVERITY_COLORS = {
  critical: '#ef4444',
  warning: '#eab308',
  info: '#3b82f6',
};

// ─── Alert Badge ───
function AlertBadge({ alerts }) {
  if (!alerts || alerts.length === 0) return null;
  const critical = alerts.filter(a => a.severity === 'critical').length;
  const color = critical > 0 ? SEVERITY_COLORS.critical : SEVERITY_COLORS.warning;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
      background: color + '22', color,
    }}>
      {alerts.length} alert{alerts.length !== 1 ? 's' : ''}
    </span>
  );
}

// ─── Alerts Panel ───
function AlertsPanel({ alerts, thresholds }) {
  if (!alerts || alerts.length === 0) {
    return (
      <div className="card" style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
          <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>Alerts</span>
          <span style={{
            padding: '2px 8px', borderRadius: 10, fontSize: 11,
            background: STATUS_COLORS.healthy + '22', color: STATUS_COLORS.healthy,
            fontWeight: 600,
          }}>All clear</span>
        </div>
      </div>
    );
  }

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>Alerts</span>
        <AlertBadge alerts={alerts} />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {alerts.map((alert, i) => (
          <div key={alert.id + i} style={{
            display: 'flex', alignItems: 'flex-start', gap: 10, padding: '8px 12px',
            borderRadius: 6, background: SEVERITY_COLORS[alert.severity] + '0a',
            borderLeft: `3px solid ${SEVERITY_COLORS[alert.severity]}`,
          }}>
            <span style={{
              fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
              color: SEVERITY_COLORS[alert.severity], minWidth: 52,
              marginTop: 2,
            }}>{alert.severity}</span>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                {alert.title}
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                {alert.message}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Summary Cards ───
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

// ─── Component Health Grid ───
function ComponentGrid({ components, filter, searchTerm }) {
  const filtered = useMemo(() => {
    let list = components || [];
    if (filter !== 'all') list = list.filter(c => c.status === filter);
    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      list = list.filter(c => c.op.toLowerCase().includes(q));
    }
    return list;
  }, [components, filter, searchTerm]);

  if (filtered.length === 0) {
    return <p className="ux-state ux-state-empty">No components match the current filter.</p>;
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="data-table" style={{ fontSize: 12 }}>
        <thead>
          <tr>
            <th>Status</th>
            <th>Op Name</th>
            <th>Used</th>
            <th>S0 Rate</th>
            <th>S1 Rate</th>
            <th>Grad Norm</th>
            <th>Fwd (us)</th>
            <th>Issues</th>
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

// ─── Live Training Stream ───
function LiveStream({ sseData }) {
  if (!sseData) {
    return (
      <div className="card" style={{ marginBottom: 12 }}>
        <div className="card-title">Live Training Stream</div>
        <p className="ux-state ux-state-empty">No active training session. Stream connects automatically when training starts.</p>
      </div>
    );
  }

  const prog = sseData;
  const lossTail = prog.loss_curve_tail || [];

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span className="card-title" style={{ margin: 0 }}>Live Training Stream</span>
        <span style={{
          width: 6, height: 6, borderRadius: '50%', background: '#22c55e',
          animation: 'pulse 2s infinite',
        }} />
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>SSE connected</span>
      </div>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12 }}>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Status: </span>
          <span style={{ fontWeight: 600 }}>{prog.status || 'unknown'}</span>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Program: </span>
          <span style={{ fontWeight: 600 }}>{prog.current_program || 0}/{prog.total_programs || '?'}</span>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>S0/S1: </span>
          <span style={{ fontWeight: 600 }}>{prog.stage0_passed || 0}/{prog.stage1_passed || 0}</span>
        </div>
        {prog.best_loss_ratio != null && (
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Best LR: </span>
            <span style={{ fontWeight: 600, color: 'var(--accent-green)' }}>{prog.best_loss_ratio.toFixed(4)}</span>
          </div>
        )}
        {prog.elapsed_seconds != null && (
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Elapsed: </span>
            <span style={{ fontWeight: 600 }}>{Math.floor(prog.elapsed_seconds / 60)}m</span>
          </div>
        )}
      </div>
      {lossTail.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Recent loss (last {lossTail.length} steps)</div>
          <div style={{
            display: 'flex', alignItems: 'flex-end', gap: 1, height: 32,
            background: 'var(--bg-tertiary)', borderRadius: 4, padding: 2,
          }}>
            {(() => {
              const vals = lossTail.map(p => p.loss ?? p);
              const mn = Math.min(...vals);
              const mx = Math.max(...vals);
              const range = mx - mn || 1;
              return vals.map((v, i) => (
                <div key={i} style={{
                  flex: 1, background: 'var(--accent-blue)',
                  borderRadius: 1, opacity: 0.7,
                  height: `${Math.max(2, ((v - mn) / range) * 28)}px`,
                }} />
              ));
            })()}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Failure Blocklist ───
function FailureBlocklist({ blocklist }) {
  if (!blocklist || Object.keys(blocklist).length === 0) {
    return null;
  }

  const entries = Object.entries(blocklist)
    .sort((a, b) => a[1] - b[1])
    .slice(0, 20);

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Failure Blocklist (auto-disabled op pairs)</div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table" style={{ fontSize: 12 }}>
          <thead>
            <tr>
              <th>Op Pair Signature</th>
              <th>Penalty</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([sig, penalty]) => (
              <tr key={sig}>
                <td style={{ fontFamily: 'monospace' }}>{sig}</td>
                <td style={{
                  textAlign: 'right', fontWeight: 600,
                  color: penalty === 0 ? STATUS_COLORS.broken : STATUS_COLORS.degraded,
                }}>
                  {penalty === 0 ? 'BLOCKED' : `${(penalty * 100).toFixed(0)}%`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Main Dashboard ───
export default function ObservabilityDashboard() {
  const [health, setHealth] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [thresholds, setThresholds] = useState({});
  const [blocklist, setBlocklist] = useState({});
  const [sseData, setSseData] = useState(null);
  const [filter, setFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const sseRef = useRef(null);

  // Fetch health + alerts
  const fetchData = useCallback(async () => {
    try {
      const [healthRes, alertRes, blockRes] = await Promise.all([
        apiCall('/api/observability/health'),
        apiCall('/api/observability/alerts'),
        apiCall('/api/observability/failure-blocklist'),
      ]);
      if (healthRes.ok) setHealth(await healthRes.json());
      if (alertRes.ok) {
        const data = await alertRes.json();
        setAlerts(data.alerts || []);
        setThresholds(data.thresholds || {});
      }
      if (blockRes.ok) {
        const data = await blockRes.json();
        setBlocklist(data.blocklist || {});
      }
    } catch (err) {
      console.error('Observability fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Force refresh
  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await apiCall('/api/observability/health/refresh', { method: 'POST' });
      await fetchData();
    } finally {
      setRefreshing(false);
    }
  }, [fetchData]);

  // Initial fetch + 30s poll
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // SSE connection for live training stream
  useEffect(() => {
    const es = new EventSource('/api/observability/stream');
    sseRef.current = es;

    es.addEventListener('progress', (e) => {
      try { setSseData(JSON.parse(e.data)); } catch {}
    });
    es.addEventListener('alerts', (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.alerts) setAlerts(data.alerts);
      } catch {}
    });
    es.onerror = () => {
      setSseData(null);
    };

    return () => {
      es.close();
      sseRef.current = null;
    };
  }, []);

  if (loading) {
    return <div className="card"><p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Loading observability data...</p></div>;
  }

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
            Pipeline Observability
          </h2>
          <AlertBadge alerts={alerts} />
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          style={{
            padding: '4px 12px', fontSize: 12, borderRadius: 6,
            background: 'var(--bg-secondary)', color: 'var(--text-primary)',
            border: '1px solid var(--border-color)', cursor: 'pointer',
            opacity: refreshing ? 0.5 : 1,
          }}
        >
          {refreshing ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {/* Alerts */}
      <AlertsPanel alerts={alerts} thresholds={thresholds} />

      {/* Live Stream */}
      <LiveStream sseData={sseData} />

      {/* Health Summary */}
      <HealthSummary health={health} />

      {/* Component Grid */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
          <div className="card-title" style={{ margin: 0 }}>Component Health Grid</div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <input
              type="text"
              placeholder="Search ops..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              style={{
                padding: '4px 10px', fontSize: 12, borderRadius: 6, width: 160,
                background: 'var(--bg-tertiary)', color: 'var(--text-primary)',
                border: '1px solid var(--border-color)', outline: 'none',
              }}
            />
            {['all', 'broken', 'degraded', 'healthy'].map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                style={{
                  padding: '3px 10px', fontSize: 11, borderRadius: 12, cursor: 'pointer',
                  background: filter === f ? (STATUS_COLORS[f] || 'var(--accent-blue)') + '22' : 'var(--bg-tertiary)',
                  color: filter === f ? (STATUS_COLORS[f] || 'var(--accent-blue)') : 'var(--text-muted)',
                  border: `1px solid ${filter === f ? (STATUS_COLORS[f] || 'var(--accent-blue)') + '44' : 'var(--border-color)'}`,
                  fontWeight: filter === f ? 600 : 400,
                  textTransform: 'capitalize',
                }}
              >
                {f}{f !== 'all' && health ? ` (${health[f] || 0})` : ''}
              </button>
            ))}
          </div>
        </div>
        <ComponentGrid
          components={health?.components}
          filter={filter}
          searchTerm={searchTerm}
        />
      </div>

      {/* Failure Blocklist */}
      <FailureBlocklist blocklist={blocklist} />
    </div>
  );
}
