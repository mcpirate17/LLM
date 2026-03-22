import React, { useState, useEffect, useCallback, useRef } from 'react';
import { apiCall } from '../services/apiService';

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
function AlertsPanel({ alerts }) {
  if (!alerts || alerts.length === 0) {
    return (
      <div className="card" style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
          <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>Alerts</span>
          <span style={{
            padding: '2px 8px', borderRadius: 10, fontSize: 11,
            background: '#22c55e22', color: '#22c55e', fontWeight: 600,
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
              color: SEVERITY_COLORS[alert.severity], minWidth: 52, marginTop: 2,
            }}>{alert.severity}</span>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{alert.title}</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>{alert.message}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Live Training Stream with exponential backoff SSE ───
function LiveStream({ sseData, connected }) {
  if (!sseData) {
    return (
      <div className="card" style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="card-title" style={{ margin: 0 }}>Live Training Stream</span>
          <span style={{
            width: 6, height: 6, borderRadius: '50%',
            background: connected ? '#22c55e' : '#ef4444',
          }} />
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {connected ? 'Waiting for data' : 'Disconnected'}
          </span>
        </div>
        <p className="ux-state ux-state-empty">No active training session.</p>
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
          width: 6, height: 6, borderRadius: '50%',
          background: connected ? '#22c55e' : '#ef4444',
          animation: connected ? 'pulse 2s infinite' : 'none',
        }} />
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          {connected ? 'SSE connected' : 'Reconnecting...'}
        </span>
      </div>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12 }}>
        <div><span style={{ color: 'var(--text-muted)' }}>Status: </span><span style={{ fontWeight: 600 }}>{prog.status || 'unknown'}</span></div>
        <div><span style={{ color: 'var(--text-muted)' }}>Program: </span><span style={{ fontWeight: 600 }}>{prog.current_program || 0}/{prog.total_programs || '?'}</span></div>
        <div><span style={{ color: 'var(--text-muted)' }}>S0/S1: </span><span style={{ fontWeight: 600 }}>{prog.stage0_passed || 0}/{prog.stage1_passed || 0}</span></div>
        {prog.best_loss_ratio != null && (
          <div><span style={{ color: 'var(--text-muted)' }}>Best LR: </span><span style={{ fontWeight: 600, color: 'var(--accent-green)' }}>{prog.best_loss_ratio.toFixed(4)}</span></div>
        )}
        {prog.elapsed_seconds != null && (
          <div><span style={{ color: 'var(--text-muted)' }}>Elapsed: </span><span style={{ fontWeight: 600 }}>{Math.floor(prog.elapsed_seconds / 60)}m</span></div>
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

// ─── Error Log Panel ───
function ErrorLogPanel({ errors }) {
  if (!errors || errors.length === 0) return null;
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Recent Errors</div>
      <div style={{ maxHeight: 260, overflowY: 'auto' }}>
        {errors.slice(0, 20).map(e => (
          <div key={e.id} style={{
            padding: '6px 0', borderBottom: '1px solid var(--border-color)',
            fontSize: 12,
          }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <span style={{
                padding: '1px 6px', borderRadius: 4, fontSize: 10, fontWeight: 600,
                background: '#ef444422', color: '#ef4444',
              }}>{e.event_type}</span>
              <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>
                {new Date(e.timestamp * 1000).toLocaleTimeString()}
              </span>
            </div>
            <div style={{ color: 'var(--text-primary)', marginTop: 2 }}>
              {e.description?.slice(0, 120)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Experiment Lifecycle Panel ───
function ExperimentLifecyclePanel({ experiments, onCleanup }) {
  if (!experiments || experiments.length === 0) return null;

  const statusColor = {
    running: '#3b82f6',
    completed: '#22c55e',
    failed: '#ef4444',
    aborted: '#eab308',
    interrupted: '#f97316',
  };

  const orphans = experiments.filter(e => e.orphan);

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <div className="card-title" style={{ margin: 0 }}>Experiment Lifecycle</div>
        {orphans.length > 0 && (
          <button onClick={onCleanup} style={{
            padding: '3px 10px', fontSize: 11, borderRadius: 6,
            background: '#ef444422', color: '#ef4444', border: '1px solid #ef444444',
            cursor: 'pointer', fontWeight: 600,
          }}>
            Cleanup {orphans.length} orphan{orphans.length > 1 ? 's' : ''}
          </button>
        )}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table" style={{ fontSize: 12 }}>
          <thead>
            <tr>
              <th>Status</th>
              <th>Type</th>
              <th>Programs</th>
              <th>S0/S1</th>
              <th>Best LR</th>
              <th>Duration</th>
            </tr>
          </thead>
          <tbody>
            {experiments.slice(0, 15).map(exp => (
              <tr key={exp.experiment_id} style={{
                background: exp.orphan ? '#ef444408' : undefined,
              }}>
                <td>
                  <span style={{
                    display: 'inline-block', padding: '1px 6px', borderRadius: 4,
                    fontSize: 10, fontWeight: 600,
                    background: (statusColor[exp.status] || '#888') + '22',
                    color: statusColor[exp.status] || '#888',
                  }}>
                    {exp.status}{exp.orphan ? ' (orphan)' : ''}
                  </span>
                </td>
                <td>{exp.experiment_type}</td>
                <td style={{ textAlign: 'right' }}>{exp.n_programs_generated || 0}</td>
                <td style={{ textAlign: 'right' }}>{exp.n_stage0_passed || 0}/{exp.n_stage1_passed || 0}</td>
                <td style={{ textAlign: 'right', fontFamily: 'monospace' }}>
                  {exp.best_loss_ratio != null ? exp.best_loss_ratio.toFixed(4) : '-'}
                </td>
                <td style={{ textAlign: 'right', color: 'var(--text-muted)' }}>
                  {exp.duration_seconds != null ? `${Math.round(exp.duration_seconds / 60)}m` : exp.orphan ? `${exp.running_hours}h` : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Throughput Panel ───
function ThroughputPanel({ throughput }) {
  if (!throughput) return null;
  const windows = ['1h', '6h', '24h'];
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Throughput</div>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        {windows.map(w => {
          const d = throughput[w];
          if (!d) return null;
          return (
            <div key={w} style={{
              flex: '1 1 140px', padding: '10px 14px', borderRadius: 8,
              background: 'var(--bg-secondary)', border: '1px solid var(--border-color)',
            }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{w} window</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }}>{d.total}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                S0: {(d.s0_rate * 100).toFixed(0)}% &middot; S1: {(d.s1_rate * 100).toFixed(1)}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Resource Utilization Card ───
function ResourceUtilizationCard({ resources }) {
  if (!resources) return null;
  const items = [
    { label: 'CPU', value: resources.cpu_percent != null ? `${resources.cpu_percent.toFixed(0)}%` : 'N/A' },
    { label: 'RAM', value: resources.ram_percent != null ? `${resources.ram_percent.toFixed(0)}% (${resources.ram_used_gb}/${resources.ram_total_gb} GB)` : 'N/A' },
    { label: 'GPU Alloc', value: resources.gpu_allocated_gb != null ? `${resources.gpu_allocated_gb} GB` : 'N/A' },
    { label: 'GPU Reserved', value: resources.gpu_reserved_gb != null ? `${resources.gpu_reserved_gb} GB` : 'N/A' },
  ];
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Resource Utilization</div>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        {items.map(item => (
          <div key={item.label} style={{
            flex: '1 1 120px', padding: '10px 14px', borderRadius: 8,
            background: 'var(--bg-secondary)', border: '1px solid var(--border-color)',
          }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 2 }}>{item.label}</div>
            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>{item.value}</div>
          </div>
        ))}
      </div>
      {resources.gpu_name && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>GPU: {resources.gpu_name}</div>
      )}
    </div>
  );
}

// ─── API Health Card ───
function ApiHealthCard({ counters }) {
  if (!counters || Object.keys(counters).length === 0) return null;

  // Aggregate by endpoint
  const byEndpoint = {};
  for (const [key, count] of Object.entries(counters)) {
    const lastColon = key.lastIndexOf(':');
    const path = key.substring(0, lastColon);
    const bucket = key.substring(lastColon + 1);
    if (!byEndpoint[path]) byEndpoint[path] = { '2xx': 0, '4xx': 0, '5xx': 0 };
    byEndpoint[path][bucket] = (byEndpoint[path][bucket] || 0) + count;
  }

  const sorted = Object.entries(byEndpoint)
    .sort((a, b) => {
      const aErr = (a[1]['4xx'] || 0) + (a[1]['5xx'] || 0);
      const bErr = (b[1]['4xx'] || 0) + (b[1]['5xx'] || 0);
      return bErr - aErr;
    })
    .slice(0, 15);

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">API Health</div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table" style={{ fontSize: 12 }}>
          <thead>
            <tr><th>Endpoint</th><th>2xx</th><th>4xx</th><th>5xx</th></tr>
          </thead>
          <tbody>
            {sorted.map(([path, counts]) => (
              <tr key={path}>
                <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{path}</td>
                <td style={{ textAlign: 'right', color: '#22c55e' }}>{counts['2xx'] || 0}</td>
                <td style={{ textAlign: 'right', color: counts['4xx'] > 0 ? '#eab308' : 'var(--text-muted)' }}>{counts['4xx'] || 0}</td>
                <td style={{ textAlign: 'right', color: counts['5xx'] > 0 ? '#ef4444' : 'var(--text-muted)' }}>{counts['5xx'] || 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── DB Health Card ───
function DbHealthCard({ dbHealth }) {
  if (!dbHealth) return null;
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="card-title">Database Health</div>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 8 }}>
        <div style={{ padding: '8px 14px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>DB Size</div>
          <div style={{ fontSize: 14, fontWeight: 600 }}>{dbHealth.db_size_mb != null ? `${dbHealth.db_size_mb} MB` : 'N/A'}</div>
        </div>
        <div style={{ padding: '8px 14px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>WAL Size</div>
          <div style={{ fontSize: 14, fontWeight: 600 }}>{dbHealth.wal_size_mb != null ? `${dbHealth.wal_size_mb} MB` : 'N/A'}</div>
        </div>
      </div>
      {dbHealth.row_counts && (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {Object.entries(dbHealth.row_counts).map(([table, count]) => (
            <span key={table} style={{
              padding: '2px 8px', borderRadius: 4, fontSize: 11,
              background: 'var(--bg-tertiary)', color: 'var(--text-muted)',
            }}>
              {table}: <strong style={{ color: 'var(--text-primary)' }}>{count != null ? count.toLocaleString() : '?'}</strong>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main Infrastructure Dashboard ───
export default function InfrastructureDashboard() {
  const [alerts, setAlerts] = useState([]);
  const [sseData, setSseData] = useState(null);
  const [sseConnected, setSseConnected] = useState(false);
  const [errors, setErrors] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [throughput, setThroughput] = useState(null);
  const [resources, setResources] = useState(null);
  const [apiHealth, setApiHealth] = useState({});
  const [dbHealth, setDbHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const sseRef = useRef(null);
  const backoffRef = useRef(1000);
  const heartbeatRef = useRef(Date.now());
  const reconnectRef = useRef(null);

  const fetchData = useCallback(async () => {
    try {
      const [alertRes, errorRes, expRes, tpRes, resRes, ahRes, dbRes] = await Promise.all([
        apiCall('/api/observability/alerts'),
        apiCall('/api/observability/error-log'),
        apiCall('/api/observability/experiment-lifecycle'),
        apiCall('/api/observability/throughput'),
        apiCall('/api/observability/resource-utilization'),
        apiCall('/api/observability/api-health'),
        apiCall('/api/observability/db-health'),
      ]);
      if (alertRes.ok) { const d = await alertRes.json(); setAlerts(d.alerts || []); }
      if (errorRes.ok) { const d = await errorRes.json(); setErrors(d.errors || []); }
      if (expRes.ok) { const d = await expRes.json(); setExperiments(d.experiments || []); }
      if (tpRes.ok) setThroughput(await tpRes.json());
      if (resRes.ok) setResources(await resRes.json());
      if (ahRes.ok) { const d = await ahRes.json(); setApiHealth(d.counters || {}); }
      if (dbRes.ok) setDbHealth(await dbRes.json());
    } catch (err) {
      console.error('Infrastructure fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // SSE with exponential backoff
  const connectSSE = useCallback(() => {
    if (sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
    }

    const es = new EventSource('/api/observability/stream');
    sseRef.current = es;

    es.addEventListener('progress', (e) => {
      try {
        setSseData(JSON.parse(e.data));
        heartbeatRef.current = Date.now();
      } catch {}
    });
    es.addEventListener('alerts', (e) => {
      try {
        const d = JSON.parse(e.data);
        if (d.alerts) setAlerts(d.alerts);
        heartbeatRef.current = Date.now();
      } catch {}
    });
    es.addEventListener('keepalive', () => {
      heartbeatRef.current = Date.now();
    });

    es.onopen = () => {
      setSseConnected(true);
      backoffRef.current = 1000; // Reset backoff on success
    };

    es.onerror = () => {
      setSseConnected(false);
      es.close();
      sseRef.current = null;
      // Exponential backoff: 1s, 2s, 4s, ... up to 30s
      const delay = backoffRef.current;
      backoffRef.current = Math.min(delay * 2, 30000);
      reconnectRef.current = setTimeout(connectSSE, delay);
    };
  }, []);

  useEffect(() => {
    fetchData();
    connectSSE();
    const interval = setInterval(fetchData, 30000);

    // Heartbeat staleness check
    const heartbeatInterval = setInterval(() => {
      if (Date.now() - heartbeatRef.current > 10000) {
        setSseConnected(false);
      }
    }, 5000);

    return () => {
      clearInterval(interval);
      clearInterval(heartbeatInterval);
      if (sseRef.current) sseRef.current.close();
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
    };
  }, [fetchData, connectSSE]);

  const handleCleanup = useCallback(async () => {
    try {
      await apiCall('/api/observability/experiment-lifecycle/cleanup', { method: 'POST' });
      fetchData();
    } catch (err) {
      console.error('Cleanup error:', err);
    }
  }, [fetchData]);

  if (loading) {
    return <div className="card"><p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Loading infrastructure data...</p></div>;
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>Infrastructure</h2>
        <AlertBadge alerts={alerts} />
      </div>

      <AlertsPanel alerts={alerts} />
      <LiveStream sseData={sseData} connected={sseConnected} />
      <ThroughputPanel throughput={throughput} />
      <ErrorLogPanel errors={errors} />
      <ExperimentLifecyclePanel experiments={experiments} onCleanup={handleCleanup} />
      <ResourceUtilizationCard resources={resources} />
      <ApiHealthCard counters={apiHealth} />
      <DbHealthCard dbHealth={dbHealth} />
    </div>
  );
}
