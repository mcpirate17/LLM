import React, { useState, useEffect, useMemo } from 'react';
import { formatTime, formatDuration } from '../utils/format';
import apiService from '../services/apiService';
import { filterRowsByQuery } from '../utils/tableFiltering';
import { CHART_DEFAULTS } from '../utils/chartScales';

const PERF_CHART_WINDOW = 30;

function quantile(sorted, q) {
  if (!sorted.length) return null;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  const next = sorted[base + 1] ?? sorted[base];
  return sorted[base] + rest * (next - sorted[base]);
}

function buildRobustScale(values, defaults) {
  const nums = (values || [])
    .map(v => Number(v))
    .filter(v => Number.isFinite(v) && v > 0)
    .sort((a, b) => a - b);

  if (!nums.length) {
    return { min: defaults.min, max: defaults.max };
  }

  let min = nums[0];
  let max = nums[nums.length - 1];

  // For small trend windows, trim tail outliers so one spike doesn't flatten the chart.
  if (nums.length >= 6) {
    const q10 = quantile(nums, 0.1);
    const q90 = quantile(nums, 0.9);
    if (Number.isFinite(q10)) min = Math.min(min, q10);
    if (Number.isFinite(q90)) max = Math.max(q90, min + 1);
  }

  // Keep non-negative baseline for perf charts.
  min = Math.max(0, min);
  if (max <= min) max = min + 1;
  return { min, max };
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function MiniPerfChart({ data, valueKey, label, color, formatValue, suffix = '', scaleKey }) {
  if (!data || data.length < 2) {
    return (
      <div style={{ textAlign: 'center', padding: 16, color: 'var(--text-muted)', fontSize: 13 }}>
        Need at least 2 experiments for {label} trend
      </div>
    );
  }

  const windowed = data.slice(-PERF_CHART_WINDOW);
  const values = windowed.map(d => d[valueKey]).filter(v => v != null && v > 0);
  if (values.length < 2) return (
    <div style={{ textAlign: 'center', padding: 16, color: 'var(--text-muted)', fontSize: 13 }}>
      Insufficient perf data for {label}
    </div>
  );

  const W = 400;
  const H = 100;
  const PAD = 20;

  const defaults = CHART_DEFAULTS[scaleKey] || { min: 0, max: 1 };
  const scale = buildRobustScale(values, defaults);
  const min = scale.min;
  const max = scale.max;
  const range = max - min || 1;

  const denom = Math.max(1, windowed.length - 1);
  const points = windowed
    .map((d, i) => {
      const v = d[valueKey];
      if (v == null || v === 0) return null;
      const x = PAD + (i / denom) * (W - 2 * PAD);
      const clamped = clamp(v, min, max);
      const y = H - PAD - ((clamped - min) / range) * (H - 2 * PAD);
      return { x, y, v: clamped, idx: i };
    })
    .filter(Boolean);

  if (points.length < 2) return null;

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
  const fmt = formatValue || (v => v.toFixed(2));

  return (
    <div className="stat-card" style={{ flex: '1 1 300px', minWidth: 300 }}>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600 }}>
        {label}
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        {[0, 0.5, 1].map(frac => {
          const y = H - PAD - frac * (H - 2 * PAD);
          return (
            <g key={frac}>
              <line x1={PAD} y1={y} x2={W - PAD} y2={y} stroke="var(--border)" strokeWidth={0.5} strokeDasharray="2 2" />
              <text x={0} y={y + 3} fontSize={8} fill="var(--text-muted)">{fmt(min + frac * range)}{suffix}</text>
            </g>
          );
        })}
        <path d={pathD} fill="none" stroke={color} strokeWidth={2} />
        {points.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={3} fill={color}>
            <title>Exp #{p.idx + 1}: {fmt(p.v)}{suffix}</title>
          </circle>
        ))}
      </svg>
    </div>
  );
}

function PerfDashboard() {
  const [trends, setTrends] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterQuery, setFilterQuery] = useState('');
  const [sortKey, setSortKey] = useState('avg_cuda_ms');
  const [sortDesc, setSortDesc] = useState(true);

  useEffect(() => {
    apiService.getTrends()
      .then(data => {
        setTrends(data?.trends || []);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const hotspots = useMemo(() => {
    // Extract hotspots from latest experiment if available
    if (!trends.length) return [];
    const latest = trends[trends.length - 1];
    try {
        const results = JSON.parse(latest.results_json);
        return results?.perf_report?.kernel_hotspots || [];
    } catch {
        return [];
    }
  }, [trends]);

  const filteredHotspots = useMemo(() => (
    filterRowsByQuery(hotspots, filterQuery, ['op'])
  ), [hotspots, filterQuery]);

  const sortedHotspots = useMemo(() => {
    const arr = [...filteredHotspots];
    arr.sort((a, b) => {
      const va = a?.[sortKey];
      const vb = b?.[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') {
        return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      }
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [filteredHotspots, sortKey, sortDesc]);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDesc(!sortDesc);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  if (loading) {
    return (
      <div className="card">
        <div className="ux-state ux-state-loading">
          <span className="ux-spinner" />
          <div className="ux-stack">
            <span className="ux-state-title">Loading performance metrics</span>
            <span className="ux-state-subtle">Aggregating kernel hotspots and hardware utilization data.</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="ux-state ux-state-error">
          <span style={{ fontSize: 18, fontWeight: 700 }}>!</span>
          <div className="ux-stack">
            <span className="ux-state-title">Failed to load performance data</span>
            <span className="ux-state-subtle">{error}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="perf-dashboard">
      <div className="card">
        <div className="card-title">Zero-Overhead Optimization Metrics</div>
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 20 }}>
          Tracking system efficiency, kernel hotspots, and hardware utilization over time.
        </p>
        
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 24 }}>
          <MiniPerfChart 
            data={trends} 
            valueKey="avg_step_time_ms" 
            label="Avg Step Latency" 
            color="var(--accent-blue, #58a6ff)" 
            suffix="ms"
            scaleKey="step_time_ms"
          />
          <MiniPerfChart 
            data={trends} 
            valueKey="avg_throughput_tok_s" 
            label="Throughput" 
            color="var(--accent-green, #3fb950)" 
            formatValue={(v) => Math.round(v).toLocaleString()}
            suffix=" tok/s"
            scaleKey="throughput_tok_s"
          />
          <MiniPerfChart 
            data={trends} 
            valueKey="gpu_starvation_ms" 
            label="GPU Starvation" 
            color="var(--accent-red, #f85149)" 
            suffix="ms"
            scaleKey="gpu_starvation_ms"
          />
        </div>

        <div className="hotspots-section">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 12 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
              Latest Kernel Hotspots
              {hotspots.length > 0 && (
                <span style={{ marginLeft: 8, fontSize: 12, fontWeight: 400, color: 'var(--text-muted)' }}>
                  ({hotspots.length} ops)
                </span>
              )}
            </div>
            {hotspots.length > 0 && (
              <input
                type="search"
                value={filterQuery}
                onChange={(e) => setFilterQuery(e.target.value)}
                placeholder="Filter kernels..."
                aria-label="Filter kernel hotspots"
                style={{
                  fontSize: 12,
                  padding: '5px 8px',
                  borderRadius: 4,
                  border: '1px solid var(--border)',
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  minWidth: 160,
                }}
              />
            )}
          </div>

          {hotspots.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">&#x23F1;</div>
              <div className="empty-state-title">No kernel profiling data yet</div>
              <p className="empty-state-hint">
                Hotspot data appears after the first experiment completes with profiling enabled.
              </p>
            </div>
          ) : sortedHotspots.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">&#x2205;</div>
              <div className="empty-state-title">No kernels match "{filterQuery}"</div>
              <p className="empty-state-hint">Try a different filter term.</p>
            </div>
          ) : (
            <table className="data-table" aria-label="Kernel hotspots">
              <thead>
                <tr>
                  {[
                    { key: 'op', label: 'Operation / Kernel' },
                    { key: 'avg_cuda_ms', label: 'Avg CUDA ms' },
                    { key: 'avg_cpu_ms', label: 'Avg CPU ms' },
                    { key: 'avg_calls', label: 'Avg Calls' },
                  ].map(col => (
                    <th
                      key={col.key}
                      className="th-sortable"
                      scope="col"
                      aria-sort={sortKey === col.key ? (sortDesc ? 'descending' : 'ascending') : 'none'}
                      onClick={() => handleSort(col.key)}
                    >
                      {col.label}
                      {sortKey === col.key && (
                        <span className="th-sort-icon" aria-hidden="true">
                          {sortDesc ? '\u25BC' : '\u25B2'}
                        </span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedHotspots.map((h, i) => (
                  <tr key={i}>
                    <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{h.op}</td>
                    <td style={{ color: h.avg_cuda_ms > 1.0 ? 'var(--accent-orange)' : 'inherit' }}>
                      {h.avg_cuda_ms.toFixed(3)} ms
                    </td>
                    <td>{h.avg_cpu_ms.toFixed(3)} ms</td>
                    <td>{h.avg_calls}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

export default PerfDashboard;
