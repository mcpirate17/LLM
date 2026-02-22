import React, { useState, useEffect, useMemo } from 'react';
import { formatTime, formatDuration } from '../utils/format';
import apiService from '../services/apiService';
import { filterRowsByQuery } from '../utils/tableFiltering';
import { CHART_DEFAULTS, clampToScale, getFixedScale } from '../utils/chartScales';

const PERF_CHART_WINDOW = 30;

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
  const scale = getFixedScale(`perf.${scaleKey}`, values, {
    defaultMin: defaults.min,
    defaultMax: defaults.max,
  });
  const min = scale.min;
  const max = scale.max;
  const range = max - min || 1;

  const denom = Math.max(1, PERF_CHART_WINDOW - 1);
  const points = windowed
    .map((d, i) => {
      const v = d[valueKey];
      if (v == null || v === 0) return null;
      const x = PAD + (i / denom) * (W - 2 * PAD);
      const clamped = clampToScale(v, scale);
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

  if (loading) return <div className="card">Loading performance metrics...</div>;
  if (error) return <div className="card error">{error}</div>;

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
            suffix=" t/s"
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

        {hotspots.length > 0 && (
          <div className="hotspots-section">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 12 }}>
              <div style={{ fontSize: 14, fontWeight: 600 }}>Latest Kernel Hotspots</div>
              <input
                value={filterQuery}
                onChange={(e) => setFilterQuery(e.target.value)}
                placeholder="Filter kernels"
                style={{
                  fontSize: 11,
                  padding: '4px 8px',
                  borderRadius: 4,
                  border: '1px solid var(--border)',
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  minWidth: 160,
                }}
              />
            </div>
            <table className="data-table">
              <thead>
                <tr>
                  <th onClick={() => handleSort('op')} style={{ cursor: 'pointer', userSelect: 'none' }}>
                    Operation / Kernel{sortKey === 'op' && (
                      <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>
                    )}
                  </th>
                  <th onClick={() => handleSort('avg_cuda_ms')} style={{ cursor: 'pointer', userSelect: 'none' }}>
                    Avg CUDA ms{sortKey === 'avg_cuda_ms' && (
                      <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>
                    )}
                  </th>
                  <th onClick={() => handleSort('avg_cpu_ms')} style={{ cursor: 'pointer', userSelect: 'none' }}>
                    Avg CPU ms{sortKey === 'avg_cpu_ms' && (
                      <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>
                    )}
                  </th>
                  <th onClick={() => handleSort('avg_calls')} style={{ cursor: 'pointer', userSelect: 'none' }}>
                    Avg Calls{sortKey === 'avg_calls' && (
                      <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>
                    )}
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedHotspots.map((h, i) => (
                  <tr key={i}>
                    <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{h.op}</td>
                    <td style={{ color: h.avg_cuda_ms > 1.0 ? 'var(--accent-orange)' : 'inherit' }}>
                        {h.avg_cuda_ms.toFixed(3)}ms
                    </td>
                    <td>{h.avg_cpu_ms.toFixed(3)}ms</td>
                    <td>{h.avg_calls}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default PerfDashboard;
