import React, { useMemo, useState } from 'react';
import { 
  ComposedChart, 
  Bar, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Legend 
} from 'recharts';
import { CHART_DEFAULTS, getFixedScale } from '../utils/chartScales';

function MetricsChart({ experiments }) {
  const [statusFilter, setStatusFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  const [outcomeFilter, setOutcomeFilter] = useState('all');
  const [windowSize, setWindowSize] = useState('20');

  const experimentTypes = useMemo(() => {
    if (!experiments || experiments.length === 0) return [];
    const unique = Array.from(new Set(
      experiments
        .map((exp) => exp?.experiment_type)
        .filter((value) => typeof value === 'string' && value.trim().length > 0)
    ));
    unique.sort((a, b) => a.localeCompare(b));
    return unique;
  }, [experiments]);

  const filteredExperiments = useMemo(() => (
    (experiments || []).filter((exp) => {
      if (statusFilter !== 'all' && exp.status !== statusFilter) return false;
      if (typeFilter !== 'all' && exp.experiment_type !== typeFilter) return false;
      if (outcomeFilter === 'has_s1' && (exp.n_stage1_passed || 0) <= 0) return false;
      if (outcomeFilter === 'no_s1' && (exp.n_stage1_passed || 0) > 0) return false;
      return true;
    })
  ), [experiments, statusFilter, typeFilter, outcomeFilter]);

  const hasActiveFilters = statusFilter !== 'all' || typeFilter !== 'all' || outcomeFilter !== 'all';
  const clearFilters = () => {
    setStatusFilter('all');
    setTypeFilter('all');
    setOutcomeFilter('all');
  };

  // Transform experiments into chart data
  // We reverse to show oldest -> newest (left -> right)
  const maxPoints = Number(windowSize);
  const scopedExperiments = Number.isFinite(maxPoints) && maxPoints > 0
    ? filteredExperiments.slice(0, maxPoints)
    : filteredExperiments;

  const chartData = [...scopedExperiments].reverse().map((exp, i) => {
    const n = exp.n_programs_generated || 0;
    const s1 = exp.n_stage1_passed || 0;
    const yield_rate = n > 0 ? (s1 / n) * 100 : 0;
    const novelty = (exp.best_novelty_score || 0) * 100;
    
    return {
      name: exp.experiment_id?.slice(0, 6) || `E${i}`,
      survivors: s1,
      yield: Number(yield_rate.toFixed(1)),
      novelty: Number(novelty.toFixed(1)),
      total: n,
    };
  });

  const survivorsValues = chartData.map(d => d.survivors);
  const survivorScale = getFixedScale('metrics.survivors', survivorsValues, {
    defaultMin: 0,
    defaultMax: 20,
  });

  if (!experiments || experiments.length === 0) {
    return (
      <div className="card" style={{ gridColumn: '1 / -1' }}>
        <div className="card-title">Discovery Performance</div>
        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
          No experiment data to chart yet.
        </p>
      </div>
    );
  }

  if (filteredExperiments.length === 0) {
    return (
      <div className="card" style={{ gridColumn: '1 / -1' }}>
        <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span>Discovery Performance</span>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              aria-label="Chart filter by status"
            >
              <option value="all">All status</option>
              <option value="completed">Completed</option>
              <option value="running">Running</option>
              <option value="failed">Failed</option>
            </select>
            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              aria-label="Chart filter by experiment type"
            >
              <option value="all">All types</option>
              {experimentTypes.map((type) => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
            <select
              value={outcomeFilter}
              onChange={(e) => setOutcomeFilter(e.target.value)}
              style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              aria-label="Chart filter by outcome"
            >
              <option value="all">All outcomes</option>
              <option value="has_s1">Has S1 pass</option>
              <option value="no_s1">No S1 pass</option>
            </select>
            <button
              className="refresh-btn"
              style={{ fontSize: 11, padding: '3px 10px' }}
              onClick={clearFilters}
              disabled={!hasActiveFilters}
            >
              Clear filters
            </button>
          </div>
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
          No experiments match the current chart filters.
        </p>
      </div>
    );
  }

  return (
    <div className="card" style={{ gridColumn: '1 / -1' }}>
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <span>Discovery Performance</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            aria-label="Chart filter by status"
          >
            <option value="all">All status</option>
            <option value="completed">Completed</option>
            <option value="running">Running</option>
            <option value="failed">Failed</option>
          </select>
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            aria-label="Chart filter by experiment type"
          >
            <option value="all">All types</option>
            {experimentTypes.map((type) => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
          <select
            value={outcomeFilter}
            onChange={(e) => setOutcomeFilter(e.target.value)}
            style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            aria-label="Chart filter by outcome"
          >
            <option value="all">All outcomes</option>
            <option value="has_s1">Has S1 pass</option>
            <option value="no_s1">No S1 pass</option>
          </select>
          <select
            value={windowSize}
            onChange={(e) => setWindowSize(e.target.value)}
            style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            aria-label="Chart window size"
          >
            <option value="10">10 points</option>
            <option value="20">20 points</option>
            <option value="50">50 points</option>
            <option value="100">100 points</option>
          </select>
          <button
            className="refresh-btn"
            style={{ fontSize: 11, padding: '3px 10px' }}
            onClick={clearFilters}
            disabled={!hasActiveFilters}
          >
            Clear filters
          </button>
        </div>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.5 }}>
        Tracks search efficiency over time. 
        <strong> Survivors</strong> (bars) show raw counts. 
        <strong style={{ color: '#3fb950' }}> Yield</strong> (line) is the % of programs that successfully learned.
        <strong style={{ color: '#bc8cff' }}> Novelty</strong> (line) tracks architectural innovation.
      </p>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
        Showing {chartData.length} of {filteredExperiments.length} filtered experiments.
      </p>
      <div className="chart-container" style={{ height: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
            <XAxis 
              dataKey="name" 
              tick={{ fontSize: 10, fill: '#8b949e' }} 
              axisLine={{ stroke: '#30363d' }}
              tickLine={false}
            />
            <YAxis
              yAxisId="left"
              domain={[0, 'auto']}
              tick={{ fontSize: 10, fill: '#8b949e' }}
              axisLine={{ stroke: '#30363d' }}
              tickLine={false}
              label={{ value: 'Survivors', angle: -90, position: 'insideLeft', style: { fill: '#8b949e', fontSize: 10 } }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              domain={[0, 100]}
              tick={{ fontSize: 10, fill: '#8b949e' }}
              axisLine={{ stroke: '#30363d' }}
              tickLine={false}
              label={{ value: '% / Novelty', angle: 90, position: 'insideRight', style: { fill: '#8b949e', fontSize: 10 } }}
            />
            <Tooltip
              contentStyle={{
                background: '#161b22',
                border: '1px solid #30363d',
                borderRadius: 8,
                fontSize: 12,
              }}
              cursor={{ fill: 'rgba(255,255,255,0.05)' }}
            />
            <Legend 
              wrapperStyle={{ fontSize: 11, paddingTop: 10 }}
              iconType="circle"
            />
            <Bar 
              yAxisId="left"
              dataKey="survivors" 
              fill="#58a6ff" 
              name="Survivors" 
              radius={[4, 4, 0, 0]}
              barSize={20}
            />
            <Line 
              yAxisId="right"
              type="monotone" 
              dataKey="yield" 
              stroke="#3fb950" 
              name="Yield Rate (%)" 
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
            <Line 
              yAxisId="right"
              type="monotone" 
              dataKey="novelty" 
              stroke="#bc8cff" 
              name="Peak Novelty" 
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default MetricsChart;
