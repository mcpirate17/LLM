import React, { useMemo } from 'react';
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

  // Transform experiments into chart data
  // We reverse to show oldest -> newest (left -> right)
  const chartData = [...experiments].reverse().map((exp, i) => {
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

  return (
    <div className="card" style={{ gridColumn: '1 / -1' }}>
      <div className="card-title">Discovery Performance</div>
      <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.5 }}>
        Tracks search efficiency over time. 
        <strong> Survivors</strong> (bars) show raw counts. 
        <strong style={{ color: '#3fb950' }}> Yield</strong> (line) is the % of programs that successfully learned.
        <strong style={{ color: '#bc8cff' }}> Novelty</strong> (line) tracks architectural innovation.
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
