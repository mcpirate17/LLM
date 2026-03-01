import React, { useMemo } from 'react';
import { 
  ScatterChart, 
  Scatter, 
  XAxis, 
  YAxis, 
  ZAxis,
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell,
  ReferenceLine,
  Line,
  ComposedChart
} from 'recharts';

function GlobalParetoChart({ programs, title = "Search Frontier: Accuracy vs Efficiency" }) {
  // 1. Filter only S1 survivors and valid metrics
  const survivors = useMemo(() => {
    return (programs || []).filter(p =>
      (p.stage1_passed || p.screening_passed || p.tier) &&
      p.loss_ratio != null &&
      p.param_count != null
    ).map(p => ({
      ...p,
      accuracy: Math.max(0, 1 - p.loss_ratio),
      params_m: p.param_count / 1e6,
      name: p.result_id?.slice(0, 8),
      family: p.architecture_family || 'Custom'
    }));
  }, [programs]);

  // 2. Calculate Pareto Front
  const frontier = useMemo(() => {
    if (survivors.length === 0) return [];
    
    // Sort by params (ascending)
    const sorted = [...survivors].sort((a, b) => a.params_m - b.params_m);
    
    const front = [];
    let maxAccSoFar = -1;
    
    for (const p of sorted) {
      if (p.accuracy > maxAccSoFar) {
        front.push({ x: p.params_m, y: p.accuracy, result_id: p.result_id });
        maxAccSoFar = p.accuracy;
      }
    }
    
    // Convert to step-wise line points
    const stepFront = [];
    for (let i = 0; i < front.length; i++) {
      if (i > 0) {
        // Horizontal step
        stepFront.push({ x: front[i].x, y: front[i-1].y });
      }
      stepFront.push(front[i]);
    }
    return stepFront;
  }, [survivors]);

  if (survivors.length === 0) {
    return (
      <div className="card" style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p style={{ color: 'var(--text-muted)' }}>Not enough evaluated candidates for Pareto analysis.</p>
      </div>
    );
  }

  const families = Array.from(new Set(survivors.map(p => p.family)));
  const PALETTE = [
    '#58a6ff', '#3fb950', '#d29922', '#bc8cff', '#f47067',
    '#39d2c0', '#e3b341', '#db61a2', '#79c0ff', '#7ee787',
  ];
  const familyColors = {};
  families.forEach((fam, i) => {
    familyColors[fam] = PALETTE[i % PALETTE.length];
  });

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      if (!data || data.params_m == null) return null;
      return (
        <div style={{ background: '#161b22', border: '1px solid #30363d', padding: '8px 12px', borderRadius: 6, fontSize: 12 }}>
          <div style={{ fontWeight: 700, marginBottom: 4, color: 'var(--accent-blue)' }}>{data.name || 'Unknown'}</div>
          <div>Family: {data.family || 'Custom'}</div>
          <div>Accuracy: {((data.accuracy || 0) * 100).toFixed(1)}%</div>
          <div>Params: {(data.params_m || 0).toFixed(1)}M</div>
          {data.compression_ratio != null && <div>Comp: {data.compression_ratio.toFixed(2)}x</div>}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12 }}>
        Accuracy (Y) vs Model Size (X). Red dashed line = Pareto Front (optimal tradeoff).
      </div>
      <div style={{ height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
            <XAxis 
              type="number" 
              dataKey="x" 
              name="Params (M)" 
              unit="M"
              domain={['auto', 'auto']}
              tick={{ fontSize: 10, fill: '#8b949e' }}
              label={{ value: 'Parameters (Millions)', position: 'bottom', offset: 0, style: { fill: '#8b949e', fontSize: 10 } }}
            />
            <YAxis 
              type="number" 
              dataKey="y" 
              name="Accuracy" 
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: '#8b949e' }}
              label={{ value: 'Accuracy (1-LR)', angle: -90, position: 'insideLeft', style: { fill: '#8b949e', fontSize: 10 } }}
            />
            <Tooltip content={<CustomTooltip />} />
            
            {/* The scatter points */}
            <Scatter name="Architectures" data={survivors.map(p => ({ x: p.params_m, y: p.accuracy, ...p }))}>
              {survivors.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={familyColors[entry.family] || familyColors['Custom']} />
              ))}
            </Scatter>

            {/* The Pareto Front line */}
            <Line
              type="stepAfter"
              data={frontier}
              dataKey="y"
              stroke="var(--accent-red)"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              activeDot={false}
              legendType="none"
              tooltipType="none"
            />
            
            {/* Target reference line */}
            <ReferenceLine y={0.8} label={{ value: 'Target', position: 'right', fill: 'var(--accent-green)', fontSize: 10 }} stroke="var(--accent-green)" strokeDasharray="3 3" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div style={{ display: 'flex', gap: 12, marginTop: 8, flexWrap: 'wrap' }}>
        {Object.entries(familyColors).map(([fam, col]) => (
          <div key={fam} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--text-muted)' }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: col }} />
            {fam}
          </div>
        ))}
      </div>
    </div>
  );
}

export default GlobalParetoChart;
