import React, { useMemo, useCallback, useEffect, useRef, useState } from 'react';
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
  ReferenceArea,
  Line,
  ComposedChart
} from 'recharts';
import { useChartInteraction } from '../hooks/useChartInteraction';
import ChartActions from './ChartActions';

function GlobalParetoChart({ programs, title = "Search Frontier: Accuracy vs Efficiency", onSelectProgram, onNavigateTab }) {
  const chartWrapRef = useRef(null);
  const [chartLayout, setChartLayout] = useState({ width: 0, height: 300 });

  useEffect(() => {
    const node = chartWrapRef.current;
    if (!node) return undefined;

    const measure = () => {
      const rect = node.getBoundingClientRect();
      setChartLayout({ width: rect.width || 0, height: rect.height || 300 });
    };
    measure();

    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(measure);
      ro.observe(node);
      return () => ro.disconnect();
    }

    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, []);

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

  // 3. Zoom + click interaction
  const zoom = useChartInteraction({
    onSelectPoint: (data) => {
      if (onSelectProgram && data?.result_id) onSelectProgram(data.result_id);
    },
    chartLayout,
  });

  // 4. Contextual actions
  const actions = useMemo(() => {
    if (survivors.length === 0) return [];
    const result = [];

    // Find non-step frontier points (the actual Pareto-optimal models)
    const frontierIds = new Set(frontier.filter(f => f.result_id).map(f => f.result_id));

    // Near-frontier: within 5% of Pareto accuracy at their param count
    const nearFrontier = survivors.filter(p => {
      if (frontierIds.has(p.result_id)) return false;
      // Find best frontier accuracy at or above this param count
      const frontierAtParam = frontier.filter(f => f.result_id && f.x <= p.params_m);
      if (frontierAtParam.length === 0) return false;
      const bestAcc = Math.max(...frontierAtParam.map(f => f.y));
      return p.accuracy >= bestAcc * 0.95;
    });
    if (nearFrontier.length > 0) {
      const best = nearFrontier.sort((a, b) => b.accuracy - a.accuracy)[0];
      result.push({
        id: 'near-frontier',
        label: `Investigate ${nearFrontier.length} near-frontier model${nearFrontier.length > 1 ? 's' : ''}`,
        detail: `Best: ${best.name} (${(best.accuracy * 100).toFixed(1)}% acc)`,
        color: 'var(--accent-blue)',
        onClick: () => onSelectProgram?.(best.result_id),
      });
    }

    // Frontier at screening tier
    const frontierScreening = survivors.filter(p =>
      frontierIds.has(p.result_id) && (p.tier === 'screening' || !p.tier)
    );
    if (frontierScreening.length > 0) {
      result.push({
        id: 'frontier-screening',
        label: `Queue ${frontierScreening.length} frontier for investigation`,
        detail: 'Pareto-optimal models still at screening tier',
        color: 'var(--accent-green)',
        onClick: () => onSelectProgram?.(frontierScreening[0].result_id),
      });
    }

    // Dense cluster detection
    if (survivors.length > 5) {
      const paramsSorted = [...survivors].sort((a, b) => a.params_m - b.params_m);
      let maxCluster = 0, clusterCenter = 0;
      for (let i = 0; i < paramsSorted.length - 4; i++) {
        const window = paramsSorted.slice(i, i + 5);
        const span = window[4].params_m - window[0].params_m;
        const avgParams = window.reduce((s, p) => s + p.params_m, 0) / 5;
        if (span < avgParams * 0.2 && 5 > maxCluster) {
          maxCluster = 5;
          clusterCenter = avgParams;
        }
      }
      if (maxCluster > 0) {
        result.push({
          id: 'dense-cluster',
          label: `Explore cluster at ~${clusterCenter.toFixed(1)}M params`,
          detail: `${maxCluster}+ models in a narrow range`,
          color: 'var(--accent-purple)',
          onClick: () => onNavigateTab?.('discoveries'),
        });
      }
    }

    return result;
  }, [survivors, frontier, onSelectProgram, onNavigateTab]);

  // Data bounds for programmatic zoom
  const dataBounds = useMemo(() => {
    if (survivors.length === 0) return null;
    const xs = survivors.map(p => p.params_m);
    const ys = survivors.map(p => p.accuracy);
    const pad = 0.05; // 5% padding
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xPad = (xMax - xMin) * pad || 0.5;
    const yPad = (yMax - yMin) * pad || 0.05;
    return { xMin: xMin - xPad, xMax: xMax + xPad, yMin: Math.max(0, yMin - yPad), yMax: Math.min(1, yMax + yPad) };
  }, [survivors]);

  const handleZoomIn = useCallback(() => zoom.zoomBy(0.5, dataBounds), [zoom, dataBounds]);
  const handleZoomOut = useCallback(() => zoom.zoomBy(2.0, dataBounds), [zoom, dataBounds]);
  const handleWheelZoom = useCallback((e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 0.85 : 1.15;
    zoom.zoomBy(factor, dataBounds);
  }, [zoom, dataBounds]);

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

  const scatterData = survivors.map(p => ({ x: p.params_m, y: p.accuracy, ...p }));

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
          {onSelectProgram && <div style={{ marginTop: 4, fontSize: 10, color: 'var(--accent-blue)' }}>Click to view details</div>}
        </div>
      );
    }
    return null;
  };

  const xDomain = zoom.domain?.x || ['auto', 'auto'];
  const yDomain = zoom.domain?.y || [0, 1];

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600 }}>{title}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
            Accuracy (Y) vs Model Size (X). Red dashed line = Pareto Front (optimal tradeoff).
          </div>
        </div>
        <div style={{ display: 'flex', gap: 4, flexShrink: 0 }}>
          <button
            onClick={handleZoomIn}
            title="Zoom in"
            style={{
              width: 28, height: 28, borderRadius: 6, border: '1px solid var(--border)',
              background: 'var(--bg-secondary)', color: 'var(--text-secondary)',
              cursor: 'pointer', fontSize: 16, lineHeight: '26px', textAlign: 'center', padding: 0,
            }}
          >+</button>
          <button
            onClick={handleZoomOut}
            title="Zoom out"
            style={{
              width: 28, height: 28, borderRadius: 6, border: '1px solid var(--border)',
              background: 'var(--bg-secondary)', color: 'var(--text-secondary)',
              cursor: 'pointer', fontSize: 16, lineHeight: '26px', textAlign: 'center', padding: 0,
            }}
          >−</button>
          {zoom.isZoomed && (
            <button
              onClick={zoom.resetZoom}
              title="Reset zoom"
              style={{
                height: 28, borderRadius: 6, border: '1px solid var(--border)',
                background: 'var(--bg-secondary)', color: 'var(--text-secondary)',
                cursor: 'pointer', fontSize: 10, padding: '0 8px', whiteSpace: 'nowrap',
              }}
            >Fit All</button>
          )}
        </div>
      </div>
      <div
        ref={chartWrapRef}
        style={{ height: 300, cursor: zoom.isZoomed ? (zoom.isPanning ? 'grabbing' : 'grab') : 'crosshair' }}
        onWheel={handleWheelZoom}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            margin={{ top: 10, right: 20, bottom: 20, left: 0 }}
            onMouseDown={zoom.onMouseDown}
            onMouseMove={zoom.onMouseMove}
            onMouseUp={zoom.onMouseUp}
            onMouseLeave={zoom.onMouseUp}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
            <XAxis
              type="number"
              dataKey="x"
              name="Params (M)"
              unit="M"
              domain={xDomain}
              allowDataOverflow={zoom.isZoomed}
              tick={{ fontSize: 10, fill: '#8b949e' }}
              label={{ value: 'Parameters (Millions)', position: 'bottom', offset: 0, style: { fill: '#8b949e', fontSize: 10 } }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Accuracy"
              domain={yDomain}
              allowDataOverflow={zoom.isZoomed}
              tick={{ fontSize: 10, fill: '#8b949e' }}
              label={{ value: 'Accuracy (1-LR)', angle: -90, position: 'insideLeft', style: { fill: '#8b949e', fontSize: 10 } }}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Drag selection rectangle */}
            {zoom.refArea && (
              <ReferenceArea
                x1={zoom.refArea.x1}
                x2={zoom.refArea.x2}
                y1={zoom.refArea.y1}
                y2={zoom.refArea.y2}
                strokeOpacity={0.3}
                fill="var(--accent-blue)"
                fillOpacity={0.15}
              />
            )}

            {/* The scatter points */}
            <Scatter
              name="Architectures"
              data={scatterData}
              cursor={onSelectProgram ? 'pointer' : 'default'}
              onClick={(pointData) => {
                if (zoom.consumeClickGuard()) return;
                if (pointData?.result_id) zoom.handlePointClick(pointData);
              }}
            >
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
      <ChartActions
        isZoomed={zoom.isZoomed}
        onResetZoom={zoom.resetZoom}
        actions={actions}
      />
    </div>
  );
}

export default GlobalParetoChart;
