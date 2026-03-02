import React, { useMemo, useRef, useCallback, useState } from 'react';
import { CHART_DEFAULTS, clampToScale, getFixedScale } from '../../utils/chartScales';
import ChartActions from '../ChartActions';

export default function EfficiencyChart({ frontier, showLabels = false, labelCount = 5, onSelectProgram }) {
  const frontierRows = Array.isArray(frontier) ? frontier : [];
  const svgRef = useRef(null);
  const [dragState, setDragState] = useState(null); // {px1,py1,px2,py2}
  const [zoomDomain, setZoomDomain] = useState(null); // {lMin,lMax,fMin,fMax}
  const dragging = useRef(false);

  const W = 700, H = 260;
  const pad = { l: 60, r: 20, t: 20, b: 35 };

  const losses = frontierRows.map(p => p.final_loss || p.loss_ratio || 0).filter(l => isFinite(l));
  const flops = frontierRows.map(p => Math.log10(Math.max(p.flops_forward || p.param_count || 1, 1)));

  const lossDefaults = CHART_DEFAULTS.loss_ratio;
  const flopsDefaults = CHART_DEFAULTS.efficiency_log_flops;
  const lossScale = getFixedScale('efficiency.loss_ratio', losses, {
    defaultMin: lossDefaults.min,
    defaultMax: lossDefaults.max,
  });
  const flopsScale = getFixedScale('efficiency.log_flops', flops, {
    defaultMin: flopsDefaults.min,
    defaultMax: flopsDefaults.max,
  });

  const minL = zoomDomain ? zoomDomain.lMin : lossScale.min;
  const maxL = zoomDomain ? zoomDomain.lMax : lossScale.max;
  const minF = zoomDomain ? zoomDomain.fMin : flopsScale.min;
  const maxF = zoomDomain ? zoomDomain.fMax : flopsScale.max;
  const rangeL = maxL - minL || 1, rangeF = maxF - minF || 1;

  const xScale = v => pad.l + ((clampToScale(Math.log10(Math.max(v, 1)), { min: minF, max: maxF }) - minF) / rangeF) * (W - pad.l - pad.r);
  const yScale = v => H - pad.b - ((clampToScale(v, { min: minL, max: maxL }) - minL) / rangeL) * (H - pad.t - pad.b);

  // Inverse scales for drag-to-zoom
  const xPixelToData = useCallback((px) => minF + ((px - pad.l) / (W - pad.l - pad.r)) * rangeF, [minF, rangeF]);
  const yPixelToData = useCallback((py) => minL + ((H - pad.b - py) / (H - pad.t - pad.b)) * rangeL, [minL, rangeL]);

  const getSvgCoords = useCallback((e) => {
    if (!svgRef.current) return null;
    const rect = svgRef.current.getBoundingClientRect();
    const scaleX = W / rect.width;
    const scaleY = H / rect.height;
    return { px: (e.clientX - rect.left) * scaleX, py: (e.clientY - rect.top) * scaleY };
  }, []);

  const onMouseDown = useCallback((e) => {
    const coords = getSvgCoords(e);
    if (!coords) return;
    dragging.current = true;
    setDragState({ px1: coords.px, py1: coords.py, px2: coords.px, py2: coords.py });
  }, [getSvgCoords]);

  const onMouseMove = useCallback((e) => {
    if (!dragging.current) return;
    const coords = getSvgCoords(e);
    if (!coords) return;
    setDragState(prev => prev ? { ...prev, px2: coords.px, py2: coords.py } : null);
  }, [getSvgCoords]);

  const onMouseUp = useCallback(() => {
    dragging.current = false;
    if (!dragState) return;
    const { px1, py1, px2, py2 } = dragState;
    if (Math.abs(px2 - px1) < 5 && Math.abs(py2 - py1) < 5) { setDragState(null); return; }
    const f1 = xPixelToData(Math.min(px1, px2));
    const f2 = xPixelToData(Math.max(px1, px2));
    const l1 = yPixelToData(Math.max(py1, py2));
    const l2 = yPixelToData(Math.min(py1, py2));
    setZoomDomain({ lMin: l1, lMax: l2, fMin: f1, fMax: f2 });
    setDragState(null);
  }, [dragState, xPixelToData, yPixelToData]);

  const resetZoom = useCallback(() => { setZoomDomain(null); setDragState(null); }, []);

  const labelCandidates = useMemo(() => {
    if (!showLabels) return [];
    return [...frontierRows]
      .filter(p => p.graph_fingerprint)
      .sort((a, b) => (a.final_loss || a.loss_ratio || 0) - (b.final_loss || b.loss_ratio || 0))
      .slice(0, labelCount);
  }, [frontierRows, showLabels, labelCount]);

  // Contextual actions
  const actions = useMemo(() => {
    if (frontierRows.length === 0) return [];
    const result = [];
    const best = [...frontierRows].sort((a, b) => (a.final_loss || a.loss_ratio || 1) - (b.final_loss || b.loss_ratio || 1))[0];
    if (best) {
      const fp = (best.graph_fingerprint || '').slice(0, 8);
      result.push({
        id: 'best-model',
        label: `Best: ${fp || best.result_id?.slice(0, 8)} \u2014 click to view`,
        color: 'var(--accent-green)',
        onClick: () => onSelectProgram?.(best.result_id),
      });
    }
    // High-FLOPs outliers: models with >2x median FLOPs but not on Pareto front
    const medianFlops = [...flops].sort((a, b) => a - b)[Math.floor(flops.length / 2)] || 0;
    const inefficient = frontierRows.filter(p => Math.log10(Math.max(p.flops_forward || p.param_count || 1, 1)) > medianFlops * 1.5);
    if (inefficient.length > 1) {
      result.push({
        id: 'high-flops',
        label: `${inefficient.length} inefficient models \u2014 consider pruning`,
        color: 'var(--accent-orange)',
        onClick: () => onSelectProgram?.(inefficient[0].result_id),
      });
    }
    return result;
  }, [frontierRows, flops, onSelectProgram]);

  if (frontierRows.length === 0) return <p style={{ color: 'var(--text-muted)' }}>No Pareto-optimal programs yet.</p>;
  if (losses.length < 2 || flops.length < 2) return null;

  return (
    <div>
      <svg ref={svgRef} width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', cursor: 'crosshair' }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => { dragging.current = false; setDragState(null); }}
      >
        <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="var(--border)" />
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="var(--border)" />
        <text x={W / 2} y={H - 5} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>log10(FLOPs / Params)</text>
        <text x={12} y={H / 2} textAnchor="middle" fill="var(--text-muted)" fontSize={10} transform={`rotate(-90, 12, ${H / 2})`}>Loss</text>

        {/* Drag selection rectangle */}
        {dragState && (
          <rect
            x={Math.min(dragState.px1, dragState.px2)}
            y={Math.min(dragState.py1, dragState.py2)}
            width={Math.abs(dragState.px2 - dragState.px1)}
            height={Math.abs(dragState.py2 - dragState.py1)}
            fill="var(--accent-blue)"
            fillOpacity={0.15}
            stroke="var(--accent-blue)"
            strokeOpacity={0.4}
            strokeWidth={1}
          />
        )}

        {frontierRows.map((p, i) => {
          const x = xScale(p.flops_forward || p.param_count || 0);
          const y = yScale(p.final_loss || p.loss_ratio || 0);
          if (!isFinite(x) || !isFinite(y)) return null;
          return (
            <circle key={i} cx={x} cy={y} r={5}
              fill="var(--accent-purple)" opacity={0.7}
              stroke="var(--bg-secondary)" strokeWidth={1.5}
              style={{ cursor: onSelectProgram ? 'pointer' : 'default' }}
              onClick={(e) => { e.stopPropagation(); onSelectProgram?.(p.result_id); }}>
              <title>{p.graph_fingerprint?.slice(0, 10)}: loss={p.final_loss || p.loss_ratio}{onSelectProgram ? ' — click to view' : ''}</title>
            </circle>
          );
        })}
        {labelCandidates.map((p, i) => {
          const x = xScale(p.flops_forward || p.param_count || 0);
          const y = yScale(p.final_loss || p.loss_ratio || 0);
          if (!isFinite(x) || !isFinite(y)) return null;
          return (
            <text key={`label-${i}`} x={x + 8} y={y - 6} fontSize={9} fill="var(--text-secondary)">
              {(p.graph_fingerprint || '').slice(0, 8)}
            </text>
          );
        })}
      </svg>
      <ChartActions
        isZoomed={zoomDomain != null}
        onResetZoom={resetZoom}
        actions={actions}
      />
    </div>
  );
}
