import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react';
import ChartActions from './ChartActions';

const PALETTE = [
  '#58a6ff', '#3fb950', '#d29922', '#bc8cff', '#f47067',
  '#39d2c0', '#e3b341', '#db61a2', '#79c0ff', '#7ee787',
];
const MAX_VISIBLE_POINTS_FOR_TOP20 = 120;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function normalize(v, lo, hi) {
  if (!Number.isFinite(v)) return 0;
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return 0;
  return ((v - lo) / (hi - lo)) * 2 - 1;
}

function finiteOrNull(v) {
  return Number.isFinite(Number(v)) ? Number(v) : null;
}

function noveltyValue(program) {
  const overall = finiteOrNull(program.novelty_score);
  if (overall != null) return overall;
  const structural = finiteOrNull(program.structural_novelty);
  const behavioral = finiteOrNull(program.behavioral_novelty);
  if (structural != null && behavioral != null) return Math.max(structural, behavioral);
  return structural ?? behavioral ?? 0;
}

function GlobalParetoChart({
  programs,
  title = 'Search Frontier: 3D Accuracy vs Novelty vs Size',
  onSelectProgram,
  onNavigateTab,
}) {
  const wrapRef = useRef(null);
  const dragRef = useRef(null);

  const [size, setSize] = useState({ width: 900, height: 340 });
  const [camera, setCamera] = useState({ yaw: 0.85, pitch: -0.45, zoom: 1.15 });
  const [hover, setHover] = useState(null);
  const [selectedIds, setSelectedIds] = useState([]);
  const [selectionRect, setSelectionRect] = useState(null);

  useEffect(() => {
    const node = wrapRef.current;
    if (!node) return undefined;
    const measure = () => {
      const rect = node.getBoundingClientRect();
      setSize((prev) => ({
        width: rect.width > 10 ? rect.width : (prev.width || 900),
        height: rect.height > 10 ? rect.height : (prev.height || 340),
      }));
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

  const survivors = useMemo(() => {
    return (programs || [])
      .filter((p) =>
        (p.stage1_passed || p.screening_passed || p.tier) &&
        p.loss_ratio != null &&
        p.param_count != null
      )
      .map((p) => {
        const accuracy = Math.max(0, 1 - Number(p.loss_ratio || 1));
        const paramsM = Number(p.param_count || 0) / 1e6;
        const noveltyAxis = noveltyValue(p);
        const explicitScore = finiteOrNull(p._score ?? p.composite_score ?? p.score);
        const fallbackScore = (accuracy * 100) + (noveltyAxis * 25) - paramsM * 0.25;
        const score = explicitScore != null ? explicitScore : fallbackScore;

        return {
          ...p,
          accuracy,
          params_m: paramsM,
          novelty_axis: noveltyAxis,
          score,
          family: p.architecture_family || 'Custom',
          name: (p.result_id || '').slice(0, 8),
        };
      });
  }, [programs]);

  const filteredFrontier = useMemo(() => {
    if (!survivors.length) return { points: [], percentile: 20 };
    const sorted = [...survivors].sort((a, b) => (b.score || 0) - (a.score || 0));
    const top20Count = Math.max(1, Math.ceil(sorted.length * 0.2));
    const top20 = sorted.slice(0, top20Count);
    if (top20.length > MAX_VISIBLE_POINTS_FOR_TOP20) {
      const top10Count = Math.max(1, Math.ceil(sorted.length * 0.1));
      return { points: sorted.slice(0, top10Count), percentile: 10 };
    }
    return { points: top20, percentile: 20 };
  }, [survivors]);

  const frontierPoints = filteredFrontier.points;
  const frontierPercentile = filteredFrontier.percentile;
  const chartTitle = `${title} (Top ${frontierPercentile}% by score)`;

const fingerprints = useMemo(() => Array.from(new Set(frontierPoints.map((p) => p.graph_fingerprint || 'unknown'))), [frontierPoints]);
  
  const fingerprintColors = useMemo(() => {
    const map = {};
    fingerprints.forEach((fp, i) => {
      map[fp] = PALETTE[i % PALETTE.length];
    });
    return map;
  }, [fingerprints]);

  const bounds = useMemo(() => {
    if (!frontierPoints.length) return null;
    const xs = frontierPoints.map((p) => p.params_m);
    const ys = frontierPoints.map((p) => p.accuracy);
    const zs = frontierPoints.map((p) => p.novelty_axis);
    const scores = frontierPoints.map((p) => p.score || 0);
    return {
      xMin: Math.min(...xs), xMax: Math.max(...xs),
      yMin: Math.min(...ys), yMax: Math.max(...ys),
      zMin: Math.min(...zs), zMax: Math.max(...zs),
      scoreMin: Math.min(...scores), scoreMax: Math.max(...scores),
    };
  }, [frontierPoints]);

  const points3d = useMemo(() => {
    if (!bounds) return [];
    return frontierPoints.map((p) => ({
      ...p,
      nx: normalize(p.params_m, bounds.xMin, bounds.xMax),
      ny: normalize(p.accuracy, bounds.yMin, bounds.yMax),
      nz: normalize(p.novelty_axis, bounds.zMin, bounds.zMax),
      nscore: normalize(p.score || 0, bounds.scoreMin, bounds.scoreMax),
      color: fingerprintColors[p.graph_fingerprint || 'unknown'] || '#58a6ff',
    }));
  }, [frontierPoints, bounds, fingerprintColors]);

  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);
  const selectedPoints = useMemo(
    () => frontierPoints.filter((p) => selectedSet.has(p.result_id)),
    [frontierPoints, selectedSet]
  );

  const projected = useMemo(() => {
    if (!points3d.length || size.width <= 0 || size.height <= 0) return { points: [], axes: [], origin: null };

    const cx = size.width * 0.5;
    const cy = size.height * 0.52;
    const scale = Math.min(size.width, size.height) * 0.28 * camera.zoom;

    const project = (x, y, z) => {
      const cosy = Math.cos(camera.yaw);
      const siny = Math.sin(camera.yaw);
      const cosp = Math.cos(camera.pitch);
      const sinp = Math.sin(camera.pitch);

      const x1 = x * cosy - z * siny;
      const z1 = x * siny + z * cosy;
      const y1 = y * cosp - z1 * sinp;
      const z2 = y * sinp + z1 * cosp;
      return { sx: cx + x1 * scale, sy: cy - y1 * scale, depth: z2 };
    };

    const origin = project(0, 0, 0);
    const axisDefs = [
      { label: 'Size (M params)', to: [1.5, 0, 0], color: '#58a6ff' },
      { label: 'Accuracy', to: [0, 1.5, 0], color: '#3fb950' },
      { label: 'Novelty', to: [0, 0, 1.5], color: '#d29922' },
    ];
    const axes = axisDefs.map((a) => ({ ...a, end: project(a.to[0], a.to[1], a.to[2]) }));

    const pts = points3d
      .map((p) => {
        const pr = project(p.nx, p.ny, p.nz);
        const chartDim = Math.min(size.width, size.height);
        const minR = Math.max(2.5, chartDim * 0.006);
        const maxR = Math.max(7.0, chartDim * 0.028);
        // p.nscore is between -1 and 1, remap it to 0-1
        const normalizedScore = (p.nscore + 1) / 2;
        let baseR = minR + (normalizedScore || 0) * (maxR - minR);
        let r = clamp(baseR * (1 + (pr.depth + 1.0) * 0.12), 1.5, maxR * 1.3);
        if (selectedSet.has(p.result_id)) r *= 1.3;
        return {
          ...p,
          ...pr,
          r,
        };
      })
      .sort((a, b) => a.depth - b.depth);

    return { points: pts, axes, origin };
  }, [points3d, size.width, size.height, camera, selectedSet]);

  const hitTest = useCallback((x, y) => {
    let best = null;
    let bestD2 = 999999;
    for (const p of projected.points) {
      const dx = p.sx - x;
      const dy = p.sy - y;
      const d2 = dx * dx + dy * dy;
      if (d2 < 110 && d2 < bestD2) {
        best = p;
        bestD2 = d2;
      }
    }
    return best;
  }, [projected.points]);

  const eventPos = useCallback((e) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }, []);

  const onMouseDown = useCallback((e) => {
    const { x, y } = eventPos(e);
    if (e.shiftKey) {
      dragRef.current = { mode: 'select', x, y, x2: x, y2: y, moved: false };
      setSelectionRect({ x1: x, y1: y, x2: x, y2: y });
      return;
    }
    dragRef.current = { mode: 'orbit', x, y, yaw: camera.yaw, pitch: camera.pitch, moved: false };
  }, [eventPos, camera]);

  const onMouseMove = useCallback((e) => {
    const d = dragRef.current;
    const { x, y } = eventPos(e);
    if (!d) {
      setHover(hitTest(x, y));
      return;
    }
    const dx = x - d.x;
    const dy = y - d.y;
    if (Math.abs(dx) > 2 || Math.abs(dy) > 2) d.moved = true;

    if (d.mode === 'select') {
      d.x2 = x;
      d.y2 = y;
      setSelectionRect({ x1: d.x, y1: d.y, x2: x, y2: y });
      return;
    }

    setCamera((prev) => ({
      ...prev,
      yaw: d.yaw + dx * 0.01,
      pitch: clamp(d.pitch + dy * 0.01, -1.2, 1.2),
    }));
  }, [eventPos, hitTest]);

  const onMouseUp = useCallback((e) => {
    const d = dragRef.current;
    const { x, y } = eventPos(e);
    dragRef.current = null;
    if (!d) return;

    if (d.mode === 'select') {
      const x1 = Math.min(d.x, d.x2 ?? x);
      const x2 = Math.max(d.x, d.x2 ?? x);
      const y1 = Math.min(d.y, d.y2 ?? y);
      const y2 = Math.max(d.y, d.y2 ?? y);
      setSelectionRect(null);
      if (!d.moved || (x2 - x1 < 4 && y2 - y1 < 4)) return;
      const selected = projected.points
        .filter((p) => p.sx >= x1 && p.sx <= x2 && p.sy >= y1 && p.sy <= y2 && p.result_id)
        .map((p) => p.result_id);
      setSelectedIds(Array.from(new Set(selected)));
      return;
    }

    if (!d.moved) {
      const p = hitTest(x, y);
      if (p?.result_id) onSelectProgram?.(p.result_id);
    }
  }, [eventPos, projected.points, hitTest, onSelectProgram]);

  const onWheel = useCallback((e) => {
    e.preventDefault();
    setCamera((prev) => ({
      ...prev,
      zoom: clamp(prev.zoom * (e.deltaY < 0 ? 1.1 : 0.9), 0.6, 3.2),
    }));
  }, []);

  const actions = useMemo(() => {
    if (!frontierPoints.length) return [];
    const sortedAcc = [...frontierPoints].sort((a, b) => b.accuracy - a.accuracy);
    const sortedNovelty = [...frontierPoints].sort((a, b) => b.novelty_axis - a.novelty_axis);
    const bestAcc = sortedAcc[0];
    const bestNovelty = sortedNovelty[0];
    const result = [];

    if (selectedPoints.length > 0) {
      const bestSelected = [...selectedPoints].sort((a, b) => b.accuracy - a.accuracy)[0];
      result.push({
        id: 'selected-best',
        label: `Inspect selected cluster (${selectedPoints.length})`,
        detail: `Best selected: ${bestSelected.name} @ ${(bestSelected.accuracy * 100).toFixed(1)}%`,
        color: 'var(--accent-green)',
        onClick: () => onSelectProgram?.(bestSelected.result_id),
      });
      result.push({
        id: 'selected-clear',
        label: 'Clear selection',
        detail: 'Reset current box selection',
        color: 'var(--accent-yellow)',
        onClick: () => setSelectedIds([]),
      });
    }

    if (bestAcc?.result_id) {
      result.push({
        id: 'best-acc',
        label: `Inspect highest accuracy (${bestAcc.name})`,
        detail: `${(bestAcc.accuracy * 100).toFixed(1)}% accuracy`,
        color: 'var(--accent-green)',
        onClick: () => onSelectProgram?.(bestAcc.result_id),
      });
    }
    if (bestNovelty?.result_id && bestNovelty.result_id !== bestAcc?.result_id) {
      result.push({
        id: 'best-novelty',
        label: `Inspect highest novelty (${bestNovelty.name})`,
        detail: `novelty=${bestNovelty.novelty_axis.toFixed(3)}`,
        color: 'var(--accent-blue)',
        onClick: () => onSelectProgram?.(bestNovelty.result_id),
      });
    }
    if (frontierPoints.length > 12) {
      result.push({
        id: 'open-discoveries',
        label: 'Open dense cluster in Discoveries',
        detail: `${frontierPoints.length} candidates in 3D frontier`,
        color: 'var(--accent-purple)',
        onClick: () => onNavigateTab?.('discoveries'),
      });
    }
    return result;
  }, [frontierPoints, selectedPoints, onSelectProgram, onNavigateTab]);

  if (!survivors.length) {
    return (
      <div className="card" style={{ height: 320, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p style={{ color: 'var(--text-muted)' }}>Not enough evaluated candidates for 3D frontier analysis.</p>
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: 16, maxWidth: '100%', overflowX: 'hidden' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600 }}>{chartTitle}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
            Drag to orbit. Scroll to zoom. Click a point to open that fingerprint. Hold Shift + drag to box-select a cluster.
          </div>
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          <button
            onClick={() => setCamera((prev) => ({ ...prev, zoom: clamp(prev.zoom * 1.15, 0.6, 3.2) }))}
            title="Zoom in"
            style={{ width: 28, height: 28, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 16, lineHeight: '26px', padding: 0 }}
          >+</button>
          <button
            onClick={() => setCamera((prev) => ({ ...prev, zoom: clamp(prev.zoom * 0.87, 0.6, 3.2) }))}
            title="Zoom out"
            style={{ width: 28, height: 28, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 16, lineHeight: '26px', padding: 0 }}
          >-</button>
          <button
            onClick={() => { setCamera({ yaw: 0.85, pitch: -0.45, zoom: 1.15 }); setSelectedIds([]); }}
            title="Reset view"
            style={{ height: 28, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 11, padding: '0 8px' }}
          >Reset</button>
        </div>
      </div>

      <div
        ref={wrapRef}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => { dragRef.current = null; setHover(null); setSelectionRect(null); }}
        onWheel={onWheel}
        style={{ position: 'relative', width: '100%', maxWidth: '100%', height: 360, border: '1px solid var(--border)', borderRadius: 8, background: 'rgba(0,0,0,0.15)', overflow: 'hidden', cursor: dragRef.current ? 'grabbing' : 'grab' }}
      >
        <svg width="100%" height="100%" style={{ display: 'block', overflow: 'hidden' }}>
          {projected.origin && projected.axes.map((a) => (
            <g key={a.label}>
              <line x1={projected.origin.sx} y1={projected.origin.sy} x2={a.end.sx} y2={a.end.sy} stroke={a.color} strokeWidth="1.4" />
              <text x={a.end.sx + 6} y={a.end.sy - 4} fill={a.color} fontSize="14" fontFamily="monospace">{a.label}</text>
            </g>
          ))}

          {projected.points.map((p) => (
            <circle
              key={p.result_id || `${p.sx}-${p.sy}`}
              cx={p.sx}
              cy={p.sy}
              r={p.r}
              fill={p.color}
              fillOpacity="0.86"
              stroke={selectedSet.has(p.result_id) ? '#e6edf3' : 'none'}
              strokeWidth={selectedSet.has(p.result_id) ? '1.2' : '0'}
            />
          ))}

          {selectionRect && (
            <rect
              x={Math.min(selectionRect.x1, selectionRect.x2)}
              y={Math.min(selectionRect.y1, selectionRect.y2)}
              width={Math.abs(selectionRect.x2 - selectionRect.x1)}
              height={Math.abs(selectionRect.y2 - selectionRect.y1)}
              fill="rgba(88,166,255,0.14)"
              stroke="var(--accent-blue)"
              strokeDasharray="4 4"
            />
          )}
        </svg>

        {hover && (
          <div
            style={{
              position: 'absolute',
              left: clamp(hover.sx + 12, 8, Math.max(8, size.width - 190)),
              top: clamp(hover.sy - 10, 8, Math.max(8, size.height - 110)),
              width: 180,
              background: '#161b22',
              border: '1px solid #30363d',
              borderRadius: 6,
              padding: '8px 10px',
              fontSize: 12,
              pointerEvents: 'none',
            }}
          >
            <div style={{ fontWeight: 700, color: 'var(--accent-blue)', marginBottom: 4 }}>{hover.name || 'Unknown'}</div>
            <div>Score: {hover.score?.toFixed(2) || 'N/A'}</div>
            <div>Fingerprint: {(hover.graph_fingerprint || 'unknown').substring(0, 8)}</div>
            <div>Family: {hover.family || 'Custom'}</div>
            <div>Accuracy: {(hover.accuracy * 100).toFixed(2)}%</div>
            <div>Novelty: {hover.novelty_axis.toFixed(3)}</div>
            <div>Size: {hover.params_m.toFixed(2)}M</div>
            <div style={{ marginTop: 4, fontSize: 10, color: 'var(--accent-blue)' }}>Click to open fingerprint</div>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', gap: 12, marginTop: 8, flexWrap: 'wrap', maxHeight: 60, overflowY: 'auto' }}>
        {Object.entries(fingerprintColors).map(([fp, col]) => (
          <div key={fp} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--text-muted)' }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: col }} />
            {fp.substring(0, 8)}
          </div>
        ))}
      </div>

      {bounds && (
        <div style={{ marginTop: 6, fontSize: 10, color: 'var(--text-muted)', display: 'flex', gap: 14, flexWrap: 'wrap' }}>
          <span>Showing: top {frontierPercentile}% ({frontierPoints.length}/{survivors.length})</span>
          <span>Size: {bounds.xMin.toFixed(2)}M to {bounds.xMax.toFixed(2)}M</span>
          <span>Accuracy: {(bounds.yMin * 100).toFixed(1)}% to {(bounds.yMax * 100).toFixed(1)}%</span>
          <span>Novelty: {bounds.zMin.toFixed(2)} to {bounds.zMax.toFixed(2)}</span>
        </div>
      )}

      <ChartActions
        isZoomed={Math.abs(camera.zoom - 1.15) > 0.01}
        onResetZoom={() => { setCamera({ yaw: 0.85, pitch: -0.45, zoom: 1.15 }); setSelectedIds([]); }}
        actions={actions}
      />
    </div>
  );
}

export default GlobalParetoChart;
