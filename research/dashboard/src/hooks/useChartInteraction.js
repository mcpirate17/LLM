import { useState, useCallback, useRef } from 'react';

/**
 * useChartInteraction — click-drag-to-zoom for recharts ComposedChart/ScatterChart.
 *
 * Returns state + handlers to wire onto recharts chart props and axis domains.
 *
 * Usage:
 *   const z = useChartInteraction({ onSelectPoint });
 *   <ComposedChart onMouseDown={z.onMouseDown} onMouseMove={z.onMouseMove} onMouseUp={z.onMouseUp}>
 *     <XAxis domain={z.domain?.x || ['auto','auto']} allowDataOverflow={z.isZoomed} />
 *     <YAxis domain={z.domain?.y || [0,1]} allowDataOverflow={z.isZoomed} />
 *     {z.refArea && <ReferenceArea ... />}
 *   </ComposedChart>
 *   <ChartActions isZoomed={z.isZoomed} onResetZoom={z.resetZoom} ... />
 */
/**
 * Extract data coordinates from a recharts mouse event.
 * ComposedChart doesn't always provide xValue/yValue, so we
 * also look at activePayload and activeCoordinate.
 */
function getRechartsCoords(e) {
  const hasPixel = e?.chartX != null && e?.chartY != null;
  const pixelX = hasPixel ? e.chartX : null;
  const pixelY = hasPixel ? e.chartY : null;

  if (e?.xValue != null && e?.yValue != null) return { x: e.xValue, y: e.yValue };
  // For ComposedChart with Scatter, use activePayload
  if (e?.activePayload?.[0]?.payload) {
    const p = e.activePayload[0].payload;
    if (p.x != null && p.y != null) return { x: p.x, y: p.y, pixelX, pixelY };
  }
  // Fallback to chart pixel coords (stored for proportional zoom)
  if (hasPixel) {
    return { x: e.chartX, y: e.chartY, pixelX, pixelY, isPixel: true };
  }
  return null;
}

export function useChartInteraction({ onSelectPoint, xDomain: xRange, yDomain: yRange, chartLayout } = {}) {
  const [domain, setDomain] = useState(null);
  const [refArea, setRefArea] = useState(null);
  const [isPanning, setIsPanning] = useState(false);
  const dragging = useRef(false);
  const suppressClickRef = useRef(false);
  const chartRef = useRef(null); // stores chart dimensions for pixel→data conversion

  const normalizeBounds = useCallback((dataBounds) => {
    if (dataBounds && Number.isFinite(dataBounds.xMin) && Number.isFinite(dataBounds.xMax) && Number.isFinite(dataBounds.yMin) && Number.isFinite(dataBounds.yMax)) {
      return {
        xMin: dataBounds.xMin,
        xMax: dataBounds.xMax,
        yMin: dataBounds.yMin,
        yMax: dataBounds.yMax,
      };
    }
    if (
      Array.isArray(xRange) && xRange.length === 2 && Number.isFinite(xRange[0]) && Number.isFinite(xRange[1]) &&
      Array.isArray(yRange) && yRange.length === 2 && Number.isFinite(yRange[0]) && Number.isFinite(yRange[1])
    ) {
      return { xMin: xRange[0], xMax: xRange[1], yMin: yRange[0], yMax: yRange[1] };
    }
    return null;
  }, [xRange, yRange]);

  const clampDomainToBounds = useCallback((next, bounds) => {
    if (!next || !bounds) return next;
    const bx = bounds.xMax - bounds.xMin;
    const by = bounds.yMax - bounds.yMin;
    if (bx <= 0 || by <= 0) return next;

    const minSpanX = Math.max(1e-9, bx * 0.01);
    const minSpanY = Math.max(1e-9, by * 0.01);
    const maxSpanX = bx;
    const maxSpanY = by;

    let x0 = Math.min(next.x[0], next.x[1]);
    let x1 = Math.max(next.x[0], next.x[1]);
    let y0 = Math.min(next.y[0], next.y[1]);
    let y1 = Math.max(next.y[0], next.y[1]);

    let spanX = Math.max(minSpanX, Math.min(maxSpanX, x1 - x0));
    let spanY = Math.max(minSpanY, Math.min(maxSpanY, y1 - y0));
    let cx = (x0 + x1) / 2;
    let cy = (y0 + y1) / 2;
    x0 = cx - spanX / 2;
    x1 = cx + spanX / 2;
    y0 = cy - spanY / 2;
    y1 = cy + spanY / 2;

    if (x0 < bounds.xMin) { x1 += bounds.xMin - x0; x0 = bounds.xMin; }
    if (x1 > bounds.xMax) { x0 -= x1 - bounds.xMax; x1 = bounds.xMax; }
    if (y0 < bounds.yMin) { y1 += bounds.yMin - y0; y0 = bounds.yMin; }
    if (y1 > bounds.yMax) { y0 -= y1 - bounds.yMax; y1 = bounds.yMax; }

    // Final hard clamp for precision drift.
    x0 = Math.max(bounds.xMin, x0);
    x1 = Math.min(bounds.xMax, x1);
    y0 = Math.max(bounds.yMin, y0);
    y1 = Math.min(bounds.yMax, y1);

    return { x: [x0, x1], y: [y0, y1] };
  }, []);

  const onMouseDown = useCallback((e) => {
    const coords = getRechartsCoords(e);
    if (!coords) return;
    suppressClickRef.current = false;
    const canPan = domain && !e?.shiftKey;
    dragging.current = {
      mode: canPan ? 'pan' : 'zoom',
      start: coords,
      originDomain: domain ? { x: [...domain.x], y: [...domain.y] } : null,
    };
    if (canPan) setIsPanning(true);
  }, [domain]);

  const onMouseMove = useCallback((e) => {
    if (!dragging.current) return;
    const coords = getRechartsCoords(e);
    if (!coords) return;
    const start = dragging.current.start;

    if (dragging.current.mode === 'pan' && dragging.current.originDomain) {
      const sx = Number.isFinite(start.pixelX) ? start.pixelX : start.x;
      const sy = Number.isFinite(start.pixelY) ? start.pixelY : start.y;
      const cx = Number.isFinite(coords.pixelX) ? coords.pixelX : coords.x;
      const cy = Number.isFinite(coords.pixelY) ? coords.pixelY : coords.y;
      if (!Number.isFinite(sx) || !Number.isFinite(sy) || !Number.isFinite(cx) || !Number.isFinite(cy)) return;

      const dxPx = cx - sx;
      const dyPx = cy - sy;
      if (Math.abs(dxPx) > 2 || Math.abs(dyPx) > 2) suppressClickRef.current = true;

      const spanX = Math.max(1e-9, dragging.current.originDomain.x[1] - dragging.current.originDomain.x[0]);
      const spanY = Math.max(1e-9, dragging.current.originDomain.y[1] - dragging.current.originDomain.y[0]);
      const plotW = Math.max(120, Number(chartLayout?.width || 0) - 44);
      const plotH = Math.max(80, Number(chartLayout?.height || 0) - 34);
      const dxData = (dxPx / plotW) * spanX;
      const dyData = (dyPx / plotH) * spanY;

      const next = {
        x: [dragging.current.originDomain.x[0] - dxData, dragging.current.originDomain.x[1] - dxData],
        y: [dragging.current.originDomain.y[0] + dyData, dragging.current.originDomain.y[1] + dyData],
      };
      setDomain(clampDomainToBounds(next, normalizeBounds()));
      return;
    }

    // Only begin visual drag after sufficient pixel movement
    if (!refArea) {
      const dx = Math.abs(coords.x - start.x);
      const dy = Math.abs(coords.y - start.y);
      const threshold = start.isPixel ? 8 : 0.001;
      if (dx < threshold && dy < threshold) return;
      setRefArea({ x1: start.x, y1: start.y, x2: coords.x, y2: coords.y, isPixel: start.isPixel });
    } else {
      setRefArea(prev => prev ? { ...prev, x2: coords.x, y2: coords.y } : null);
    }
  }, [refArea, clampDomainToBounds, normalizeBounds, chartLayout]);

  const onMouseUp = useCallback(() => {
    if (dragging.current?.mode === 'pan') {
      dragging.current = false;
      setIsPanning(false);
      return;
    }
    dragging.current = false;
    setIsPanning(false);
    if (!refArea) return;
    const { x1, y1, x2, y2, isPixel } = refArea;

    if (isPixel) {
      // Pixel-based: skip zoom if too small, otherwise ignore (can't map to data without axis info)
      setRefArea(null);
      return;
    }

    const dx = Math.abs(x2 - x1);
    const dy = Math.abs(y2 - y1);
    if (dx < 0.001 && dy < 0.001) {
      setRefArea(null);
      return;
    }
    const bounds = normalizeBounds();
    const next = {
      x: [Math.min(x1, x2), Math.max(x1, x2)],
      y: [Math.min(y1, y2), Math.max(y1, y2)],
    };
    setDomain(clampDomainToBounds(next, bounds));
    setRefArea(null);
  }, [refArea, clampDomainToBounds, normalizeBounds]);

  const resetZoom = useCallback(() => {
    setDomain(null);
    setRefArea(null);
  }, []);

  /** Zoom in/out by a factor (e.g. 0.5 = zoom in 50%, 2.0 = zoom out 2x) around the center of current view. */
  const zoomBy = useCallback((factor, dataBounds) => {
    const bounds = normalizeBounds(dataBounds);
    const cur = domain || (bounds ? { x: [bounds.xMin, bounds.xMax], y: [bounds.yMin, bounds.yMax] } : null);
    if (!cur) return;
    const cx = (cur.x[0] + cur.x[1]) / 2;
    const cy = (cur.y[0] + cur.y[1]) / 2;
    const hw = (cur.x[1] - cur.x[0]) / 2 * factor;
    const hy = (cur.y[1] - cur.y[0]) / 2 * factor;
    setDomain(clampDomainToBounds({ x: [cx - hw, cx + hw], y: [cy - hy, cy + hy] }, bounds));
  }, [domain, clampDomainToBounds, normalizeBounds]);

  const consumeClickGuard = useCallback(() => {
    const blocked = suppressClickRef.current;
    suppressClickRef.current = false;
    return blocked;
  }, []);

  const handlePointClick = useCallback((pointData) => {
    if (onSelectPoint) onSelectPoint(pointData);
  }, [onSelectPoint]);

  return {
    domain,
    refArea,
    isZoomed: domain != null,
    isPanning,
    onMouseDown,
    onMouseMove,
    onMouseUp,
    resetZoom,
    zoomBy,
    consumeClickGuard,
    handlePointClick,
  };
}

/**
 * useSvgDragZoom — click-drag-to-zoom for raw SVG charts.
 *
 * Converts pixel coordinates to data coordinates using provided scale functions.
 *
 * @param {object} opts
 * @param {function} opts.xPixelToData - converts SVG x pixel → data x value
 * @param {function} opts.yPixelToData - converts SVG y pixel → data y value
 * @param {function} opts.onSelectPoint - callback when a point is clicked
 */
export function useSvgDragZoom({ xPixelToData, yPixelToData, onSelectPoint } = {}) {
  const [domain, setDomain] = useState(null);
  const [refArea, setRefArea] = useState(null);
  const dragging = useRef(false);
  const startPx = useRef(null);

  const getSvgCoords = useCallback((e, svgRef) => {
    if (!svgRef?.current) return null;
    const rect = svgRef.current.getBoundingClientRect();
    return { px: e.clientX - rect.left, py: e.clientY - rect.top };
  }, []);

  const onMouseDown = useCallback((e, svgRef) => {
    const coords = getSvgCoords(e, svgRef);
    if (!coords) return;
    dragging.current = true;
    startPx.current = coords;
    setRefArea({ px1: coords.px, py1: coords.py, px2: coords.px, py2: coords.py });
  }, [getSvgCoords]);

  const onMouseMove = useCallback((e, svgRef) => {
    if (!dragging.current) return;
    const coords = getSvgCoords(e, svgRef);
    if (!coords) return;
    setRefArea(prev => prev ? { ...prev, px2: coords.px, py2: coords.py } : null);
  }, [getSvgCoords]);

  const onMouseUp = useCallback(() => {
    dragging.current = false;
    if (!refArea || !xPixelToData || !yPixelToData) { setRefArea(null); return; }
    const { px1, py1, px2, py2 } = refArea;
    const dx = Math.abs(px2 - px1);
    const dy = Math.abs(py2 - py1);
    if (dx < 5 && dy < 5) { setRefArea(null); return; }

    const x1 = xPixelToData(Math.min(px1, px2));
    const x2 = xPixelToData(Math.max(px1, px2));
    const y1 = yPixelToData(Math.max(py1, py2)); // SVG y is inverted
    const y2 = yPixelToData(Math.min(py1, py2));

    setDomain({ x: [x1, x2], y: [y1, y2] });
    setRefArea(null);
  }, [refArea, xPixelToData, yPixelToData]);

  const resetZoom = useCallback(() => {
    setDomain(null);
    setRefArea(null);
  }, []);

  const handlePointClick = useCallback((pointData) => {
    if (onSelectPoint) onSelectPoint(pointData);
  }, [onSelectPoint]);

  return {
    domain,
    refArea,
    isZoomed: domain != null,
    onMouseDown,
    onMouseMove,
    onMouseUp,
    resetZoom,
    handlePointClick,
    getSvgCoords,
  };
}
