import { useEffect, useRef } from 'react';

const MAX_SAMPLES = 400;

function profilingEnabled() {
  return (
    process.env.NODE_ENV !== 'production'
    && typeof window !== 'undefined'
    && typeof performance !== 'undefined'
    && Boolean(window.__ariaRenderPerfCollect || window.__ariaRenderPerfDebug)
  );
}

export default function useRenderPerf(componentName, options = {}) {
  const { thresholdMs = 10 } = options;
  const enabled = profilingEnabled();
  const renderStartRef = useRef(0);

  if (enabled) {
    renderStartRef.current = performance.now();
  }

  useEffect(() => {
    if (!enabled) return;

    const duration = performance.now() - renderStartRef.current;
    const sample = {
      component: componentName,
      durationMs: Number(duration.toFixed(2)),
      timestamp: Date.now(),
    };

    const existing = Array.isArray(window.__ariaRenderPerf)
      ? window.__ariaRenderPerf
      : [];

    existing.push(sample);
    if (existing.length > MAX_SAMPLES) {
      existing.splice(0, existing.length - MAX_SAMPLES);
    }
    window.__ariaRenderPerf = existing;

    if (window.__ariaRenderPerfDebug && duration >= thresholdMs) {
      console.debug(`[render-perf] ${componentName}: ${duration.toFixed(2)}ms`);
    }
  }, [componentName, enabled, thresholdMs]);
}
