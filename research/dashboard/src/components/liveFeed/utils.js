import {
  EVENT_TYPE_ALIASES,
  GENERATION_EVENT_TYPES,
  RENDERABLE_EVENT_TYPES,
  TERMINAL_EVENT_TYPES,
} from './constants';

export function normalizeLiveFeedEvent(rawEvent) {
  if (!rawEvent || typeof rawEvent !== 'object') return null;
  const rawType = String(rawEvent.type || rawEvent.event_type || rawEvent.event || '').trim();
  if (!rawType) return null;
  const normalizedType = EVENT_TYPE_ALIASES[rawType] || rawType;
  if (!RENDERABLE_EVENT_TYPES.has(normalizedType)) return null;
  return {
    ...rawEvent,
    type: normalizedType,
    ts: rawEvent.ts || Date.now(),
  };
}

export function annotateGenerationHistory(events) {
  const generationState = new Map();
  return events.map((event) => {
    if (!GENERATION_EVENT_TYPES.has(event?.type)) {
      return event;
    }

    const generation = Number(event?.generation);
    if (!Number.isFinite(generation)) {
      return event;
    }

    const runKey = `${event.type}:${event.experiment_id || 'unknown'}:${event.total_generations || 'unknown'}`;
    const previous = generationState.get(runKey);

    const annotated = { ...event };
    if (previous == null) {
      annotated._runStart = true;
      if (generation > 1) {
        annotated._missingPrefixFrom = 1;
        annotated._missingPrefixTo = generation - 1;
      }
    } else if (generation - previous > 1) {
      annotated._missingGapFrom = previous + 1;
      annotated._missingGapTo = generation - 1;
    }

    generationState.set(runKey, generation);
    return annotated;
  });
}

export function reconcileTerminalEvents(events) {
  const latestTerminalIndex = new Map();
  events.forEach((event, index) => {
    const experimentId = event?.experiment_id || null;
    if (!experimentId || !TERMINAL_EVENT_TYPES.has(event?.type)) return;
    latestTerminalIndex.set(experimentId, index);
  });

  return events.filter((event, index) => {
    const experimentId = event?.experiment_id || null;
    if (!experimentId || !TERMINAL_EVENT_TYPES.has(event?.type)) return true;
    return latestTerminalIndex.get(experimentId) === index;
  });
}

export function splitCurveIntoSegments(curve) {
  const segments = [];
  let current = [];
  for (const point of curve || []) {
    if (current.length > 0 && Number(point.step) < Number(current[current.length - 1].step)) {
      segments.push(current);
      current = [];
    }
    current.push(point);
  }
  if (current.length > 0) segments.push(current);
  return segments;
}

export function buildCurveSnapshot(experimentId, curve, overrides = {}) {
  if (!experimentId || !Array.isArray(curve) || curve.length < 2) return null;
  return {
    experimentId,
    curve: curve.map((point) => ({ ...point })),
    statusText: overrides.statusText || '',
    statusTone: overrides.statusTone || 'info',
    label: overrides.label || `Run ${String(experimentId).slice(0, 8)}`,
    updatedTs: Date.now(),
  };
}

export function describeCurveEvent(event) {
  const type = String(event?.type || '');
  const shortId = (event?.experiment_id || '').slice(0, 8);
  if (type === 'failed') {
    return {
      statusText: event?.error ? `Failed: ${event.error}` : 'Experiment failed.',
      statusTone: 'warn',
      label: shortId ? `Failed ${shortId}` : 'Failed run',
    };
  }
  if (type === 'validate_complete') {
    return {
      statusText: 'Validation completed. Analysis recorded.',
      statusTone: 'success',
      label: shortId ? `Validation ${shortId}` : 'Validation run',
    };
  }
  if (type === 'invest_complete' || type === 'scaleup_complete' || type === 'nov_complete' || type === 'evo_complete' || type === 'complete') {
    return {
      statusText: 'Run completed.',
      statusTone: 'success',
      label: shortId ? `Completed ${shortId}` : 'Completed run',
    };
  }
  return null;
}

