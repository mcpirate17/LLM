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
  const rawTs = Number(rawEvent.ts ?? rawEvent.timestamp ?? rawEvent.created_at);
  const ts = Number.isFinite(rawTs)
    ? (rawTs > 1e12 ? rawTs : rawTs * 1000)
    : Date.now();
  return {
    ...rawEvent,
    type: normalizedType,
    ts,
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
  let currentKey = null;
  for (const point of curve || []) {
    const pointKey = [
      point?.source_result_id || '',
      point?.candidate_index ?? '',
      point?.training_program_index ?? '',
      point?.training_program_label || '',
      point?.training_seed ?? '',
      point?.run_kind || point?.phase || '',
    ].join('|');
    const hasExplicitSegment = Boolean(
      point?.source_result_id
      || point?.candidate_index != null
      || point?.training_program_index != null
      || point?.training_program_label
      || point?.training_seed != null
      || point?.run_kind
    );
    const shouldSplit = current.length > 0 && (
      (hasExplicitSegment && currentKey !== null && pointKey !== currentKey)
      || Number(point.step) < Number(current[current.length - 1].step)
    );
    if (shouldSplit) {
      segments.push(current);
      current = [];
    }
    if (current.length === 0) {
      currentKey = hasExplicitSegment ? pointKey : null;
    }
    current.push(point);
  }
  if (current.length > 0) segments.push(current);
  return segments;
}

export function curveSegmentLabel(segment, fallbackPrefix = 'run', index = 0) {
  const first = Array.isArray(segment) ? segment[0] : null;
  if (!first) return `${fallbackPrefix} ${index + 1}`;
  const candidate = first.candidate_index != null && first.total_candidates != null
    ? `c${first.candidate_index}/${first.total_candidates}`
    : first.candidate_index != null
      ? `c${first.candidate_index}`
      : '';
  const program = first.training_program_label
    || (first.training_program_index != null && first.total_training_programs != null
      ? `p${first.training_program_index}/${first.total_training_programs}`
      : first.training_program_index != null
        ? `p${first.training_program_index}`
        : '');
  const label = [candidate, program].filter(Boolean).join(' ');
  return label || `${fallbackPrefix} ${index + 1}`;
}

export function buildCurveSnapshot(experimentId, curve, overrides = {}) {
  if (!experimentId || !Array.isArray(curve) || curve.length < 2) return null;
  const phase = String(overrides.phase || curve[curve.length - 1]?.phase || '').toLowerCase();
  const segmentLabelPrefix = overrides.segmentLabelPrefix
    || (phase === 'investigation' ? 'program' : phase === 'validation' ? 'seed' : 'run');
  return {
    experimentId,
    curve: curve.map((point) => ({ ...point })),
    statusText: overrides.statusText || '',
    statusTone: overrides.statusTone || 'info',
    label: overrides.label || `Run ${String(experimentId).slice(0, 8)}`,
    segmentLabelPrefix,
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
