import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useEventBus } from '../hooks/useEventBus';
import apiService, { apiCall } from '../services/apiService';

const LIVE_LOSS_CURVE_MAX_POINTS = 20000;
const LIVE_FEED_MAX_EVENTS = 100;
const LIVE_FEED_MAX_GRAPHS = 5;

const RESULT_COLORS = {
  'S1 PASS': 'var(--accent-green)',
  'S0': 'var(--accent-blue)',
  'FAIL': 'var(--accent-red)',
  'invalid': 'var(--accent-red)',
  'compile_error': 'var(--accent-orange)',
};

const EVENT_TYPE_ALIASES = {
  program_evaluated: 'program',
  experiment_started: 'start',
  experiment_completed: 'complete',
  experiment_failed: 'failed',
  experiment_stopping: 'stopping',
  evolution_started: 'evo_start',
  evolution_generation: 'evo_gen',
  evolution_completed: 'evo_complete',
  novelty_started: 'nov_start',
  novelty_generation: 'nov_gen',
  novelty_completed: 'nov_complete',
  scale_up_started: 'scaleup_start',
  scale_up_progress: 'scaleup_progress',
  scale_up_completed: 'scaleup_complete',
  auto_scale_up_queued: 'auto_scaleup',
  investigation_started: 'invest_start',
  investigation_progress: 'invest_progress',
  investigation_completed: 'invest_complete',
  validation_started: 'validate_start',
  validation_progress: 'validate_progress',
  validation_phase: 'validate_phase',
  validation_completed: 'validate_complete',
  breakthrough_detected: 'breakthrough',
  auto_investigate_queued: 'auto_investigate',
  auto_validate_queued: 'auto_validate',
  auto_report_generated: 'auto_report',
  aria_recommendation: 'recommendation',
  hypothesis_recorded: 'hyp_recorded',
  hypothesis_resolved: 'hyp_resolved',
  decision_recorded: 'decision',
  knowledge_extracted: 'knowledge',
  campaign_created: 'campaign_created',
  campaign_completed: 'campaign_completed',
  aria_cycle_phase: 'aria_phase',
  continuous_limit_reached: 'limit_reached',
  learning_event: 'learning',
  training_step: 'training_step',
  log_message: 'log',
};

const RENDERABLE_EVENT_TYPES = new Set([
  'program',
  'start',
  'complete',
  'failed',
  'stopping',
  'evo_start',
  'evo_gen',
  'evo_complete',
  'nov_start',
  'nov_gen',
  'nov_complete',
  'scaleup_start',
  'scaleup_progress',
  'scaleup_complete',
  'auto_scaleup',
  'mode_selected',
  'invest_start',
  'invest_progress',
  'invest_complete',
  'validate_start',
  'validate_progress',
  'validate_phase',
  'validate_complete',
  'breakthrough',
  'auto_investigate',
  'auto_validate',
  'auto_report',
  'recommendation',
  'hyp_recorded',
  'hyp_resolved',
  'decision',
  'knowledge',
  'campaign_created',
  'campaign_completed',
  'aria_phase',
  'limit_reached',
  'learning',
  'log',
]);

const GENERATION_EVENT_TYPES = new Set(['evo_gen', 'nov_gen']);
const RUN_START_EVENT_TYPES = new Set(['start', 'evo_start', 'nov_start', 'invest_start', 'validate_start', 'scaleup_start']);
const CONTEXT_SWITCH_EVENT_TYPES = new Set([
  ...RUN_START_EVENT_TYPES,
  'invest_progress',
  'validate_progress',
  'scaleup_progress',
  'evo_gen',
  'nov_gen',
]);

function normalizeLiveFeedEvent(rawEvent) {
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

function annotateGenerationHistory(events) {
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

function splitCurveIntoSegments(curve) {
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

function buildCurveSnapshot(experimentId, curve, overrides = {}) {
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

function describeCurveEvent(event) {
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

function MiniNoveltyChart({ points, label = '', width = 600 }) {
  if (!Array.isArray(points) || points.length === 0) return null;
  const W = width;
  const H = 113;
  const pad = { l: 8, r: 8, t: 8, b: 14 };
  const maxGeneration = Math.max(...points.map((point) => Number(point.generation) || 0), 1);
  const metricValues = points.flatMap((point) => [Number(point.best_fitness) || 0, Number(point.best_novelty) || 0]);
  const minValue = Math.min(...metricValues);
  const maxValue = Math.max(...metricValues);
  const rangeValue = maxValue - minValue || 1;
  const xScale = (generation) => pad.l + ((Number(generation) || 0) / Math.max(maxGeneration, 1)) * (W - pad.l - pad.r);
  const yScale = (value) => H - pad.b - (((Number(value) || 0) - minValue) / rangeValue) * (H - pad.t - pad.b);
  const buildPath = (key) => points.map((point, idx) => `${idx === 0 ? 'M' : 'L'} ${xScale(point.generation).toFixed(1)} ${yScale(point[key]).toFixed(1)}`).join(' ');
  const latest = points[points.length - 1];

  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 8,
      padding: '4px 8px', borderRadius: 6,
      background: 'rgba(63,185,80,0.08)', border: '1px solid rgba(63,185,80,0.2)',
      marginBottom: 4,
    }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
        <rect width={W} height={H} rx={4} fill="rgba(0,0,0,0.3)" />
        <path d={buildPath('best_fitness')} fill="none" stroke="var(--accent-green)" strokeWidth={1.5} />
        <path d={buildPath('best_novelty')} fill="none" stroke="var(--accent-yellow)" strokeWidth={1.5} />
        {points.map((point) => (
          <line
            key={point.generation}
            x1={xScale(point.generation)}
            y1={H - pad.b}
            x2={xScale(point.generation)}
            y2={H - pad.b + 4}
            stroke="rgba(255,255,255,0.25)"
          />
        ))}
        <text x={pad.l} y={pad.t + 10} fill="var(--text-muted)" fontSize="9" fontFamily="monospace">best_fit</text>
        <text x={pad.l + 46} y={pad.t + 10} fill="var(--accent-yellow)" fontSize="9" fontFamily="monospace">best_novelty</text>
      </svg>
      <div style={{ fontSize: 10, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
        {label && <div style={{ color: 'var(--text-muted)', marginBottom: 4, maxWidth: 220 }}>{label}</div>}
        <div style={{ color: 'var(--accent-green)', fontWeight: 700, fontSize: 12, fontFamily: 'monospace' }}>
          fit {Number(latest.best_fitness || 0).toFixed(3)}
        </div>
        <div style={{ color: 'var(--accent-yellow)', fontFamily: 'monospace' }}>
          nov {Number(latest.best_novelty || 0).toFixed(3)}
        </div>
        <div>gen {latest.generation}/{latest.total_generations}</div>
        <div>archive {latest.archive_size}</div>
      </div>
    </div>
  );
}

// Mini inline SVG chart for live training loss
function MiniLossChart({ curve, statusText = '', statusTone = 'info', label = '', width = 600 }) {
  if (!curve || curve.length < 2) return null;
  const W = width, H = 113;
  const pad = { l: 4, r: 4, t: 4, b: 4 };
  const segments = splitCurveIntoSegments(curve);

  const losses = curve.map(p => p.loss);
  const minL = Math.min(...losses);
  const maxL = Math.max(...losses);
  const rangeL = maxL - minL || 1;

  const xScale = i => pad.l + (i / Math.max(curve.length - 1, 1)) * (W - pad.l - pad.r);
  const yScale = v => H - pad.b - ((v - minL) / rangeL) * (H - pad.t - pad.b);

  let pointOffset = 0;
  const segmentPaths = segments.map((segment) => {
    const startOffset = pointOffset;
    pointOffset += segment.length;
    const pathD = segment
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xScale(startOffset + i).toFixed(1)} ${yScale(p.loss).toFixed(1)}`)
      .join(' ');
    return { pathD, first: segment[0], last: segment[segment.length - 1], startOffset };
  });
  const currentLoss = losses[losses.length - 1];
  const currentStep = curve[curve.length - 1].step;
  const totalSteps = curve[curve.length - 1].total_steps;
  const phase = curve[curve.length - 1].phase || '';
  const statusColor = statusTone === 'warn'
    ? 'var(--accent-yellow)'
    : statusTone === 'success'
      ? 'var(--accent-green)'
      : 'var(--text-secondary)';

  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 8,
      padding: '4px 8px', borderRadius: 6,
      background: 'rgba(63,185,80,0.08)', border: '1px solid rgba(63,185,80,0.2)',
      marginBottom: 4,
    }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
        <rect width={W} height={H} rx={4} fill="rgba(0,0,0,0.3)" />
        {segmentPaths.map((segment, idx) => (
          <g key={`${segment.startOffset}-${idx}`}>
            <path d={segment.pathD} fill="none" stroke="var(--accent-green)" strokeWidth={1.5} />
            {idx < segmentPaths.length - 1 && (
              <line
                x1={xScale(segment.startOffset + Math.max(0, segments[idx].length - 1))}
                y1={pad.t}
                x2={xScale(segment.startOffset + Math.max(0, segments[idx].length - 1))}
                y2={H - pad.b}
                stroke="rgba(255,255,255,0.14)"
                strokeDasharray="3 3"
              />
            )}
            <text
              x={xScale(segment.startOffset) + 6}
              y={pad.t + 12}
              fill="var(--text-muted)"
              fontSize="9"
              fontFamily="monospace"
            >
              {`seed ${idx + 1}`}
            </text>
          </g>
        ))}
      </svg>
      <div style={{ fontSize: 10, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
        {label && (
          <div style={{ color: 'var(--text-muted)', marginBottom: 4, maxWidth: 220 }}>
            {label}
          </div>
        )}
        <div style={{ color: 'var(--accent-green)', fontWeight: 700, fontSize: 12, fontFamily: 'monospace' }}>
          {currentLoss < 0.0001 && currentLoss !== 0 ? currentLoss.toExponential(2) : currentLoss.toFixed(4)}
        </div>
        <div>step {currentStep}/{totalSteps}</div>
        {phase && <div style={{ textTransform: 'capitalize', color: 'var(--text-muted)' }}>{phase}</div>}
        {statusText && <div style={{ color: statusColor, maxWidth: 220 }}>{statusText}</div>}
      </div>
    </div>
  );
}

function LiveFeed({ apiBase, experimentId = null, progress = null }) {
  const [events, setEvents] = useState([]);
  const [lossCurve, setLossCurve] = useState([]);
  const [curveHistory, setCurveHistory] = useState([]);
  const lossCurveExpRef = useRef(null);
  const lossCurveRef = useRef([]);
  const activeExperimentRef = useRef(experimentId || null);
  const feedRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [showControls, setShowControls] = useState(false);
  const [nowTs, setNowTs] = useState(Date.now());
  const prevConnectedRef = useRef(null);
  const displayEvents = useMemo(() => annotateGenerationHistory(events), [events]);

  useEffect(() => {
    const interval = setInterval(() => setNowTs(Date.now()), 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    lossCurveRef.current = lossCurve;
  }, [lossCurve]);

  const archiveCurveSnapshot = useCallback((experimentIdOverride = null, overrides = {}) => {
    const experimentId = experimentIdOverride || lossCurveExpRef.current || activeExperimentRef.current || null;
    const snapshot = buildCurveSnapshot(experimentId, lossCurveRef.current, overrides);
    if (!snapshot) return;
    setCurveHistory((prev) => {
      const next = [snapshot, ...prev.filter((item) => item.experimentId !== snapshot.experimentId)];
      return next.slice(0, LIVE_FEED_MAX_GRAPHS);
    });
  }, []);

  // Helper to add an event with a mapped type
  const addEvent = useCallback((type) => (data) => {
    const eventExpId = data?.experiment_id || null;
    const currentExpId = activeExperimentRef.current || null;

    // If this is a new run starting, clear stale feed and switch context.
    if (RUN_START_EVENT_TYPES.has(type) && eventExpId && eventExpId !== currentExpId) {
      archiveCurveSnapshot(currentExpId, {
        statusText: 'Previous run archived.',
        statusTone: 'info',
        label: currentExpId ? `Run ${currentExpId.slice(0, 8)}` : 'Previous run',
      });
      activeExperimentRef.current = eventExpId;
      lossCurveExpRef.current = eventExpId;
      setEvents([]);
      setLossCurve([]);
    }

    // If we missed the explicit start event (SSE reconnect, queue overflow),
    // let later phase-specific events switch context as well.
    if (currentExpId && eventExpId && eventExpId !== currentExpId) {
      if (CONTEXT_SWITCH_EVENT_TYPES.has(type)) {
        activeExperimentRef.current = eventExpId;
        lossCurveExpRef.current = eventExpId;
        setEvents([]);
        setLossCurve([]);
      } else {
        return;
      }
    }

    const normalized = normalizeLiveFeedEvent({ type, ...data, ts: Date.now() });
    if (!normalized) return;
    const curveEventMeta = describeCurveEvent(normalized);
    if (curveEventMeta && (!eventExpId || eventExpId === (activeExperimentRef.current || eventExpId))) {
      archiveCurveSnapshot(eventExpId, curveEventMeta);
    }
    setEvents(prev => [...prev.slice(-(LIVE_FEED_MAX_EVENTS - 1)), normalized]);
  }, [archiveCurveSnapshot]);

  // Handle training_step events for the mini loss chart
  const handleTrainingStep = useCallback((data) => {
    const expId = data.experiment_id || '';
    const currentExpId = activeExperimentRef.current || null;
    if (currentExpId && expId && expId !== currentExpId) return;
    if (!currentExpId && expId) activeExperimentRef.current = expId;
    setLossCurve(prev => {
      // Clear buffer when experiment changes
      if (lossCurveExpRef.current !== expId) {
        lossCurveExpRef.current = expId;
        return [{ step: data.step, loss: data.loss, total_steps: data.total_steps, phase: data.phase, received_ts: Date.now() }];
      }
      const next = [...prev, { step: data.step, loss: data.loss, total_steps: data.total_steps, phase: data.phase, received_ts: Date.now() }];
      return next.length > LIVE_LOSS_CURVE_MAX_POINTS
        ? next.slice(-LIVE_LOSS_CURVE_MAX_POINTS)
        : next;
    });
  }, []);

  // Subscribe to all SSE events via shared EventBus
  const { connected } = useEventBus('program_evaluated', addEvent('program'));
  useEventBus('experiment_started', addEvent('start'));
  useEventBus('experiment_completed', addEvent('complete'));
  useEventBus('experiment_failed', addEvent('failed'));
  useEventBus('experiment_stopping', addEvent('stopping'));
  useEventBus('evolution_started', addEvent('evo_start'));
  useEventBus('evolution_generation', addEvent('evo_gen'));
  useEventBus('evolution_completed', addEvent('evo_complete'));
  useEventBus('novelty_started', addEvent('nov_start'));
  useEventBus('novelty_generation', addEvent('nov_gen'));
  useEventBus('novelty_completed', addEvent('nov_complete'));
  useEventBus('scale_up_started', addEvent('scaleup_start'));
  useEventBus('scale_up_progress', addEvent('scaleup_progress'));
  useEventBus('scale_up_completed', addEvent('scaleup_complete'));
  useEventBus('auto_scale_up_queued', addEvent('auto_scaleup'));
  useEventBus('mode_selected', addEvent('mode_selected'));
  useEventBus('investigation_started', addEvent('invest_start'));
  useEventBus('investigation_progress', addEvent('invest_progress'));
  useEventBus('investigation_completed', addEvent('invest_complete'));
  useEventBus('validation_started', addEvent('validate_start'));
  useEventBus('validation_progress', addEvent('validate_progress'));
  useEventBus('validation_phase', addEvent('validate_phase'));
  useEventBus('validation_completed', addEvent('validate_complete'));
  useEventBus('breakthrough_detected', addEvent('breakthrough'));
  useEventBus('auto_investigate_queued', addEvent('auto_investigate'));
  useEventBus('auto_validate_queued', addEvent('auto_validate'));
  useEventBus('auto_report_generated', addEvent('auto_report'));
  useEventBus('aria_recommendation', addEvent('recommendation'));
  useEventBus('hypothesis_recorded', addEvent('hyp_recorded'));
  useEventBus('hypothesis_resolved', addEvent('hyp_resolved'));
  useEventBus('decision_recorded', addEvent('decision'));
  useEventBus('knowledge_extracted', addEvent('knowledge'));
  useEventBus('campaign_created', addEvent('campaign_created'));
  useEventBus('campaign_completed', addEvent('campaign_completed'));
  useEventBus('aria_cycle_phase', addEvent('aria_phase'));
  useEventBus('continuous_limit_reached', addEvent('limit_reached'));
  useEventBus('learning_event', addEvent('learning'));
  useEventBus('log_message', addEvent('log'));
  useEventBus('training_step', handleTrainingStep);

  // Fetch loss curve on mount (regardless of experimentId)
  useEffect(() => {
    apiCall(`/api/live-loss-curve`)
      .then(r => r.json())
      .then(curve => {
        if (Array.isArray(curve) && curve.length >= 2) {
          const curveExpId = curve[0]?.experiment_id || '';
          if (!experimentId || curveExpId === experimentId) {
            lossCurveExpRef.current = curveExpId;
            setLossCurve(curve.map(p => ({
              step: p.step, loss: p.loss,
              total_steps: p.total_steps, phase: p.phase, received_ts: Date.now(),
            })));
          }
        }
      })
      .catch(() => {});
  }, [apiBase]);

  // Load history from REST when experimentId changes
  useEffect(() => {
    activeExperimentRef.current = experimentId || null;
    setEvents([]);
    setLossCurve([]);
    setCurveHistory([]);
    lossCurveExpRef.current = null;
    if (!experimentId) return;

    apiService.getLiveFeed(experimentId, 100)
      .then((history) => {
        if (!Array.isArray(history) || history.length === 0) return;
        const normalizedHistory = history
          .map((event) => normalizeLiveFeedEvent(event))
          .filter(Boolean);
        if (normalizedHistory.length === 0) return;
        setEvents((prev) => [...normalizedHistory, ...prev].slice(-LIVE_FEED_MAX_EVENTS));
      })
      .catch(() => {});

    // Restore loss curve from server buffer
    apiCall(`/api/live-loss-curve`)
      .then(r => r.json())
      .then(curve => {
        if (Array.isArray(curve) && curve.length >= 2) {
          const curveExpId = curve[0]?.experiment_id || '';
          if (curveExpId === experimentId) {
            lossCurveExpRef.current = curveExpId;
            setLossCurve(curve.map(p => ({
              step: p.step, loss: p.loss,
              total_steps: p.total_steps, phase: p.phase, received_ts: Date.now(),
            })));
          }
        }
      })
      .catch(() => {});
  }, [apiBase, experimentId]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [autoScroll, events]);

  // On reconnect, re-fetch history to heal gaps.
  useEffect(() => {
    const prev = prevConnectedRef.current;
    if (prev === null) {
      prevConnectedRef.current = connected;
      return;
    }
    if (!prev && connected && experimentId) {
      console.log(`LiveFeed: Reconnected. Re-fetching history for exp ${experimentId} to heal gaps.`);
      apiService.getLiveFeed(experimentId, 150)
        .then((history) => {
          if (!Array.isArray(history) || history.length === 0) return;
          const normalizedHistory = history
            .map((event) => normalizeLiveFeedEvent(event))
            .filter(Boolean);
          if (normalizedHistory.length === 0) return;
          
          setEvents((prevEvents) => {
            // Merge and deduplicate by a combined key
            const merged = [...normalizedHistory, ...prevEvents];
            const seen = new Set();
            const deduped = [];
            
            // Sort by timestamp if available to ensure correct order
            merged.sort((a, b) => (a.ts || 0) - (b.ts || 0));

            for (const evt of merged) {
              const key = [
                evt.type,
                evt.experiment_id || '',
                evt.result_id || '',
                evt.index || '',
                evt.fingerprint || '',
                evt.generation || '',
                evt.ts || ''
              ].join('|');
              
              if (seen.has(key)) continue;
              seen.add(key);
              deduped.push(evt);
            }
            return deduped.slice(-Math.max(LIVE_FEED_MAX_EVENTS, 150));
          });
        })
        .catch((err) => console.error('LiveFeed: Gap heal fetch failed', err));

      // Also restore loss curve on reconnect
      apiCall(`/api/live-loss-curve`)
        .then(r => r.json())
        .then(curve => {
          if (Array.isArray(curve) && curve.length >= 2) {
            const curveExpId = curve[0]?.experiment_id || '';
            if (curveExpId === experimentId) {
              lossCurveExpRef.current = curveExpId;
              setLossCurve(curve.map(p => ({
                step: p.step, loss: p.loss,
                total_steps: p.total_steps, phase: p.phase, received_ts: Date.now(),
              })));
            }
          }
        })
        .catch(() => {});
    }
    prevConnectedRef.current = connected;
  }, [connected, experimentId, apiBase]);

  const latestValidationProgress = useMemo(
    () => [...displayEvents].reverse().find((evt) => evt?.type === 'validate_progress'),
    [displayEvents],
  );

  const latestValidationPhase = useMemo(
    () => [...displayEvents].reverse().find((evt) => evt?.type === 'validate_phase'),
    [displayEvents],
  );

  const latestValidationCompletion = useMemo(
    () => [...displayEvents].reverse().find((evt) => evt?.type === 'validate_complete'),
    [displayEvents],
  );

  const lossCurveMeta = useMemo(() => {
    const lastPoint = lossCurve.length ? lossCurve[lossCurve.length - 1] : null;
    const segments = splitCurveIntoSegments(lossCurve);
    const staleSeconds = lastPoint?.received_ts ? Math.max(0, Math.floor((nowTs - lastPoint.received_ts) / 1000)) : 0;
    return { lastPoint, segmentCount: segments.length, staleSeconds };
  }, [lossCurve, nowTs]);

  const liveStatus = useMemo(() => {
    const status = String(progress?.status || '').toLowerCase();
    if (status === 'completed') {
      return {
        tone: 'success',
        text: 'Experiment completed. Analysis recorded in the notebook.',
      };
    }
    if (status === 'failed') {
      return {
        tone: 'warn',
        text: `Experiment failed${progress?.error ? `: ${progress.error}` : '.'}`,
      };
    }
    if (status === 'validating' || status.startsWith('validation:')) {
      const seedText = latestValidationProgress?.seed && latestValidationProgress?.total_seeds
        ? `seed ${latestValidationProgress.seed}/${latestValidationProgress.total_seeds}`
        : `seed run ${lossCurveMeta.segmentCount || 1}`;
      const phaseText = latestValidationPhase?.phase
        ? ` — ${latestValidationPhase.phase}`
        : '';
      const testProgress = latestValidationPhase?.test_index && latestValidationPhase?.total_tests
        ? ` (${latestValidationPhase.test_index}/${latestValidationPhase.total_tests})`
        : '';
      if (lossCurveMeta.staleSeconds >= 15) {
        return {
          tone: 'warn',
          text: `Validation active on ${seedText}${phaseText}${testProgress}, but loss updates idle for ${lossCurveMeta.staleSeconds}s.`,
        };
      }
      return {
        tone: 'info',
        text: `Validating ${latestValidationProgress?.source_result_id?.slice(0, 8) || 'candidate'} on ${seedText}${phaseText}${testProgress}`,
      };
    }
    if (latestValidationCompletion) {
      return {
        tone: 'success',
        text: 'Validation completed. Analysis recorded in the notebook.',
      };
    }
    return {
      tone: 'info',
      text: progress?.aria_message || '',
    };
  }, [progress, latestValidationProgress, latestValidationPhase, latestValidationCompletion, lossCurveMeta]);

  const curveCards = useMemo(() => {
    const currentExperimentId = lossCurveExpRef.current || activeExperimentRef.current || null;
    const currentCard = buildCurveSnapshot(currentExperimentId, lossCurve, {
      statusText: liveStatus.text,
      statusTone: liveStatus.tone,
      label: currentExperimentId ? `Live ${String(currentExperimentId).slice(0, 8)}` : 'Live run',
    });
    const historicalCards = curveHistory.filter((card) => card.experimentId !== currentExperimentId);
    const cards = currentCard
      ? [...historicalCards.slice(0, Math.max(0, LIVE_FEED_MAX_GRAPHS - 1)).reverse(), currentCard]
      : historicalCards.slice(0, LIVE_FEED_MAX_GRAPHS).reverse();
    return cards;
  }, [curveHistory, liveStatus.text, liveStatus.tone, lossCurve]);

  const noveltyChartPoints = useMemo(() => {
    const currentExperimentId = activeExperimentRef.current || experimentId || null;
    return displayEvents
      .filter((event) => event?.type === 'nov_gen' && event?.experiment_id === currentExperimentId)
      .map((event) => ({
        generation: Number(event.generation) || 0,
        total_generations: Number(event.total_generations) || 0,
        best_fitness: Number(event.best_fitness) || 0,
        best_novelty: Number(event.best_novelty) || 0,
        archive_size: Number(event.archive_size) || 0,
      }))
      .sort((a, b) => a.generation - b.generation);
  }, [displayEvents, experimentId]);

  return (
    <div className="live-feed">
      <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span>Live Feed</span>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* Connection status — visible at all times */}
          <span
            className={`connection-dot ${connected ? 'connected' : ''}`}
            role="status"
            aria-label={connected ? 'Connected' : 'Reconnecting'}
          />
          <span style={{
            fontSize: 11,
            fontWeight: 700,
            textTransform: 'uppercase',
            color: connected ? 'var(--accent-green)' : 'var(--accent-yellow)',
            minWidth: 80,
          }}>
            {connected ? 'Live' : 'Reconnecting'}
          </span>
          {/* Auto-scroll toggle — always visible, not hidden behind hover */}
          <button
            className={`refresh-btn ${!autoScroll ? 'active' : ''}`}
            style={{
              fontSize: 11,
              padding: '3px 8px',
            }}
            aria-pressed={!autoScroll}
            onClick={() => setAutoScroll(!autoScroll)}
            title={autoScroll ? 'Pause auto-scroll' : 'Resume auto-scroll'}
          >
            {autoScroll ? 'Pause Scroll' : 'Resume Scroll'}
          </button>
        </div>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Real-time stream of architectures being tested. Green = survived, red = failed.
      </p>
      {!connected && events.length > 0 && (
        <div className="reconnect-banner">
          <span className="pulse-dot" />
          Connection lost. Reconnecting and healing feed gaps...
        </div>
      )}
      {(liveStatus.text || curveCards.length > 0 || noveltyChartPoints.length > 0) && (
        <div style={{ marginBottom: 8 }}>
          {liveStatus.text && (
            <div style={{ fontSize: 11, color: liveStatus.tone === 'warn' ? 'var(--accent-yellow)' : liveStatus.tone === 'success' ? 'var(--accent-green)' : 'var(--text-secondary)', marginBottom: 6 }}>
              {liveStatus.text}
            </div>
          )}
          {curveCards.length > 0 && (
            <div style={{
              display: 'flex',
              gap: 8,
              overflowX: 'auto',
              paddingBottom: 4,
              justifyContent: 'flex-start',
            }}>
              {curveCards.map((card) => (
                <div key={card.experimentId} style={{ flex: '0 0 auto' }}>
                  <MiniLossChart
                    curve={card.curve}
                    statusText={card.statusText}
                    statusTone={card.statusTone}
                    label={card.label}
                    width={600}
                  />
                </div>
              ))}
            </div>
          )}
          {curveCards.length === 0 && noveltyChartPoints.length > 0 && (
            <MiniNoveltyChart
              points={noveltyChartPoints}
              label={activeExperimentRef.current ? `Novelty ${String(activeExperimentRef.current).slice(0, 8)}` : 'Novelty run'}
              width={600}
            />
          )}
        </div>
      )}
      <div className="feed-container" ref={feedRef}>
        {events.length === 0 ? (
          <div className="feed-empty">
            {connected ? 'Waiting for experiment events...' : 'Unable to connect to event stream. Is the server running?'}
          </div>
        ) : (
          displayEvents.map((evt, i) => {
            const prev = i > 0 ? displayEvents[i - 1] : null;
            const prevExpId = prev?.experiment_id || null;
            const currExpId = evt?.experiment_id || null;
            const hasRunBoundary =
              i > 0 &&
              RUN_START_EVENT_TYPES.has(evt?.type) &&
              prevExpId &&
              currExpId &&
              prevExpId !== currExpId;
            return (
            <div
              key={i}
              className={`feed-item feed-${evt.type}`}
              style={hasRunBoundary ? { marginTop: 12, paddingTop: 12, borderTop: '1px solid var(--border)' } : undefined}
            >
              {evt.type === 'program' && (
                <>
                  <span className="feed-index">#{evt.index + 1}</span>
                  <span className="feed-fp">{evt.fingerprint}</span>
                  <span
                    className="feed-result"
                    style={{ color: RESULT_COLORS[evt.result] || 'var(--text-secondary)' }}
                  >
                    {evt.result}
                  </span>
                  {evt.loss_ratio != null && (
                    <span className="feed-loss">L:{evt.loss_ratio}</span>
                  )}
                  {evt.novelty != null && (
                    <span className="feed-novelty">N:{evt.novelty}</span>
                  )}
                  {evt.throughput != null && (
                    <span className="feed-metric">{evt.throughput}tok/s</span>
                  )}
                  {evt.memory_mb != null && (
                    <span className="feed-metric">{evt.memory_mb}MB</span>
                  )}
                  {evt.params != null && (
                    <span className="feed-params">{(evt.params / 1000).toFixed(0)}K</span>
                  )}
                  {evt.stability != null && (
                    <span className="feed-metric" style={{ color: 'var(--accent-yellow)' }}>stab:{evt.stability}</span>
                  )}
                  {evt.has_nan && (
                    <span style={{ color: 'var(--accent-red)', fontWeight: 600, marginLeft: 4 }}>NaN!</span>
                  )}
                  {evt.has_inf && (
                    <span style={{ color: 'var(--accent-red)', fontWeight: 600, marginLeft: 4 }}>Inf!</span>
                  )}
                  {evt.error && (
                    <span className="feed-error-detail" style={{ color: 'var(--accent-orange)', marginLeft: 4, fontSize: 11 }}>
                      {evt.error_type ? `[${evt.error_type}] ` : ''}{evt.error}
                    </span>
                  )}
                </>
              )}
              {evt.type === 'start' && (
                <span className="feed-event-msg">
                  Experiment started: {evt.experiment_id?.slice(0, 8)} —
                  "{evt.hypothesis?.slice(0, 60)}"
                </span>
              )}
              {evt.type === 'complete' && (
                <span className="feed-event-msg feed-success">
                  Experiment {evt.experiment_id?.slice(0, 8)} completed!
                </span>
              )}
              {evt.type === 'failed' && (
                <span className="feed-event-msg feed-error">
                  Experiment failed: {evt.error?.slice(0, 80)}
                </span>
              )}
              {evt.type === 'stopping' && (
                <span className="feed-event-msg">Stopping experiment...</span>
              )}
              {/* Evolution events */}
              {evt.type === 'evo_start' && (
                <span className="feed-event-msg">
                  Evolution started: {evt.experiment_id?.slice(0, 8)} —
                  "{evt.hypothesis?.slice(0, 60)}"
                </span>
              )}
              {evt.type === 'evo_gen' && (
                <span className="feed-event-msg">
                  {evt._runStart && (
                    <div style={{ color: 'var(--text-muted)' }}>
                      Evolution run {(evt.experiment_id || '').slice(0, 8) || 'unknown'}
                    </div>
                  )}
                  {(evt._missingPrefixTo || evt._missingGapTo) && (
                    <div style={{ color: 'var(--accent-yellow)' }}>
                      {evt._missingPrefixTo
                        ? `Generations ${evt._missingPrefixFrom}-${evt._missingPrefixTo} are not in current feed history.`
                        : `Generations ${evt._missingGapFrom}-${evt._missingGapTo} are not in current feed history.`}
                    </div>
                  )}
                  <div>
                    Gen {evt.generation}/{evt.total_generations}:
                    best={evt.best_fitness?.toFixed(3)},
                    avg={evt.avg_fitness?.toFixed(3)},
                    pop={evt.population_size}
                  </div>
                  {evt.n_routing != null && (
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                      Routing: {evt.n_routing} | Standard: {evt.n_standard}
                    </div>
                  )}
                </span>
              )}
              {evt.type === 'evo_complete' && (
                <span className="feed-event-msg feed-success">
                  Evolution {evt.experiment_id?.slice(0, 8)} completed!
                </span>
              )}
              {/* Novelty events */}
              {evt.type === 'nov_start' && (
                <span className="feed-event-msg">
                  Novelty search started: {evt.experiment_id?.slice(0, 8)} —
                  "{evt.hypothesis?.slice(0, 60)}"
                </span>
              )}
              {evt.type === 'nov_gen' && (
                <span className="feed-event-msg">
                  {evt._runStart && (
                    <div style={{ color: 'var(--text-muted)' }}>
                      Novelty run {(evt.experiment_id || '').slice(0, 8) || 'unknown'}
                    </div>
                  )}
                  {(evt._missingPrefixTo || evt._missingGapTo) && (
                    <div style={{ color: 'var(--accent-yellow)' }}>
                      {evt._missingPrefixTo
                        ? `Generations ${evt._missingPrefixFrom}-${evt._missingPrefixTo} are not in current feed history.`
                        : `Generations ${evt._missingGapFrom}-${evt._missingGapTo} are not in current feed history.`}
                    </div>
                  )}
                  <div>
                    Gen {evt.generation}/{evt.total_generations}:
                    best_fit={evt.best_fitness?.toFixed(3)},
                    archive={evt.archive_size},
                    novelty={evt.best_novelty?.toFixed(3)}
                  </div>
                  {evt.n_routing != null && (
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                      Routing: {evt.n_routing} | Standard: {evt.n_standard}
                    </div>
                  )}
                </span>
              )}
              {evt.type === 'nov_complete' && (
                <span className="feed-event-msg feed-success">
                  Novelty search {evt.experiment_id?.slice(0, 8)} completed!
                  Archive: {evt.archive_size} behaviors
                </span>
              )}
              {evt.type === 'limit_reached' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-yellow)' }}>
                  Session ended: {evt.reason}
                  {evt.experiments_completed && ` (${evt.experiments_completed} experiments`}
                  {evt.estimated_cost > 0 && `, $${evt.estimated_cost.toFixed(2)} spent`}
                  {evt.experiments_completed && ')'}
                </span>
              )}
              {/* Scale-up events */}
              {evt.type === 'scaleup_start' && (
                <span className="feed-event-msg">
                  Scale-up started: {evt.experiment_id?.slice(0, 8)} —
                  {evt.result_ids?.length} program(s),
                  {evt.config?.steps} steps
                </span>
              )}
              {evt.type === 'scaleup_progress' && (
                <span className="feed-event-msg">
                  Scale-up {evt.current_program}/{evt.total_programs}:
                  {evt.source_result_id?.slice(0, 8)} — {evt.status}
                  {evt.passed != null && (
                    <span style={{
                      color: evt.passed ? 'var(--accent-green)' : 'var(--accent-red)',
                      marginLeft: 4,
                    }}>
                      {evt.passed ? 'PASS' : 'FAIL'}
                    </span>
                  )}
                  {evt.loss_ratio != null && (
                    <span style={{ marginLeft: 4 }}>L:{evt.loss_ratio}</span>
                  )}
                </span>
              )}
              {evt.type === 'scaleup_complete' && (
                <span className="feed-event-msg feed-success">
                  Scale-up {evt.experiment_id?.slice(0, 8)} completed!
                </span>
              )}
              {evt.type === 'auto_scaleup' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-green)' }}>
                  Auto-scale-up queued: {evt.n_programs} program(s) — {evt.reason}
                </span>
              )}
              {/* Mode selection */}
              {evt.type === 'mode_selected' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-purple)' }}>
                  Aria selected mode: <strong>{evt.mode}</strong> — {evt.reasoning}
                </span>
              )}
              {/* Investigation events */}
              {evt.type === 'invest_start' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-yellow)' }}>
                  Investigation started: {evt.experiment_id?.slice(0, 8)} —
                  {evt.n_candidates} candidate(s)
                </span>
              )}
              {evt.type === 'invest_progress' && (
                (() => {
                  const candidateCurrent = evt.current_candidate ?? evt.current;
                  const candidateTotal = evt.total_candidates ?? evt.total;
                  const sourceId = evt.result_id || evt.source_result_id;
                  const programCurrent = evt.current_program ?? evt.training_program;
                  const programTotal = evt.total_programs;
                  return (
                    <span className="feed-event-msg">
                      Investigating {candidateCurrent ?? '?'} / {candidateTotal ?? '?'}:
                      {' '}{sourceId?.slice(0, 8) || 'unknown'}
                      {(programCurrent != null || programTotal != null) && (
                        <>
                          {' '}— program {programCurrent ?? '?'} / {programTotal ?? '?'}
                        </>
                      )}
                      {evt.loss_ratio != null && (
                        <span style={{ marginLeft: 4 }}>L:{evt.loss_ratio}</span>
                      )}
                    </span>
                  );
                })()
              )}
              {evt.type === 'invest_complete' && (
                <span className="feed-event-msg feed-success">
                  Investigation {evt.experiment_id?.slice(0, 8)} completed!
                  {evt.n_passed != null && ` ${evt.n_passed} passed`}
                </span>
              )}
              {/* Validation events */}
              {evt.type === 'validate_start' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-purple)' }}>
                  Validation started: {evt.experiment_id?.slice(0, 8)} —
                  {evt.n_candidates || evt.result_ids?.length || '?'} candidate(s)
                </span>
              )}
              {evt.type === 'validate_progress' && (
                <span className="feed-event-msg">
                  Validating {evt.current}/{evt.total}:
                  {' '}{(evt.source_result_id || '')?.slice(0, 8)} — seed {evt.seed}/{evt.total_seeds}
                  {evt.loss_ratio != null && (
                    <span style={{ marginLeft: 4 }}>L:{evt.loss_ratio}</span>
                  )}
                  {evt.status === 'starting' && ' starting...'}
                </span>
              )}
              {evt.type === 'validate_phase' && (
                <span className="feed-event-msg" style={{ color: 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontFamily: 'monospace', fontSize: 11 }}>
                    {(evt.result_id || evt.experiment_id || '').slice(0, 8)}
                  </span>
                  <span>{evt.phase}</span>
                  {(() => {
                    const idx = evt.test_index ?? evt.outer_index;
                    const total = evt.total_tests ?? evt.outer_total;
                    if (idx == null || total == null) return null;
                    return (
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                        <span style={{
                          display: 'inline-block',
                          width: 60,
                          height: 6,
                          borderRadius: 3,
                          background: 'rgba(139,92,246,0.15)',
                          overflow: 'hidden',
                        }}>
                          <span style={{
                            display: 'block',
                            width: `${Math.round((idx / total) * 100)}%`,
                            height: '100%',
                            borderRadius: 3,
                            background: 'var(--accent-purple)',
                            transition: 'width 0.3s ease',
                          }} />
                        </span>
                        <span style={{ fontSize: 10, fontFamily: 'monospace', opacity: 0.7 }}>
                          {idx}/{total}
                        </span>
                      </span>
                    );
                  })()}
                </span>
              )}
              {evt.type === 'validate_complete' && (
                <span className="feed-event-msg feed-success">
                  Validation {evt.experiment_id?.slice(0, 8)} completed!
                  {evt.error ? (
                    <span style={{ color: 'var(--accent-red)' }}> Error: {evt.error}</span>
                  ) : evt.summary ? (
                    <span style={{ color: 'var(--text-secondary)', marginLeft: 4 }}>
                      {evt.summary.split('\n').pop()?.slice(0, 80)}
                    </span>
                  ) : null}
                </span>
              )}
              {/* Breakthrough */}
              {evt.type === 'breakthrough' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-green)', fontWeight: 600 }}>
                  BREAKTHROUGH DETECTED: {evt.result_id?.slice(0, 8)} —
                  baseline ratio: {evt.baseline_ratio?.toFixed(4)}
                </span>
              )}
              {/* Auto-escalation */}
              {evt.type === 'auto_investigate' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-yellow)' }}>
                  Auto-investigation queued: {evt.n_candidates} candidate(s)
                </span>
              )}
              {evt.type === 'auto_validate' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-purple)' }}>
                  Auto-validation queued: {evt.n_candidates} candidate(s)
                </span>
              )}
              {evt.type === 'auto_report' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-purple)' }}>
                  Research report generated ({evt.reason}) — {evt.narrative_length} chars
                </span>
              )}
              {evt.type === 'recommendation' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-purple)' }}>
                  Aria suggests: {evt.reasoning?.slice(0, 100)}
                  {evt.confidence != null && ` (${(evt.confidence * 100).toFixed(0)}% confidence)`}
                </span>
              )}
              {/* Hypothesis events */}
              {evt.type === 'hyp_recorded' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-blue)' }}>
                  Hypothesis: {evt.prediction?.slice(0, 80)}
                  {evt.confidence != null && ` (${(evt.confidence * 100).toFixed(0)}% confidence)`}
                </span>
              )}
              {evt.type === 'hyp_resolved' && (
                <span className="feed-event-msg" style={{
                  color: evt.status === 'confirmed' ? 'var(--accent-green)' :
                         evt.status === 'refuted' ? 'var(--accent-red)' : 'var(--accent-yellow)',
                }}>
                  Hypothesis {evt.status?.toUpperCase()}: {evt.evidence?.slice(0, 80)}
                  {evt.confidence_after != null && ` (${(evt.confidence_after * 100).toFixed(0)}%)`}
                </span>
              )}
              {/* Decision events */}
              {evt.type === 'decision' && (
                <span className="feed-event-msg" style={{
                  color: evt.decision_type === 'go' ? 'var(--accent-green)' :
                         evt.decision_type === 'no_go' ? 'var(--accent-red)' : 'var(--accent-yellow)',
                  fontWeight: 600,
                  whiteSpace: 'normal', wordBreak: 'break-word',
                }}>
                  {evt.decision_type?.toUpperCase().replace('_', '-')}: {evt.subject}
                  {evt.rationale && ` — ${evt.rationale}`}
                </span>
              )}
              {/* Knowledge events */}
              {evt.type === 'knowledge' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-purple)' }}>
                  Knowledge extracted: {evt.n_entries} insight(s)
                  {evt.categories && ` [${evt.categories.join(', ')}]`}
                </span>
              )}
              {/* Campaign events */}
              {evt.type === 'campaign_created' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>
                  New campaign: {evt.title}
                </span>
              )}
              {evt.type === 'campaign_completed' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-green)', fontWeight: 600 }}>
                  Campaign completed: {evt.title}
                </span>
              )}
              {evt.type === 'aria_phase' && (
                <span className="feed-event-msg" style={{ color: 'var(--text-secondary)' }}>
                  Aria phase: <strong>{evt.phase_label || evt.phase || 'running'}</strong>
                  {evt.selected_mode && <> — mode {evt.selected_mode}</>}
                  {evt.last_note && <> — {evt.last_note}</>}
                </span>
              )}
              {/* Learning events */}
              {evt.type === 'learning' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-orange, #f0883e)', fontWeight: 600 }}>
                  {evt.description || 'Grammar weights adjusted'}
                  {evt.n_changed != null && ` (${evt.n_changed} categories)`}
                </span>
              )}
              {evt.type === 'log' && (
                <span className="feed-event-msg" style={{
                  fontFamily: 'monospace',
                  fontSize: 11,
                  color: evt.level === 'WARNING' ? 'var(--accent-yellow)'
                       : evt.level === 'ERROR' ? 'var(--accent-red)'
                       : 'var(--text-secondary)',
                }}>
                  <span style={{ opacity: 0.5, marginRight: 4 }}>[{evt.logger}]</span>
                  {evt.message}
                </span>
              )}
            </div>
          )})
        )}
      </div>
    </div>
  );
}

export default LiveFeed;
