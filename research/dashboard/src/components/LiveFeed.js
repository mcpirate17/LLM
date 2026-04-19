import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useEventBus } from '../hooks/useEventBus';
import apiService, { apiCall } from '../services/apiService';
import FeedItem from './liveFeed/FeedItem';
import { MiniLossChart, MiniNoveltyChart } from './liveFeed/LiveFeedCharts';
import {
  CONTEXT_SWITCH_EVENT_TYPES,
  LIVE_FEED_MAX_EVENTS,
  LIVE_FEED_MAX_GRAPHS,
  LIVE_LOSS_CURVE_MAX_POINTS,
  PROGRESSION_EVENT_TYPES,
  RUN_START_EVENT_TYPES,
  TERMINAL_EVENT_TYPES,
} from './liveFeed/constants';
import {
  annotateGenerationHistory,
  buildCurveSnapshot,
  describeCurveEvent,
  normalizeLiveFeedEvent,
  reconcileTerminalEvents,
  splitCurveIntoSegments,
} from './liveFeed/utils';

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
  const eventQueueRef = useRef([]);
  const eventFlushTimerRef = useRef(null);
  const pendingLossPointRef = useRef(null);
  const lossFlushTimerRef = useRef(null);
  const prevConnectedRef = useRef(null);
  const displayEvents = useMemo(
    () => annotateGenerationHistory(reconcileTerminalEvents(events)),
    [events]
  );
  const mainDisplayEvents = useMemo(
    () => displayEvents.filter((evt) => !PROGRESSION_EVENT_TYPES.has(evt?.type)),
    [displayEvents]
  );
  const progressionDisplayEvents = useMemo(
    () => displayEvents.filter((evt) => PROGRESSION_EVENT_TYPES.has(evt?.type)),
    [displayEvents]
  );

  const isValidationActive = useMemo(() => {
    const status = String(progress?.status || '').toLowerCase();
    return status === 'validating' || status.startsWith('validation:');
  }, [progress?.status]);

  useEffect(() => {
    if (!isValidationActive || lossCurve.length === 0) return undefined;
    const interval = setInterval(() => setNowTs(Date.now()), 10000);
    return () => clearInterval(interval);
  }, [isValidationActive, lossCurve.length]);

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

  const flushQueuedEvents = useCallback(() => {
    eventFlushTimerRef.current = null;
    const queued = eventQueueRef.current;
    if (queued.length === 0) return;
    eventQueueRef.current = [];
    setEvents((prev) => [...prev, ...queued].slice(-LIVE_FEED_MAX_EVENTS));
  }, []);

  const enqueueEvent = useCallback((event) => {
    eventQueueRef.current.push(event);
    if (eventFlushTimerRef.current === null) {
      eventFlushTimerRef.current = setTimeout(flushQueuedEvents, 120);
    }
  }, [flushQueuedEvents]);

  const flushPendingLossPoint = useCallback(() => {
    lossFlushTimerRef.current = null;
    const pendingPoint = pendingLossPointRef.current;
    if (!pendingPoint) return;
    pendingLossPointRef.current = null;
    setLossCurve((prev) => {
      if (lossCurveExpRef.current !== pendingPoint.expId) {
        lossCurveExpRef.current = pendingPoint.expId;
        return [pendingPoint.point];
      }
      const next = [...prev, pendingPoint.point];
      return next.length > LIVE_LOSS_CURVE_MAX_POINTS
        ? next.slice(-LIVE_LOSS_CURVE_MAX_POINTS)
        : next;
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
      eventQueueRef.current = [];
      if (eventFlushTimerRef.current !== null) {
        clearTimeout(eventFlushTimerRef.current);
        eventFlushTimerRef.current = null;
      }
      setEvents([]);
      setLossCurve([]);
    }

    // If we missed the explicit start event (SSE reconnect, queue overflow),
    // let later phase-specific events switch context as well.
    if (currentExpId && eventExpId && eventExpId !== currentExpId) {
      if (CONTEXT_SWITCH_EVENT_TYPES.has(type)) {
        activeExperimentRef.current = eventExpId;
        lossCurveExpRef.current = eventExpId;
        eventQueueRef.current = [];
        if (eventFlushTimerRef.current !== null) {
          clearTimeout(eventFlushTimerRef.current);
          eventFlushTimerRef.current = null;
        }
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
    enqueueEvent(normalized);
  }, [archiveCurveSnapshot, enqueueEvent]);

  // Handle training_step events for the mini loss chart
  const handleTrainingStep = useCallback((data) => {
    const expId = data.experiment_id || '';
    const currentExpId = activeExperimentRef.current || null;
    if (currentExpId && expId && expId !== currentExpId) return;
    if (!currentExpId && expId) activeExperimentRef.current = expId;
    pendingLossPointRef.current = {
      expId,
      point: {
        step: data.step,
        loss: data.loss,
        total_steps: data.total_steps,
        phase: data.phase,
        received_ts: Date.now(),
      },
    };
    if (lossFlushTimerRef.current === null) {
      lossFlushTimerRef.current = setTimeout(flushPendingLossPoint, 250);
    }
  }, [flushPendingLossPoint]);

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

  useEffect(() => () => {
    if (eventFlushTimerRef.current !== null) {
      clearTimeout(eventFlushTimerRef.current);
      eventFlushTimerRef.current = null;
    }
    if (lossFlushTimerRef.current !== null) {
      clearTimeout(lossFlushTimerRef.current);
      lossFlushTimerRef.current = null;
    }
  }, []);

  // Fetch generic loss curve only when no experiment is selected.
  useEffect(() => {
    if (experimentId) return undefined;
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
    return undefined;
  }, [apiBase, experimentId]);

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

  const { latestValidationProgress, latestValidationPhase, latestValidationCompletion } = useMemo(() => {
    let progress = null, phase = null, completion = null;
    for (let i = displayEvents.length - 1; i >= 0; i--) {
      const evt = displayEvents[i];
      if (!progress && evt?.type === 'validate_progress') progress = evt;
      if (!phase && evt?.type === 'validate_phase') phase = evt;
      if (!completion && evt?.type === 'validate_complete') completion = evt;
      if (progress && phase && completion) break;
    }
    return { latestValidationProgress: progress, latestValidationPhase: phase, latestValidationCompletion: completion };
  }, [displayEvents]);

  const latestTerminalEvent = useMemo(() => {
    const currentExperimentId = activeExperimentRef.current || experimentId || null;
    for (let i = mainDisplayEvents.length - 1; i >= 0; i--) {
      const evt = mainDisplayEvents[i];
      if (!TERMINAL_EVENT_TYPES.has(evt?.type)) continue;
      if (currentExperimentId && evt?.experiment_id !== currentExperimentId) continue;
      return evt;
    }
    return null;
  }, [mainDisplayEvents, experimentId]);

  const lossCurveMeta = useMemo(() => {
    const lastPoint = lossCurve.length ? lossCurve[lossCurve.length - 1] : null;
    const segments = splitCurveIntoSegments(lossCurve);
    const staleSeconds = lastPoint?.received_ts ? Math.max(0, Math.floor((nowTs - lastPoint.received_ts) / 1000)) : 0;
    return { lastPoint, segmentCount: segments.length, staleSeconds };
  }, [lossCurve, nowTs]);

  const liveStatus = useMemo(() => {
    if (latestTerminalEvent?.type === 'complete') {
      return {
        tone: 'success',
        text: 'Experiment completed. Analysis recorded in the notebook.',
      };
    }
    if (latestTerminalEvent?.type === 'failed') {
      return {
        tone: 'warn',
        text: `Experiment failed${latestTerminalEvent?.error ? `: ${latestTerminalEvent.error}` : '.'}`,
      };
    }
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
  }, [progress, latestTerminalEvent, latestValidationProgress, latestValidationPhase, latestValidationCompletion, lossCurveMeta]);

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
          <>
            {mainDisplayEvents.map((evt, i) => (
              <FeedItem
                key={`main-${i}`}
                evt={evt}
                prevExpId={i > 0 ? (mainDisplayEvents[i - 1]?.experiment_id || null) : null}
              />
            ))}
            {progressionDisplayEvents.length > 0 && (
              <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
                <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 8 }}>
                  Progression Activity
                </div>
                {progressionDisplayEvents.map((evt, i) => (
                  <FeedItem
                    key={`progress-${i}`}
                    evt={evt}
                    prevExpId={i > 0 ? (progressionDisplayEvents[i - 1]?.experiment_id || null) : null}
                  />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default LiveFeed;
