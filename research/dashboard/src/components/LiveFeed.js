import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useEventBus } from '../hooks/useEventBus';
import apiService from '../services/apiService';

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
  continuous_limit_reached: 'limit_reached',
  learning_event: 'learning',
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
  'limit_reached',
  'learning',
]);

const GENERATION_EVENT_TYPES = new Set(['evo_gen', 'nov_gen']);

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

function LiveFeed({ apiBase, experimentId = null }) {
  const [events, setEvents] = useState([]);
  const feedRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [showControls, setShowControls] = useState(false);
  const prevConnectedRef = useRef(null);
  const displayEvents = useMemo(() => annotateGenerationHistory(events), [events]);

  // Helper to add an event with a mapped type
  const addEvent = useCallback((type) => (data) => {
    const normalized = normalizeLiveFeedEvent({ type, ...data, ts: Date.now() });
    if (!normalized) return;
    setEvents(prev => [...prev.slice(-99), normalized]);
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
  useEventBus('continuous_limit_reached', addEvent('limit_reached'));
  useEventBus('learning_event', addEvent('learning'));

  // Load history from REST when experimentId changes
  useEffect(() => {
    setEvents([]);
    if (!experimentId) return;

    apiService.getLiveFeed(experimentId, 100)
      .then((history) => {
        if (!Array.isArray(history) || history.length === 0) return;
        const normalizedHistory = history
          .map((event) => normalizeLiveFeedEvent(event))
          .filter(Boolean);
        if (normalizedHistory.length === 0) return;
        setEvents((prev) => [...normalizedHistory, ...prev].slice(-100));
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
            return deduped.slice(-150);
          });
        })
        .catch((err) => console.error('LiveFeed: Gap heal fetch failed', err));
    }
    prevConnectedRef.current = connected;
  }, [connected, experimentId]);

  return (
    <div
      className="live-feed"
      onMouseEnter={() => setShowControls(true)}
      onMouseLeave={() => setShowControls(false)}
    >
      <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span>Live Feed</span>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className={`connection-dot ${connected ? 'connected' : ''}`}></span>
          <span style={{
            fontSize: 11,
            fontWeight: 700,
            textTransform: 'uppercase',
            color: connected ? 'var(--accent-green)' : 'var(--accent-yellow)',
            minWidth: 80,
          }}>
            {connected ? 'Live' : 'Reconnecting'}
          </span>
          <button
            className={`refresh-btn ${!autoScroll ? 'active' : ''}`}
            style={{ 
              fontSize: 11, 
              padding: '2px 8px', 
              opacity: showControls ? 1 : 0, 
              transition: 'opacity 0.2s',
              pointerEvents: showControls ? 'auto' : 'none',
              background: !autoScroll ? 'rgba(88, 166, 255, 0.15)' : 'var(--bg-tertiary)',
              color: !autoScroll ? 'var(--accent-blue)' : 'var(--text-primary)',
              borderColor: !autoScroll ? 'var(--accent-blue)' : 'var(--border)',
            }}
            onClick={() => setAutoScroll(!autoScroll)}
          >
            {autoScroll ? 'Pause Scroll' : 'Resume Scroll'}
          </button>
        </div>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Real-time stream of architectures being tested. Each line shows a generated computation graph and whether it passed or failed. Green = survived, red = failed at some stage.
      </p>
      {!connected && events.length > 0 && (
        <div className="reconnect-banner">
          <span className="pulse-dot" />
          Connection lost. Reconnecting and healing feed gaps...
        </div>
      )}
      <div className="feed-container" ref={feedRef}>
        {events.length === 0 ? (
          <div className="feed-empty">
            {connected ? 'Waiting for experiment events...' : 'Unable to connect to event stream. Is the server running?'}
          </div>
        ) : (
          displayEvents.map((evt, i) => (
            <div key={i} className={`feed-item feed-${evt.type}`}>
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
                <span className="feed-event-msg">
                  Investigating {evt.current_candidate}/{evt.total_candidates}:
                  {evt.result_id?.slice(0, 8)} — program {evt.current_program}/{evt.total_programs}
                  {evt.loss_ratio != null && (
                    <span style={{ marginLeft: 4 }}>L:{evt.loss_ratio}</span>
                  )}
                </span>
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
              {/* Learning events */}
              {evt.type === 'learning' && (
                <span className="feed-event-msg" style={{ color: 'var(--accent-orange, #f0883e)', fontWeight: 600 }}>
                  {evt.description || 'Grammar weights adjusted'}
                  {evt.n_changed != null && ` (${evt.n_changed} categories)`}
                </span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default LiveFeed;
