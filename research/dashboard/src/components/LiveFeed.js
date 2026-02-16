import React, { useState, useEffect, useRef } from 'react';

const RESULT_COLORS = {
  'S1 PASS': 'var(--accent-green)',
  'S0': 'var(--accent-blue)',
  'FAIL': 'var(--accent-red)',
  'invalid': 'var(--accent-red)',
  'compile_error': 'var(--accent-orange)',
};

function LiveFeed({ apiBase }) {
  const [events, setEvents] = useState([]);
  const [connected, setConnected] = useState(false);
  const feedRef = useRef(null);
  const eventSourceRef = useRef(null);

  useEffect(() => {
    const es = new EventSource(`${apiBase}/api/events`);
    eventSourceRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    es.addEventListener('program_evaluated', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), { type: 'program', ...data, ts: Date.now() }]);
    });

    es.addEventListener('experiment_started', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), { type: 'start', ...data, ts: Date.now() }]);
    });

    es.addEventListener('experiment_completed', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), { type: 'complete', ...data, ts: Date.now() }]);
    });

    es.addEventListener('experiment_failed', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), { type: 'failed', ...data, ts: Date.now() }]);
    });

    es.addEventListener('experiment_stopping', () => {
      setEvents(prev => [...prev.slice(-99), { type: 'stopping', ts: Date.now() }]);
    });

    // Evolution events
    es.addEventListener('evolution_started', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'evo_start', ...data, ts: Date.now()
      }]);
    });

    es.addEventListener('evolution_generation', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'evo_gen', ...data, ts: Date.now()
      }]);
    });

    es.addEventListener('evolution_completed', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'evo_complete', ...data, ts: Date.now()
      }]);
    });

    // Novelty events
    es.addEventListener('novelty_started', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'nov_start', ...data, ts: Date.now()
      }]);
    });

    es.addEventListener('novelty_generation', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'nov_gen', ...data, ts: Date.now()
      }]);
    });

    es.addEventListener('novelty_completed', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'nov_complete', ...data, ts: Date.now()
      }]);
    });

    // Continuous mode limit reached
    es.addEventListener('continuous_limit_reached', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'limit_reached', ...data, ts: Date.now()
      }]);
    });

    // Scale-up events
    es.addEventListener('scale_up_started', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'scaleup_start', ...data, ts: Date.now()
      }]);
    });

    es.addEventListener('scale_up_progress', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'scaleup_progress', ...data, ts: Date.now()
      }]);
    });

    es.addEventListener('scale_up_completed', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'scaleup_complete', ...data, ts: Date.now()
      }]);
    });

    // Auto-scale-up queued
    es.addEventListener('auto_scale_up_queued', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'auto_scaleup', ...data, ts: Date.now()
      }]);
    });

    // Mode selection events
    es.addEventListener('mode_selected', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'mode_selected', ...data, ts: Date.now()
      }]);
    });

    // Investigation events
    es.addEventListener('investigation_started', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'invest_start', ...data, ts: Date.now()
      }]);
    });

    es.addEventListener('investigation_progress', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'invest_progress', ...data, ts: Date.now()
      }]);
    });

    es.addEventListener('investigation_completed', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'invest_complete', ...data, ts: Date.now()
      }]);
    });

    // Validation events
    es.addEventListener('validation_started', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'validate_start', ...data, ts: Date.now()
      }]);
    });

    es.addEventListener('validation_progress', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'validate_progress', ...data, ts: Date.now()
      }]);
    });

    es.addEventListener('validation_completed', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'validate_complete', ...data, ts: Date.now()
      }]);
    });

    // Breakthrough detection
    es.addEventListener('breakthrough_detected', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'breakthrough', ...data, ts: Date.now()
      }]);
    });

    // Auto-escalation queued
    es.addEventListener('auto_investigate_queued', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'auto_investigate', ...data, ts: Date.now()
      }]);
    });

    es.addEventListener('auto_validate_queued', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'auto_validate', ...data, ts: Date.now()
      }]);
    });

    // Auto-report generated
    es.addEventListener('auto_report_generated', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'auto_report', ...data, ts: Date.now()
      }]);
    });

    // Hypothesis events
    es.addEventListener('hypothesis_recorded', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'hyp_recorded', ...data, ts: Date.now()
      }]);
    });

    es.addEventListener('hypothesis_resolved', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'hyp_resolved', ...data, ts: Date.now()
      }]);
    });

    // Decision events
    es.addEventListener('decision_recorded', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'decision', ...data, ts: Date.now()
      }]);
    });

    // Knowledge events
    es.addEventListener('knowledge_extracted', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'knowledge', ...data, ts: Date.now()
      }]);
    });

    // Campaign events
    es.addEventListener('campaign_created', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'campaign_created', ...data, ts: Date.now()
      }]);
    });

    es.addEventListener('campaign_completed', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'campaign_completed', ...data, ts: Date.now()
      }]);
    });

    // Aria auto-recommendation
    es.addEventListener('aria_recommendation', (e) => {
      const data = JSON.parse(e.data);
      setEvents(prev => [...prev.slice(-99), {
        type: 'recommendation', ...data, ts: Date.now()
      }]);
    });

    return () => {
      es.close();
      setConnected(false);
    };
  }, [apiBase]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [events]);

  return (
    <div className="card live-feed">
      <div className="card-title">
        Live Feed
        <span className={`connection-dot ${connected ? 'connected' : ''}`}></span>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Real-time stream of architectures being tested. Each line shows a generated computation graph and whether it passed or failed. Green = survived, red = failed at some stage.
      </p>
      {!connected && events.length > 0 && (
        <div style={{
          padding: '6px 12px', marginBottom: 8, borderRadius: 4,
          background: 'rgba(248, 81, 73, 0.1)', border: '1px solid rgba(248, 81, 73, 0.3)',
          fontSize: 12, color: 'var(--accent-red)', display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-red)', flexShrink: 0 }} />
          Connection lost. Reconnecting...
        </div>
      )}
      <div className="feed-container" ref={feedRef}>
        {events.length === 0 ? (
          <div className="feed-empty">
            {connected ? 'Waiting for experiment events...' : 'Unable to connect to event stream. Is the server running?'}
          </div>
        ) : (
          events.map((evt, i) => (
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
                  {evt.novelty != null && (
                    <span className="feed-novelty">N:{evt.novelty}</span>
                  )}
                  {evt.loss_ratio != null && (
                    <span className="feed-loss">L:{evt.loss_ratio}</span>
                  )}
                  {evt.params != null && (
                    <span className="feed-params">{(evt.params / 1000).toFixed(0)}K</span>
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
                  Gen {evt.generation}/{evt.total_generations}:
                  best={evt.best_fitness?.toFixed(3)},
                  avg={evt.avg_fitness?.toFixed(3)},
                  pop={evt.population_size}
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
                  Gen {evt.generation}/{evt.total_generations}:
                  best_fit={evt.best_fitness?.toFixed(3)},
                  archive={evt.archive_size},
                  novelty={evt.best_novelty?.toFixed(3)}
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
                }}>
                  {evt.decision_type?.toUpperCase().replace('_', '-')}: {evt.subject}
                  {evt.rationale && ` — ${evt.rationale.slice(0, 60)}`}
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
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default LiveFeed;
