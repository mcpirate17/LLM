import React from 'react';
import { RESULT_COLORS, RUN_BOUNDARY_STYLE, RUN_START_EVENT_TYPES } from './constants';

const FeedItem = React.memo(function FeedItem({ evt, prevExpId }) {
  const currExpId = evt?.experiment_id || null;
  const hasRunBoundary =
    prevExpId != null &&
    RUN_START_EVENT_TYPES.has(evt?.type) &&
    currExpId &&
    prevExpId !== currExpId;
  return (
    <div
      className={`feed-item feed-${evt.type}`}
      style={hasRunBoundary ? RUN_BOUNDARY_STYLE : undefined}
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
          {evt.loss_ratio != null && <span className="feed-loss">L:{evt.loss_ratio}</span>}
          {evt.novelty != null && <span className="feed-novelty">N:{evt.novelty}</span>}
          {evt.throughput != null && <span className="feed-metric">{evt.throughput}tok/s</span>}
          {evt.memory_mb != null && <span className="feed-metric">{evt.memory_mb}MB</span>}
          {evt.params != null && <span className="feed-params">{(evt.params / 1000).toFixed(0)}K</span>}
          {evt.stability != null && <span className="feed-metric" style={{ color: 'var(--accent-yellow)' }}>stab:{evt.stability}</span>}
          {evt.has_nan && <span style={{ color: 'var(--accent-red)', fontWeight: 600, marginLeft: 4 }}>NaN!</span>}
          {evt.has_inf && <span style={{ color: 'var(--accent-red)', fontWeight: 600, marginLeft: 4 }}>Inf!</span>}
          {evt.error && (
            <span className="feed-error-detail" style={{ color: 'var(--accent-orange)', marginLeft: 4, fontSize: 11 }}>
              {evt.error_type ? `[${evt.error_type}] ` : ''}{evt.error}
            </span>
          )}
        </>
      )}
      {evt.type === 'start' && <span className="feed-event-msg">Experiment started: {evt.experiment_id?.slice(0, 8)} — "{evt.hypothesis?.slice(0, 60)}"</span>}
      {evt.type === 'complete' && <span className="feed-event-msg feed-success">Experiment {evt.experiment_id?.slice(0, 8)} completed!</span>}
      {evt.type === 'failed' && <span className="feed-event-msg feed-error">Experiment failed: {evt.error?.slice(0, 80)}</span>}
      {evt.type === 'stopping' && <span className="feed-event-msg">Stopping experiment...</span>}
      {evt.type === 'evo_start' && <span className="feed-event-msg">Evolution started: {evt.experiment_id?.slice(0, 8)} — "{evt.hypothesis?.slice(0, 60)}"</span>}
      {evt.type === 'evo_gen' && (
        <span className="feed-event-msg">
          {evt._runStart && <div style={{ color: 'var(--text-muted)' }}>Evolution run {(evt.experiment_id || '').slice(0, 8) || 'unknown'}</div>}
          {(evt._missingPrefixTo || evt._missingGapTo) && (
            <div style={{ color: 'var(--accent-yellow)' }}>
              {evt._missingPrefixTo
                ? `Generations ${evt._missingPrefixFrom}-${evt._missingPrefixTo} are not in current feed history.`
                : `Generations ${evt._missingGapFrom}-${evt._missingGapTo} are not in current feed history.`}
            </div>
          )}
          <div>Gen {evt.generation}/{evt.total_generations}: best={evt.best_fitness?.toFixed(3)}, avg={evt.avg_fitness?.toFixed(3)}, pop={evt.population_size}</div>
          {evt.n_routing != null && <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Routing: {evt.n_routing} | Standard: {evt.n_standard}</div>}
        </span>
      )}
      {evt.type === 'evo_complete' && <span className="feed-event-msg feed-success">Evolution {evt.experiment_id?.slice(0, 8)} completed!</span>}
      {evt.type === 'nov_start' && <span className="feed-event-msg">Novelty search started: {evt.experiment_id?.slice(0, 8)} — "{evt.hypothesis?.slice(0, 60)}"</span>}
      {evt.type === 'nov_gen' && (
        <span className="feed-event-msg">
          {evt._runStart && <div style={{ color: 'var(--text-muted)' }}>Novelty run {(evt.experiment_id || '').slice(0, 8) || 'unknown'}</div>}
          {(evt._missingPrefixTo || evt._missingGapTo) && (
            <div style={{ color: 'var(--accent-yellow)' }}>
              {evt._missingPrefixTo
                ? `Generations ${evt._missingPrefixFrom}-${evt._missingPrefixTo} are not in current feed history.`
                : `Generations ${evt._missingGapFrom}-${evt._missingGapTo} are not in current feed history.`}
            </div>
          )}
          <div>Gen {evt.generation}/{evt.total_generations}: best_fit={evt.best_fitness?.toFixed(3)}, archive={evt.archive_size}, novelty={evt.best_novelty?.toFixed(3)}</div>
          {evt.n_routing != null && <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Routing: {evt.n_routing} | Standard: {evt.n_standard}</div>}
        </span>
      )}
      {evt.type === 'nov_complete' && <span className="feed-event-msg feed-success">Novelty search {evt.experiment_id?.slice(0, 8)} completed! Archive: {evt.archive_size} behaviors</span>}
      {evt.type === 'limit_reached' && (
        <span className="feed-event-msg" style={{ color: 'var(--accent-yellow)' }}>
          Session ended: {evt.reason}
          {evt.experiments_completed && ` (${evt.experiments_completed} experiments`}
          {evt.estimated_cost > 0 && `, $${evt.estimated_cost.toFixed(2)} spent`}
          {evt.experiments_completed && ')'}
        </span>
      )}
      {evt.type === 'scaleup_start' && <span className="feed-event-msg">Scale-up started: {evt.experiment_id?.slice(0, 8)} — {evt.result_ids?.length} program(s), {evt.config?.steps} steps</span>}
      {evt.type === 'confirm_start' && (
        <span className="feed-event-msg" style={{ color: 'var(--score-elite)' }}>
          Champion confirmation started: {evt.experiment_id?.slice(0, 8)} — {evt.result_ids?.length} candidate(s), {evt.config?.steps} steps
        </span>
      )}
      {evt.type === 'scaleup_progress' && (
        <span className="feed-event-msg">
          Scale-up {evt.current_program}/{evt.total_programs}: {evt.source_result_id?.slice(0, 8)} — {evt.status}
          {evt.passed != null && (
            <span style={{ color: evt.passed ? 'var(--accent-green)' : 'var(--accent-red)', marginLeft: 4 }}>
              {evt.passed ? 'PASS' : 'FAIL'}
            </span>
          )}
          {evt.loss_ratio != null && <span style={{ marginLeft: 4 }}>L:{evt.loss_ratio}</span>}
        </span>
      )}
      {evt.type === 'scaleup_complete' && <span className="feed-event-msg feed-success">Scale-up {evt.experiment_id?.slice(0, 8)} completed!</span>}
      {evt.type === 'auto_scaleup' && <span className="feed-event-msg" style={{ color: 'var(--accent-green)' }}>Auto-scale-up queued: {evt.n_programs} program(s) — {evt.reason}</span>}
      {evt.type === 'mode_selected' && <span className="feed-event-msg" style={{ color: 'var(--accent-purple)' }}>Aria selected mode: <strong>{evt.mode}</strong> — {evt.reasoning}</span>}
      {evt.type === 'invest_start' && <span className="feed-event-msg" style={{ color: 'var(--accent-yellow)' }}>Investigation started: {evt.experiment_id?.slice(0, 8)} — {evt.n_candidates} candidate(s)</span>}
      {evt.type === 'invest_progress' && (() => {
        const candidateCurrent = evt.current_candidate ?? evt.current;
        const candidateTotal = evt.total_candidates ?? evt.total;
        const sourceId = evt.result_id || evt.source_result_id;
        const programCurrent = evt.current_program ?? evt.training_program;
        const programTotal = evt.total_programs;
        return (
          <span className="feed-event-msg">
            Investigating {candidateCurrent ?? '?'} / {candidateTotal ?? '?'}: {sourceId?.slice(0, 8) || 'unknown'}
            {(programCurrent != null || programTotal != null) && <> — program {programCurrent ?? '?'} / {programTotal ?? '?'}</>}
            {evt.loss_ratio != null && <span style={{ marginLeft: 4 }}>L:{evt.loss_ratio}</span>}
          </span>
        );
      })()}
      {evt.type === 'invest_train_complete' && <span className="feed-event-msg" style={{ color: 'var(--accent-yellow)' }}>Investigation {evt.experiment_id?.slice(0, 8)} training complete; finalizing benchmark/probe writes</span>}
      {evt.type === 'invest_complete' && <span className="feed-event-msg feed-success">Investigation {evt.experiment_id?.slice(0, 8)} completed!{evt.n_passed != null && ` ${evt.n_passed} passed`}</span>}
      {evt.type === 'invest_failed' && (
        <span className="feed-event-msg feed-fail">
          Investigation {evt.experiment_id?.slice(0, 8)} failed!
          {evt.error ? ` ${evt.error}` : ''}
        </span>
      )}
      {evt.type === 'validate_start' && <span className="feed-event-msg" style={{ color: 'var(--accent-purple)' }}>Validation started: {evt.experiment_id?.slice(0, 8)} — {evt.n_candidates || evt.result_ids?.length || '?'} candidate(s)</span>}
      {evt.type === 'validate_progress' && (
        <span className="feed-event-msg">
          Validating {evt.current}/{evt.total}: {(evt.source_result_id || '')?.slice(0, 8)} — seed {evt.seed}/{evt.total_seeds}
          {evt.loss_ratio != null && <span style={{ marginLeft: 4 }}>L:{evt.loss_ratio}</span>}
          {evt.status === 'starting' && ' starting...'}
        </span>
      )}
      {evt.type === 'validate_phase' && (
        <span className="feed-event-msg" style={{ color: 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontFamily: 'monospace', fontSize: 11 }}>{(evt.result_id || evt.experiment_id || '').slice(0, 8)}</span>
          <span>{evt.phase}</span>
          {(() => {
            const idx = evt.test_index ?? evt.outer_index;
            const total = evt.total_tests ?? evt.outer_total;
            if (idx == null || total == null) return null;
            return (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                <span style={{ display: 'inline-block', width: 60, height: 6, borderRadius: 3, background: 'rgba(139,92,246,0.15)', overflow: 'hidden' }}>
                  <span style={{ display: 'block', width: `${Math.round((idx / total) * 100)}%`, height: '100%', borderRadius: 3, background: 'var(--accent-purple)', transition: 'width 0.3s ease' }} />
                </span>
                <span style={{ fontSize: 10, fontFamily: 'monospace', opacity: 0.7 }}>{idx}/{total}</span>
              </span>
            );
          })()}
        </span>
      )}
      {evt.type === 'validate_complete' && (
        <span className="feed-event-msg feed-success">
          Validation {evt.experiment_id?.slice(0, 8)} completed!
          {evt.error ? <span style={{ color: 'var(--accent-red)' }}> Error: {evt.error}</span> : evt.summary ? <span style={{ color: 'var(--text-secondary)', marginLeft: 4 }}>{evt.summary.split('\n').pop()?.slice(0, 80)}</span> : null}
        </span>
      )}
      {evt.type === 'breakthrough' && <span className="feed-event-msg" style={{ color: 'var(--accent-green)', fontWeight: 600 }}>BREAKTHROUGH DETECTED: {evt.result_id?.slice(0, 8)} — baseline ratio: {evt.baseline_ratio?.toFixed(4)}</span>}
      {evt.type === 'auto_investigate' && <span className="feed-event-msg" style={{ color: 'var(--accent-yellow)' }}>Auto-investigation queued: {evt.n_candidates} candidate(s)</span>}
      {evt.type === 'auto_validate' && <span className="feed-event-msg" style={{ color: 'var(--accent-purple)' }}>Auto-validation queued: {evt.n_candidates} candidate(s)</span>}
      {evt.type === 'auto_report' && <span className="feed-event-msg" style={{ color: 'var(--accent-purple)' }}>Research report generated ({evt.reason}) — {evt.narrative_length} chars</span>}
      {evt.type === 'recommendation' && <span className="feed-event-msg" style={{ color: 'var(--accent-purple)' }}>Aria suggests: {evt.reasoning?.slice(0, 100)}{evt.confidence != null && ` (${(evt.confidence * 100).toFixed(0)}% confidence)`}</span>}
      {evt.type === 'hyp_recorded' && <span className="feed-event-msg" style={{ color: 'var(--accent-blue)' }}>Hypothesis: {evt.prediction?.slice(0, 80)}{evt.confidence != null && ` (${(evt.confidence * 100).toFixed(0)}% confidence)`}</span>}
      {evt.type === 'hyp_resolved' && (
        <span className="feed-event-msg" style={{ color: evt.status === 'confirmed' ? 'var(--accent-green)' : evt.status === 'refuted' ? 'var(--accent-red)' : 'var(--accent-yellow)' }}>
          Hypothesis {evt.status?.toUpperCase()}: {evt.evidence?.slice(0, 80)}
          {evt.confidence_after != null && ` (${(evt.confidence_after * 100).toFixed(0)}%)`}
        </span>
      )}
      {evt.type === 'decision' && (
        <span className="feed-event-msg" style={{ color: evt.decision_type === 'go' ? 'var(--accent-green)' : evt.decision_type === 'no_go' ? 'var(--accent-red)' : 'var(--accent-yellow)', fontWeight: 600, whiteSpace: 'normal', wordBreak: 'break-word' }}>
          {evt.decision_type?.toUpperCase().replace('_', '-')}: {evt.subject}
          {evt.rationale && ` — ${evt.rationale}`}
        </span>
      )}
      {evt.type === 'knowledge' && <span className="feed-event-msg" style={{ color: 'var(--accent-purple)' }}>Knowledge extracted: {evt.n_entries} insight(s){evt.categories && ` [${evt.categories.join(', ')}]`}</span>}
      {evt.type === 'campaign_created' && <span className="feed-event-msg" style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>New campaign: {evt.title}</span>}
      {evt.type === 'campaign_completed' && <span className="feed-event-msg" style={{ color: 'var(--accent-green)', fontWeight: 600 }}>Campaign completed: {evt.title}</span>}
      {evt.type === 'aria_phase' && <span className="feed-event-msg" style={{ color: 'var(--text-secondary)' }}>Aria phase: <strong>{evt.phase_label || evt.phase || 'running'}</strong>{evt.selected_mode && <> — mode {evt.selected_mode}</>}{evt.last_note && <> — {evt.last_note}</>}</span>}
      {evt.type === 'learning' && <span className="feed-event-msg" style={{ color: 'var(--accent-orange, #f0883e)', fontWeight: 600 }}>{evt.description || 'Grammar weights adjusted'}{evt.n_changed != null && ` (${evt.n_changed} categories)`}</span>}
      {evt.type === 'log' && (
        <span className="feed-event-msg" style={{ fontFamily: 'monospace', fontSize: 11, color: evt.level === 'WARNING' ? 'var(--accent-yellow)' : evt.level === 'ERROR' ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
          <span style={{ opacity: 0.5, marginRight: 4 }}>[{evt.logger}]</span>
          {evt.message}
        </span>
      )}
    </div>
  );
});

export default FeedItem;
