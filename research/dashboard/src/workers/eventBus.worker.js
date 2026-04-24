/**
 * Web Worker for handling SSE events from /api/events.
 * Offloads JSON parsing and event dispatching from the main UI thread.
 */

/* eslint-disable no-restricted-globals */

let eventSource = null;
let trainingStepFlushTimer = null;
let pendingTrainingStep = null;

function postEvent(eventName, data) {
  self.postMessage({
    type: 'event',
    eventName,
    data,
  });
}

function flushPendingTrainingStep() {
  trainingStepFlushTimer = null;
  if (pendingTrainingStep === null) {
    return;
  }
  postEvent('training_step', pendingTrainingStep);
  pendingTrainingStep = null;
}

function clearBufferedEvents() {
  pendingTrainingStep = null;
  if (trainingStepFlushTimer !== null) {
    clearTimeout(trainingStepFlushTimer);
    trainingStepFlushTimer = null;
  }
}

function resolveEventsUrl(payload) {
  const eventsUrl = String(payload?.eventsUrl || '').trim();
  if (eventsUrl) return eventsUrl;
  const apiBase = String(payload?.apiBase || '').trim();
  if (!apiBase) return '/api/events';
  return `${apiBase.replace(/\/+$/, '')}/api/events`;
}

self.onmessage = function(e) {
  const { type } = e.data || {};

  if (type === 'connect') {
    if (eventSource) {
      eventSource.close();
    }
    const eventsUrl = resolveEventsUrl(e.data || {});
    eventSource = new EventSource(eventsUrl);

    eventSource.onopen = () => {
      self.postMessage({ type: 'status', connected: true });
    };

    eventSource.onerror = () => {
      self.postMessage({ type: 'status', connected: false });
    };

    const knownEvents = [
      'program_evaluated', 'experiment_started', 'experiment_completed',
      'experiment_failed', 'experiment_stopping', 'evolution_started',
      'evolution_generation', 'evolution_completed', 'novelty_started',
      'novelty_generation', 'novelty_completed',
      'scale_up_started', 'scale_up_progress', 'scale_up_completed',
      'mode_selected', 'investigation_started', 'investigation_progress',
      'investigation_completed', 'investigation_failed',
      'validation_started', 'validation_progress',
      'validation_completed', 'hypothesis_generated', 'hypothesis_recorded',
      'hypothesis_resolved', 'decision_made', 'decision_recorded',
      'knowledge_updated', 'knowledge_extracted', 'campaign_updated',
      'campaign_created', 'campaign_completed', 'learning_event',
      'continuous_limit_reached', 'aria_cycle_phase', 'aria_cycle_completed',
      'training_step',
      'auto_scale_up_queued', 'auto_investigate_queued',
      'auto_validate_queued', 'auto_report_generated',
      'aria_recommendation', 'breakthrough_detected',
      'log_message', 'validation_phase',
    ];

    knownEvents.forEach(eventName => {
      eventSource.addEventListener(eventName, (event) => {
        let parsedData = {};
        try {
          parsedData = JSON.parse(event.data);
        } catch (err) {
          console.error(`Worker failed to parse ${eventName}:`, err);
        }

        if (eventName === 'training_step') {
          pendingTrainingStep = parsedData;
          if (trainingStepFlushTimer === null) {
            trainingStepFlushTimer = setTimeout(flushPendingTrainingStep, 250);
          }
          return;
        }

        postEvent(eventName, parsedData);
      });
    });
  }

  if (type === 'disconnect') {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    clearBufferedEvents();
    self.postMessage({ type: 'status', connected: false });
  }
};
