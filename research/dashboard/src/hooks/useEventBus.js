import { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';

const EventBusContext = createContext(null);

/**
 * Provider that opens a single EventSource to /api/events and dispatches
 * events to all subscribers. Mount once at the app root.
 */
export function EventBusProvider({ apiBase, children }) {
  const listenersRef = useRef({}); // { eventName: Set<callback> }
  const [connected, setConnected] = useState(false);
  const esRef = useRef(null);

  useEffect(() => {
    const es = new EventSource(`${apiBase}/api/events`);
    esRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    // Generic message handler — route named events to subscribers
    const knownEvents = [
      'program_evaluated', 'experiment_started', 'experiment_completed',
      'experiment_failed', 'experiment_stopping', 'evolution_started',
      'evolution_generation', 'evolution_completed', 'novelty_started',
      'novelty_generation', 'novelty_completed',
      'scale_up_started', 'scale_up_progress', 'scale_up_completed',
      'mode_selected', 'investigation_started', 'investigation_progress',
      'investigation_completed', 'validation_started', 'validation_progress',
      'validation_completed', 'hypothesis_generated', 'hypothesis_recorded',
      'hypothesis_resolved', 'decision_made', 'decision_recorded',
      'knowledge_updated', 'knowledge_extracted', 'campaign_updated',
      'campaign_created', 'campaign_completed', 'learning_event',
      'continuous_limit_reached', 'aria_cycle_completed',
      'auto_scale_up_queued', 'auto_investigate_queued',
      'auto_validate_queued', 'auto_report_generated',
      'aria_recommendation', 'breakthrough_detected',
    ];

    const handler = (eventName) => (e) => {
      const callbacks = listenersRef.current[eventName];
      if (!callbacks) return;
      let data;
      try { data = JSON.parse(e.data); } catch { data = {}; }
      callbacks.forEach(cb => cb(data, e));
    };

    const cleanups = knownEvents.map(name => {
      const h = handler(name);
      es.addEventListener(name, h);
      return () => es.removeEventListener(name, h);
    });

    return () => {
      cleanups.forEach(fn => fn());
      es.close();
      setConnected(false);
    };
  }, [apiBase]);

  const subscribe = useCallback((eventName, callback) => {
    if (!listenersRef.current[eventName]) {
      listenersRef.current[eventName] = new Set();
    }
    listenersRef.current[eventName].add(callback);
    return () => {
      listenersRef.current[eventName]?.delete(callback);
    };
  }, []);

  return (
    <EventBusContext.Provider value={{ subscribe, connected }}>
      {children}
    </EventBusContext.Provider>
  );
}

/**
 * Subscribe to a named SSE event. Callback receives (data, rawEvent).
 * Returns { connected } status.
 */
export function useEventBus(eventName, callback) {
  const ctx = useContext(EventBusContext);
  const cbRef = useRef(callback);
  cbRef.current = callback;

  useEffect(() => {
    if (!ctx) return;
    const stable = (data, e) => cbRef.current(data, e);
    return ctx.subscribe(eventName, stable);
  }, [ctx, eventName]);

  return { connected: ctx?.connected ?? false };
}

export default EventBusContext;
