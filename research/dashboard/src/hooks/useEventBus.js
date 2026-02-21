import { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';

const EventBusContext = createContext(null);

function resolveApiBase(apiBase) {
  const configured = String(apiBase || '').trim();
  if (configured) return configured.replace(/\/+$/, '');

  if (typeof window === 'undefined') return '';
  const { protocol, hostname, port } = window.location;
  // When running dashboard via CRA dev server (localhost:3000) and no API base is set,
  // default SSE/API traffic to the Flask backend on :5000.
  if (port === '3000' || port === '3001' || port === '3002') {
    return `${protocol}//${hostname}:5000`;
  }
  return '';
}

/**
 * Provider that offloads SSE handling to a Web Worker and dispatches
 * events to all subscribers. Ensures the main thread stays responsive.
 */
export function EventBusProvider({ apiBase, children }) {
  const listenersRef = useRef({}); // { eventName: Set<callback> }
  const [connected, setConnected] = useState(false);
  const workerRef = useRef(null);
  const resolvedApiBase = resolveApiBase(apiBase);
  const eventsUrl = `${resolvedApiBase}/api/events`;

  useEffect(() => {
    let worker;
    try {
      // Create worker using CRA 5.0 compatible syntax
      worker = new Worker(new URL('../workers/eventBus.worker.js', import.meta.url));
      workerRef.current = worker;

      worker.onmessage = (e) => {
        const { type, eventName, data, connected: workerConnected } = e.data;

        if (type === 'status') {
          setConnected(workerConnected);
        } else if (type === 'event') {
          const callbacks = listenersRef.current[eventName];
          if (callbacks) {
            callbacks.forEach(cb => cb(data));
          }
        }
      };

      worker.postMessage({ type: 'connect', apiBase: resolvedApiBase, eventsUrl });
    } catch (_err) {
      // If worker creation fails (browser/build edge cases), stay gracefully disconnected
      // instead of crashing the dashboard.
      setConnected(false);
    }

    return () => {
      if (worker) {
        worker.postMessage({ type: 'disconnect' });
        worker.terminate();
      }
      setConnected(false);
    };
  }, [resolvedApiBase, eventsUrl]);

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
    if (!ctx || !eventName || typeof callback !== 'function') return;
    const stable = (data, e) => cbRef.current(data, e);
    return ctx.subscribe(eventName, stable);
  }, [ctx, eventName, callback]);

  return {
    connected: ctx?.connected ?? false,
    subscribe: ctx?.subscribe,
  };
}

export default EventBusContext;
