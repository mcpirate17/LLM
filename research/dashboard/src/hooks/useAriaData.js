import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { useEventBus } from './useEventBus';
import { apiCall } from '../services/apiService';

const AriaDataContext = createContext(null);

/**
 * Provider that owns the 4 shared analytics endpoints.
 * Polls on a coordinated schedule and exposes an atomic snapshot
 * so all consumers see consistent data.
 *
 * Shared endpoints:
 *   /api/analytics/learning-trajectory
 *   /api/leaderboard?sort=composite_score&limit=300
 *   /api/analytics/math-family-coverage
 *   /api/diagnostics/fingerprint
 */
export function AriaDataProvider({ apiBase, isRunning, children }) {
  const [learningTrajectory, setLearningTrajectory] = useState(null);
  const [leaderboardEntries, setLeaderboardEntries] = useState([]);
  const [leaderboardRaw, setLeaderboardRaw] = useState(null);
  const [mathFamilyCoverage, setMathFamilyCoverage] = useState(null);
  const [fingerprintDiagnostics, setFingerprintDiagnostics] = useState(null);
  const [summary, setSummary] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const apiBaseRef = useRef(apiBase);
  apiBaseRef.current = apiBase;
  const inFlightRef = useRef(false);
  const abortRef = useRef(null);

  const fetchSharedData = useCallback(async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    const base = apiBaseRef.current;
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const [ltRes, lbRes, mcRes, fpRes] = await Promise.all([
        apiCall(`/api/analytics/learning-trajectory`, { signal: controller.signal }),
        apiCall(`/api/leaderboard?sort=composite_score&limit=300`, { signal: controller.signal }),
        apiCall(`/api/analytics/math-family-coverage`, { signal: controller.signal }),
        apiCall(`/api/diagnostics/fingerprint`, { signal: controller.signal }),
      ]);

      // Parse all responses (tolerant of individual failures)
      const ltData = ltRes.ok ? await ltRes.json() : null;
      const lbData = lbRes.ok ? await lbRes.json() : null;
      const mcData = mcRes.ok ? await mcRes.json() : null;
      const fpData = fpRes.ok ? await fpRes.json() : null;

      // Atomic state update — all consumers see a consistent snapshot
      setLearningTrajectory(ltData);
      if (lbData) {
        setLeaderboardRaw(lbData);
        setLeaderboardEntries(lbData.entries || []);
      }
      setMathFamilyCoverage(mcData);
      setFingerprintDiagnostics(fpData?.sensitivity_skips || null);
      setLastUpdated(Date.now());
    } catch {
      // Silently fail — keep stale data rather than clearing
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
      inFlightRef.current = false;
    }
  }, []);

  // Initial fetch on mount
  useEffect(() => {
    fetchSharedData();
  }, [fetchSharedData]);

  // Poll at adaptive interval
  useEffect(() => {
    const interval = setInterval(fetchSharedData, isRunning ? 5000 : 10000);
    return () => clearInterval(interval);
  }, [fetchSharedData, isRunning]);

  useEffect(() => () => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  }, []);

  // SSE-driven refresh on experiment/cycle completion
  const sseTimersRef = useRef([]);

  useEventBus('experiment_completed', useCallback(() => {
    sseTimersRef.current.push(setTimeout(fetchSharedData, 2000));
  }, [fetchSharedData]));

  useEventBus('aria_cycle_completed', useCallback(() => {
    sseTimersRef.current.push(setTimeout(fetchSharedData, 2000));
  }, [fetchSharedData]));

  // Cancel SSE-driven timers on unmount
  useEffect(() => () => {
    sseTimersRef.current.forEach(clearTimeout);
    sseTimersRef.current = [];
  }, []);

  return (
    <AriaDataContext.Provider value={{
      learningTrajectory,
      leaderboardEntries,
      leaderboardRaw,
      mathFamilyCoverage,
      fingerprintDiagnostics,
      summary,
      setSummary,
      lastUpdated,
      refreshSharedData: fetchSharedData,
    }}>
      {children}
    </AriaDataContext.Provider>
  );
}

export function useAriaData() {
  return useContext(AriaDataContext);
}

export default AriaDataContext;
