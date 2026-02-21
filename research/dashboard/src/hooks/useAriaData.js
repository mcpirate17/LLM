import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { useEventBus } from './useEventBus';

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

  const fetchSharedData = useCallback(async () => {
    const base = apiBaseRef.current;
    try {
      const [ltRes, lbRes, mcRes, fpRes] = await Promise.all([
        fetch(`${base}/api/analytics/learning-trajectory`),
        fetch(`${base}/api/leaderboard?sort=composite_score&limit=300`),
        fetch(`${base}/api/analytics/math-family-coverage`),
        fetch(`${base}/api/diagnostics/fingerprint`),
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

  // SSE-driven refresh on experiment/cycle completion
  useEventBus('experiment_completed', useCallback(() => {
    setTimeout(fetchSharedData, 2000);
  }, [fetchSharedData]));

  useEventBus('aria_cycle_completed', useCallback(() => {
    setTimeout(fetchSharedData, 2000);
  }, [fetchSharedData]));

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
