import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { useEventBus } from './useEventBus';
import { apiCall } from '../services/apiService';

const AriaDataContext = createContext(null);

/**
 * Provider that owns the shared analytics and dashboard endpoints.
 * Polls on a coordinated schedule and exposes an atomic snapshot
 * so all consumers see consistent data.
 */
export function AriaDataProvider({ apiBase, isRunning, children }) {
  const [learningTrajectory, setLearningTrajectory] = useState(null);
  const [leaderboardEntries, setLeaderboardEntries] = useState([]);
  const [leaderboardRaw, setLeaderboardRaw] = useState(null);
  const [mathFamilyCoverage, setMathFamilyCoverage] = useState(null);
  const [fingerprintDiagnostics, setFingerprintDiagnostics] = useState(null);
  const [summary, setSummary] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  
  // New centralized data
  const [dashboardData, setDashboardData] = useState(null);
  const [ariaCycle, setAriaCycle] = useState(null);
  const [healerTasks, setHealerTasks] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [programs, setPrograms] = useState([]);
  const [entries, setEntries] = useState([]);
  const [insights, setInsights] = useState([]);
  const [cycleHistory, setCycleHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const apiBaseRef = useRef(apiBase);
  apiBaseRef.current = apiBase;
  const inFlightRef = useRef(false);
  const abortRef = useRef(null);

  const fetchSharedData = useCallback(async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    const controller = new AbortController();
    abortRef.current = controller;
    
    try {
      const [ltRes, lbRes, mcRes, fpRes, dashRes, cycleRes, historyRes] = await Promise.all([
        apiCall(`/api/analytics/learning-trajectory`, { signal: controller.signal }),
        apiCall(`/api/leaderboard?sort=composite_score&limit=80&quality=promotable&include_references=0&compact=1`, { signal: controller.signal }),
        apiCall(`/api/analytics/math-family-coverage`, { signal: controller.signal }),
        apiCall(`/api/diagnostics/fingerprint`, { signal: controller.signal }),
        apiCall(`/api/dashboard/summary`, { signal: controller.signal }),
        apiCall(`/api/aria/cycle-status`, { signal: controller.signal }).catch(() => ({ ok: false })),
        apiCall(`/api/aria/cycle-history?n=60&compact=1`, { signal: controller.signal }).catch(() => ({ ok: false })),
      ]);

      const ltData = ltRes.ok ? await ltRes.json() : null;
      const lbData = lbRes.ok ? await lbRes.json() : null;
      const mcData = mcRes.ok ? await mcRes.json() : null;
      const fpData = fpRes.ok ? await fpRes.json() : null;
      const dashData = dashRes.ok ? await dashRes.json() : null;
      const cycleData = cycleRes.ok ? await cycleRes.json() : null;
      const historyData = historyRes.ok ? await historyRes.json() : [];

      // Atomic state updates
      setLearningTrajectory(ltData);
      if (lbData) {
        setLeaderboardRaw(lbData);
        setLeaderboardEntries(lbData.entries || []);
      }
      setMathFamilyCoverage(mcData);
      setFingerprintDiagnostics(fpData?.sensitivity_skips || null);
      if (dashData) {
        setDashboardData(dashData);
        if (dashData.summary) setSummary(dashData.summary);
      }
      if (cycleData && !cycleData.error) setAriaCycle(cycleData);
      if (Array.isArray(historyData)) setCycleHistory(historyData);
      
      setLastUpdated(Date.now());
      setError(null);
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message);
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
      inFlightRef.current = false;
      setLoading(false);
    }
  }, []);

  // Specialized fetchers for tab data (less frequent)
  const fetchTabData = useCallback(async (tab) => {
    const endpoints = {
      experiments: `/api/experiments?n=200`,
      programs: '/api/programs?n=50&sort=novelty_score',
      entries: '/api/entries?n=50',
      insights: '/api/insights',
    };
    
    const endpoint = endpoints[tab];
    if (!endpoint) return;

    try {
      const res = await apiCall(endpoint);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      const data = Array.isArray(json) ? json : [];
      
      if (tab === 'experiments') setExperiments(data);
      if (tab === 'programs') setPrograms(data);
      if (tab === 'entries') setEntries(data);
      if (tab === 'insights') setInsights(data);
    } catch (err) {
      console.error(`Failed to fetch ${tab} data:`, err);
    }
  }, []);

  // Initial fetch on mount
  useEffect(() => {
    fetchSharedData();
  }, [fetchSharedData]);

  // Poll at adaptive interval
  useEffect(() => {
    const interval = setInterval(fetchSharedData, isRunning ? 3000 : 10000);
    return () => clearInterval(interval);
  }, [fetchSharedData, isRunning]);

  useEffect(() => () => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  }, []);

  const sseTimersRef = useRef([]);
  const debouncedRefresh = useCallback(() => {
    sseTimersRef.current.push(setTimeout(fetchSharedData, 2000));
  }, [fetchSharedData]);

  useEventBus('experiment_completed', debouncedRefresh);
  useEventBus('aria_cycle_completed', debouncedRefresh);

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
      dashboardData,
      ariaCycle,
      healerTasks,
      experiments,
      programs,
      entries,
      insights,
      loading,
      initialLoading: loading && !dashboardData,
      error,
      lastUpdated,
      refreshSharedData: fetchSharedData,
      fetchTabData,
    }}>
      {children}
    </AriaDataContext.Provider>
  );
}

export function useAriaData() {
  return useContext(AriaDataContext);
}

export default AriaDataContext;
