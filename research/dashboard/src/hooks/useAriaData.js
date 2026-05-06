import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { useEventBus } from './useEventBus';
import useDocumentVisible from './useDocumentVisible';
import { apiCall } from '../services/apiService';

const AriaDataContext = createContext(null);
const ANALYTICS_STALE_MS = 15000;
const SLOW_TICK_DIVISOR = 3; // slowPollTick increments every Nth core poll (~9-30s)
const CORE_RETRY_BASE_MS = 15000;
const CORE_RETRY_MAX_MS = 120000;

/**
 * Provider that owns the shared analytics and dashboard endpoints.
 * Polls on a coordinated schedule and exposes an atomic snapshot
 * so all consumers see consistent data.
 */
export function AriaDataProvider({ apiBase, isRunning, autoRefreshEnabled = true, children }) {
  const [learningTrajectory, setLearningTrajectory] = useState(null);
  const [leaderboardEntries, setLeaderboardEntries] = useState([]);
  const [mathFamilyCoverage, setMathFamilyCoverage] = useState(null);
  const [fingerprintDiagnostics, setFingerprintDiagnostics] = useState(null);
  const [summary, setSummary] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  
  // New centralized data
  const [dashboardData, setDashboardData] = useState(null);
  const [ariaCycle, setAriaCycle] = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [programs, setPrograms] = useState([]);
  const [entries, setEntries] = useState([]);
  const [insights, setInsights] = useState([]);
  const [cycleHistory, setCycleHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [pollTick, setPollTick] = useState(0);
  const isDocumentVisible = useDocumentVisible();
  const pollCountRef = useRef(0);
  const [slowPollTick, setSlowPollTick] = useState(0);

  const apiBaseRef = useRef(apiBase);
  apiBaseRef.current = apiBase;
  const inFlightRef = useRef(false);
  const analyticsInFlightRef = useRef(false);
  const abortRef = useRef(null);
  const analyticsAbortRef = useRef(null);
  const analyticsLoadedAtRef = useRef(0);
  const coreRetryDelayRef = useRef(0);
  const coreRetryAfterRef = useRef(0);
  const fullDashboardModeRef = useRef(false);
  const sseRefreshTimeoutRef = useRef(null);

  const fetchCoreData = useCallback(async ({ force = false, includeFullDashboard = false } = {}) => {
    if (!force && (!autoRefreshEnabled || !isDocumentVisible)) return;
    if (!force && coreRetryAfterRef.current > Date.now()) return;
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const dashboardEndpoint = (includeFullDashboard || fullDashboardModeRef.current)
        ? `/api/dashboard`
        : `/api/dashboard/summary`;
      const [dashRes, cycleRes, historyRes] = await Promise.all([
        apiCall(dashboardEndpoint, { signal: controller.signal }),
        apiCall(`/api/aria/cycle-status`, { signal: controller.signal }).catch(() => ({ ok: false })),
        apiCall(`/api/aria/cycle-history?n=60&compact=1`, { signal: controller.signal }).catch(() => ({ ok: false })),
      ]);

      const dashData = dashRes.ok ? await dashRes.json() : null;
      const cycleData = cycleRes.ok ? await cycleRes.json() : null;
      const historyData = historyRes.ok ? await historyRes.json() : [];

      if (dashData) {
        setDashboardData(dashData);
        if (dashData.summary) setSummary(dashData.summary);
      }
      if (cycleData && !cycleData.error) setAriaCycle(cycleData);
      if (Array.isArray(historyData)) setCycleHistory(historyData);

      coreRetryDelayRef.current = 0;
      coreRetryAfterRef.current = 0;
      setLastUpdated(Date.now());
      setError(null);
      // Increment coordinated poll ticks so subscribers refresh in sync
      setPollTick(t => t + 1);
      pollCountRef.current += 1;
      if (pollCountRef.current % SLOW_TICK_DIVISOR === 0) {
        setSlowPollTick(t => t + 1);
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        const nextDelay = Math.min(
          coreRetryDelayRef.current ? coreRetryDelayRef.current * 2 : CORE_RETRY_BASE_MS,
          CORE_RETRY_MAX_MS,
        );
        coreRetryDelayRef.current = nextDelay;
        coreRetryAfterRef.current = Date.now() + nextDelay;
        setError(err.message);
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
      inFlightRef.current = false;
      setLoading(false);
    }
  }, [autoRefreshEnabled, isDocumentVisible]);

  const fetchAnalyticsData = useCallback(async ({ force = false } = {}) => {
    const now = Date.now();
    if (!force && analyticsLoadedAtRef.current && now - analyticsLoadedAtRef.current < ANALYTICS_STALE_MS) {
      return;
    }
    if (analyticsInFlightRef.current) return;
    analyticsInFlightRef.current = true;
    const controller = new AbortController();
    analyticsAbortRef.current = controller;

    try {
      const [ltRes, lbRes, mcRes, fpRes] = await Promise.all([
        apiCall(`/api/analytics/learning-trajectory`, { signal: controller.signal }),
        apiCall(`/api/leaderboard?sort=composite_score&limit=80&quality=promotable&include_references=0&compact=1&trusted_only=0`, { signal: controller.signal }),
        apiCall(`/api/analytics/math-family-coverage`, { signal: controller.signal }),
        apiCall(`/api/diagnostics/fingerprint`, { signal: controller.signal }),
      ]);

      const ltData = ltRes.ok ? await ltRes.json() : null;
      const lbData = lbRes.ok ? await lbRes.json() : null;
      const mcData = mcRes.ok ? await mcRes.json() : null;
      const fpData = fpRes.ok ? await fpRes.json() : null;

      setLearningTrajectory(ltData);
      if (lbData) {
        setLeaderboardEntries(lbData.entries || []);
      }
      setMathFamilyCoverage(mcData);
      setFingerprintDiagnostics(fpData?.sensitivity_skips || null);
      analyticsLoadedAtRef.current = Date.now();
      setError(null);
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message);
      }
    } finally {
      if (analyticsAbortRef.current === controller) {
        analyticsAbortRef.current = null;
      }
      analyticsInFlightRef.current = false;
    }
  }, []);

  // Specialized fetchers for tab data (less frequent), with per-tab dedup
  const tabInFlightRef = useRef(new Set());
  const tabCacheRef = useRef({});
  const TAB_CACHE_TTL_MS = 5000;

  const fetchTabData = useCallback(async (tab) => {
    const endpoints = {
      experiments: `/api/experiments?n=200`,
      programs: '/api/programs?n=50&sort=novelty_score',
      entries: '/api/entries?n=50',
      insights: '/api/insights',
    };

    const endpoint = endpoints[tab];
    if (!endpoint) return;

    // Skip if this tab is already being fetched
    if (tabInFlightRef.current.has(tab)) return;

    // Return cached data if still fresh
    const cached = tabCacheRef.current[tab];
    if (cached && Date.now() - cached.ts < TAB_CACHE_TTL_MS) return;

    tabInFlightRef.current.add(tab);
    try {
      const res = await apiCall(endpoint);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      const data = Array.isArray(json) ? json : [];

      tabCacheRef.current[tab] = { ts: Date.now() };
      if (tab === 'experiments') setExperiments(data);
      if (tab === 'programs') setPrograms(data);
      if (tab === 'entries') setEntries(data);
      if (tab === 'insights') setInsights(data);
    } catch (err) {
      console.error(`Failed to fetch ${tab} data:`, err);
    } finally {
      tabInFlightRef.current.delete(tab);
    }
  }, []);

  // Initial fetch on mount
  useEffect(() => {
    fetchCoreData({ force: true });
  }, [fetchCoreData]);

  // Poll at adaptive interval only when auto-refresh is enabled and the tab is visible.
  useEffect(() => {
    if (!autoRefreshEnabled || !isDocumentVisible) return undefined;
    const interval = setInterval(fetchCoreData, isRunning ? 3000 : 10000);
    return () => clearInterval(interval);
  }, [autoRefreshEnabled, fetchCoreData, isDocumentVisible, isRunning]);

  useEffect(() => () => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    if (analyticsAbortRef.current) {
      analyticsAbortRef.current.abort();
      analyticsAbortRef.current = null;
    }
    if (sseRefreshTimeoutRef.current) {
      clearTimeout(sseRefreshTimeoutRef.current);
      sseRefreshTimeoutRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!isDocumentVisible || !autoRefreshEnabled) {
      return;
    }
    coreRetryAfterRef.current = 0;
    fetchCoreData({ force: true });
  }, [autoRefreshEnabled, fetchCoreData, isDocumentVisible]);

  useEffect(() => {
    if (typeof window === 'undefined') return undefined;
    const resumePolling = () => {
      if (!autoRefreshEnabled) return;
      coreRetryAfterRef.current = 0;
      fetchCoreData({ force: true });
    };
    window.addEventListener('online', resumePolling);
    window.addEventListener('focus', resumePolling);
    return () => {
      window.removeEventListener('online', resumePolling);
      window.removeEventListener('focus', resumePolling);
    };
  }, [autoRefreshEnabled, fetchCoreData]);

  const invalidateTabCache = useCallback((tab) => {
    if (tab) {
      delete tabCacheRef.current[tab];
    } else {
      tabCacheRef.current = {};
    }
  }, []);

  const setDashboardDetailMode = useCallback((enabled) => {
    fullDashboardModeRef.current = Boolean(enabled);
  }, []);

  const debouncedRefresh = useCallback(() => {
    if (!autoRefreshEnabled || !isDocumentVisible) return;
    if (sseRefreshTimeoutRef.current) {
      clearTimeout(sseRefreshTimeoutRef.current);
    }
    sseRefreshTimeoutRef.current = setTimeout(() => {
      sseRefreshTimeoutRef.current = null;
      fetchCoreData();
    }, 2000);
  }, [autoRefreshEnabled, fetchCoreData, isDocumentVisible]);

  const onExperimentCompleted = useCallback(() => {
    invalidateTabCache('experiments');
    debouncedRefresh();
  }, [invalidateTabCache, debouncedRefresh]);

  useEventBus('experiment_completed', onExperimentCompleted);
  useEventBus('aria_cycle_completed', debouncedRefresh);

  return (
    <AriaDataContext.Provider value={{
      learningTrajectory,
      leaderboardEntries,
      mathFamilyCoverage,
      fingerprintDiagnostics,
      summary,
      dashboardData,
      ariaCycle,
      experiments,
      programs,
      entries,
      insights,
      loading,
      initialLoading: loading && !dashboardData,
      error,
      lastUpdated,
      refreshSharedData: fetchCoreData,
      setDashboardDetailMode,
      refreshAnalyticsData: fetchAnalyticsData,
      fetchTabData,
      invalidateTabCache,
      pollTick,
      slowPollTick,
    }}>
      {children}
    </AriaDataContext.Provider>
  );
}

export function useAriaData() {
  return useContext(AriaDataContext);
}

export default AriaDataContext;
