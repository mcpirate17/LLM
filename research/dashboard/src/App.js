import React, { useState, useEffect, useCallback, useMemo, useRef, Suspense, startTransition } from 'react';
import {
  ErrorBoundary,
  LazyFallback,
} from './components/app/AppShellShared';
import {
  ChatDrawer,
  DesignerDrawerOverlay,
  HelpOverlay,
  ProgramDetailOverlay,
  SettingsOverlay,
} from './components/app/AppOverlays';
import AppTabContent from './components/app/AppTabContent';
import {
  DashboardHeader,
  DashboardNav,
  DashboardStatusBanners,
} from './components/app/AppShellSections';
import { NAV_CATEGORIES } from './components/app/appConfig';
import {
  AriaChatPanel,
  ArchitectureDrawer,
  ControlPanel,
  ProgramDetail,
} from './components/app/lazyComponents';
import { EventBusProvider } from './hooks/useEventBus';
import { AriaDataProvider, useAriaData } from './hooks/useAriaData';
import { apiCall } from './services/apiService';
import useLocalStorage from './hooks/useLocalStorage';
import useInvestigationQueue from './hooks/useInvestigationQueue';
import useAutoRepair from './hooks/useAutoRepair';
import useDashboardActions from './hooks/useDashboardActions';
import useKeyboardShortcuts from './hooks/useKeyboardShortcuts';
import { buildEligibilityByResultId } from './utils/candidateState';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || '';
const DEFAULT_EXPERIMENTS_PAGE_SIZE = 200;
const OVERRIDE_INELIGIBLE_ALWAYS_KEY = 'aria_override_ineligible_always_v1';

function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  return (
    <EventBusProvider apiBase={API_BASE}>
      <AriaDataProvider apiBase={API_BASE} isRunning={isRunning} autoRefreshEnabled={autoRefresh}>
        <AppContent
          autoRefresh={autoRefresh}
          onAutoRefreshChange={setAutoRefresh}
          onRunningChange={setIsRunning}
        />
      </AriaDataProvider>
    </EventBusProvider>
  );
}

function AppContent({ autoRefresh, onAutoRefreshChange, onRunningChange }) {
  // Centralized data from AriaDataProvider
  const {
    learningTrajectory,
    leaderboardEntries,
    dashboardData: data,
    ariaCycle,
    experiments: centralizedExperiments,
    entries: centralizedEntries,
    insights: centralizedInsights,
    initialLoading,
    error,
    lastUpdated: dashboardUpdatedAt,
    refreshSharedData,
    setDashboardDetailMode,
    refreshAnalyticsData,
    fetchTabData,
    invalidateTabCache,
    pollTick,
    slowPollTick,
  } = useAriaData() || {};

  const fetchDashboard = refreshSharedData || (() => {});

  const [activeTab, _setActiveTab] = useState('command');
  const setActiveTab = useCallback((tab) => startTransition(() => _setActiveTab(tab)), []);
  const [showHelp, setShowHelp] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [reportsDeferredReady, setReportsDeferredReady] = useState(false);
  const [reportsCampaignsVisible, setReportsCampaignsVisible] = useState(false);
  const [reportsKnowledgeVisible, setReportsKnowledgeVisible] = useState(false);

  const [experimentsPageSize, setExperimentsPageSize] = useState(DEFAULT_EXPERIMENTS_PAGE_SIZE);
  const [experimentsHasMore, setExperimentsHasMore] = useState(true);
  const [experimentsLoadingMore, setExperimentsLoadingMore] = useState(false);

  // Local state for experiments pagination (optional, but keeping for now)
  const [paginatedExperiments, setPaginatedExperiments] = useState([]);

  useEffect(() => {
    const heavyTabs = new Set(['command', 'trends', 'discoveries', 'references']);
    if (!heavyTabs.has(activeTab) || typeof refreshAnalyticsData !== 'function') {
      return;
    }
    refreshAnalyticsData();
  }, [activeTab, refreshAnalyticsData]);

  useEffect(() => {
    if (typeof setDashboardDetailMode === 'function') {
      setDashboardDetailMode(activeTab === 'templates');
    }
    if (activeTab === 'templates' && typeof refreshSharedData === 'function') {
      refreshSharedData({ force: true, includeFullDashboard: true });
    }
  }, [activeTab, refreshSharedData, setDashboardDetailMode]);

  useEffect(() => {
    if (activeTab !== 'reports') {
      setReportsDeferredReady(false);
      return undefined;
    }

    let cancelled = false;
    let timeoutId = null;
    let idleId = null;
    const activate = () => {
      if (!cancelled) {
        setReportsDeferredReady(true);
      }
    };

    if (typeof window !== 'undefined' && typeof window.requestIdleCallback === 'function') {
      idleId = window.requestIdleCallback(activate, { timeout: 300 });
    } else {
      timeoutId = window.setTimeout(activate, 150);
    }

    return () => {
      cancelled = true;
      if (idleId !== null && typeof window !== 'undefined' && typeof window.cancelIdleCallback === 'function') {
        window.cancelIdleCallback(idleId);
      }
      if (timeoutId !== null && typeof window !== 'undefined') {
        window.clearTimeout(timeoutId);
      }
    };
  }, [activeTab]);

  // Drill-down state
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [selectedProgram, setSelectedProgram] = useState(null);

  // Action error state (replaces alert())
  const [actionError, setActionError] = useState(null);
  const [actionNotice, setActionNotice] = useState(null);
  const [blockedConfig, setBlockedConfig] = useState(null);
  const [overrideIneligibleAlways, setOverrideIneligibleAlways] = useLocalStorage(OVERRIDE_INELIGIBLE_ALWAYS_KEY, false);

  // Architecture designer drawer
  const [designerSession, setDesignerSession] = useState({
    open: false,
    resultId: null,
    readOnly: true,
  });
  const openDesignerBlank = useCallback(() => {
    setDesignerSession({ open: true, resultId: null, readOnly: false });
  }, []);
  const openDesignerForResult = useCallback((rid) => {
    setDesignerSession({ open: true, resultId: rid || null, readOnly: true });
  }, []);
  const closeDesigner = useCallback(() => {
    setDesignerSession({ open: false, resultId: null, readOnly: true });
  }, []);

  // Cross-view navigation state
  const [leaderboardHighlight, setLeaderboardHighlight] = useState(null);
  const [selectedCampaignId, setSelectedCampaignId] = useState(null);
  const [controlPanelPrefill, setControlPanelPrefill] = useState(null);

  useEffect(() => {
    if (activeTab !== 'reports') {
      return;
    }
    if (selectedCampaignId) {
      setReportsCampaignsVisible(true);
    }
  }, [activeTab, selectedCampaignId]);
  const [activeOverviewStrategy, setActiveOverviewStrategy] = useState(null);
  const [cycleControlBusy, setCycleControlBusy] = useState(false);
  const [allowAdvancedStartOverride, setAllowAdvancedStartOverride] = useState(false);
  const [autonomousMode, setAutonomousMode] = useState(false);
  const [comparisonList, setComparisonList] = useState([]);

  const handleAddToComparison = useCallback((resultId) => {
    if (!resultId) return;
    setComparisonList(prev => {
      if (prev.includes(resultId)) return prev;
      if (prev.length >= 5) {
        setActionError("Max 5 candidates for comparison.");
        return prev;
      }
      return [...prev, resultId];
    });
    setActiveTab('comparison');
  }, []);

  const handleRemoveFromComparison = useCallback((resultId) => {
    setComparisonList(prev => prev.filter(id => id !== resultId));
  }, []);


  // Sync isRunning up to App for AriaDataProvider polling speed
  useEffect(() => {
    if (onRunningChange) onRunningChange(Boolean(data?.is_running));
  }, [data?.is_running, onRunningChange]);

  useEffect(() => {
    if (
      ariaCycle
      && typeof ariaCycle.continuous_active === 'boolean'
    ) {
      const autonomousActive = Boolean(ariaCycle.continuous_active) && !Boolean(ariaCycle.cycle_paused);
      setAutonomousMode(autonomousActive);
    }
  }, [ariaCycle]);

  useEffect(() => {
    const experimentId = actionNotice?.clearOnExperimentId;
    if (!experimentId || !Array.isArray(centralizedExperiments)) {
      return;
    }
    const match = centralizedExperiments.find((exp) => exp?.experiment_id === experimentId);
    if (!match) {
      return;
    }
    const status = String(match.status || '').trim().toLowerCase();
    if (status && !['running', 'queued', 'starting', 'pending'].includes(status)) {
      setActionNotice(null);
    }
  }, [actionNotice, centralizedExperiments]);

  // Global keyboard shortcuts
  useKeyboardShortcuts({
    showHelp, setShowHelp,
    showChat, setShowChat,
    showSettings, setShowSettings,
    selectedProgram,
    closeSelectedProgram: () => setSelectedProgram(null),
    designerSession, closeDesigner,
    setActiveTab, setSelectedExperiment,
  });

  const eligibilityByResultId = useMemo(
    () => buildEligibilityByResultId(leaderboardEntries || []),
    [leaderboardEntries],
  );

  // Investigation queue hook
  const {
    investigationQueue,
    addToInvestigationQueue: handleQueueAdd,
    removeFromInvestigationQueue: handleQueueRemove,
    clearInvestigationQueue: handleQueueClear,
    queueBreakdown,
  } = useInvestigationQueue({ eligibilityByResultId, setActionError });

  // Auto-repair hook
  const {
    emitAutoRepairStarted,
  } = useAutoRepair({ pollTick });

  useEffect(() => {
    setAllowAdvancedStartOverride(false);
  }, [activeOverviewStrategy?.id]);

  // Use centralized tab data fetching with polling
  const tabDataKey = activeTab === 'experiments' ? 'experiments'
    : activeTab === 'discoveries' ? 'programs'
    : activeTab === 'log' ? 'entries'
    : activeTab === 'trends' ? 'insights'
    : null;

  useEffect(() => {
    if (tabDataKey) fetchTabData(tabDataKey);
  }, [tabDataKey, fetchTabData]);

  useEffect(() => {
    if (!tabDataKey) return;
    if (invalidateTabCache) invalidateTabCache(tabDataKey);
    fetchTabData(tabDataKey);
  }, [tabDataKey, slowPollTick, fetchTabData, invalidateTabCache]);

  const handleLoadMoreExperiments = useCallback(async () => {
    if (experimentsLoadingMore || !experimentsHasMore) return;
    const currentCount = paginatedExperiments.length;
    
    setExperimentsLoadingMore(true);
    try {
      const res = await apiCall(`/api/experiments?n=${experimentsPageSize}&offset=${currentCount}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      const page = Array.isArray(json) ? json : [];
      setPaginatedExperiments(prev => [...prev, ...page]);
      setExperimentsHasMore(page.length === experimentsPageSize);
    } catch (err) {
      setActionError('Failed to load more experiments: ' + err.message);
    } finally {
      setExperimentsLoadingMore(false);
    }
  }, [experimentsHasMore, experimentsLoadingMore, experimentsPageSize, paginatedExperiments.length]);

  useEffect(() => {
    if (centralizedExperiments) {
      setPaginatedExperiments(centralizedExperiments);
      setExperimentsHasMore(centralizedExperiments.length >= 200);
    }
  }, [centralizedExperiments]);

  const handleExperimentPageSizeChange = useCallback((nextSize) => {
    const parsed = Number(nextSize);
    if (!Number.isFinite(parsed) || parsed <= 0) return;
    setExperimentsPageSize(parsed);
    setExperimentsHasMore(true);
  }, []);

  // Compute per-tab delta indicators from latest vs previous experiment
  const tabDeltas = useMemo(() => {
    const d = data?.deltas;
    if (!d) return {};
    const deltas = {};
    if (d.programs !== 0) deltas.experiments = { text: d.programs > 0 ? `+${d.programs}` : `${d.programs}`, positive: d.programs > 0 };
    if (d.stage1 !== 0) {
      const entry = { text: d.stage1 > 0 ? `+${d.stage1} S1` : `${d.stage1} S1`, positive: d.stage1 > 0 };
      deltas.discoveries = entry;
    }
    if (d.best_loss != null && d.best_loss !== 0) {
      const sign = d.best_loss < 0 ? '' : '+';
      deltas.trends = { text: `${sign}${d.best_loss.toFixed(3)} loss`, positive: d.best_loss < 0 };
    }
    if (d.best_novelty != null && d.best_novelty !== 0) {
      const sign = d.best_novelty > 0 ? '+' : '';
      deltas.reports = { text: `${sign}${d.best_novelty.toFixed(3)} nov`, positive: d.best_novelty > 0 };
    }
    return deltas;
  }, [data?.deltas]);

  const handleSelectExperiment = (expId) => {
    setSelectedExperiment(expId);
    setActiveTab('experiment-detail');
  };

  const handleBackFromExperiment = () => {
    setSelectedExperiment(null);
    setActiveTab('experiments');
  };

  const handleSelectProgram = (resultId) => {
    if (resultId === '_CANDIDATES_TAB_') {
      setActiveTab('experiments');
      return;
    }
    if (resultId === '_QUALIFIED_TAB_') {
      setActiveTab('discoveries');
      return;
    }
    setSelectedProgram(resultId);
  };
  const handleViewInLeaderboard = (resultId) => {
    setLeaderboardHighlight(resultId);
    setActiveTab('discoveries');
  };

  const handleHypothesisHandoff = useCallback((handoff) => {
    const suggestedMode = ['single', 'investigation', 'validation'].includes(handoff?.suggestedMode)
      ? handoff.suggestedMode
      : 'single';
    setControlPanelPrefill({
      source: handoff?.source || 'campaign',
      campaignId: handoff?.campaignId || null,
      campaignTitle: handoff?.campaignTitle || null,
      objective: handoff?.objective || null,
      hypothesis: handoff?.hypothesis || null,
      suggestedMode,
      requestedAt: Date.now(),
    });
    setActiveTab('command');
  }, []);

  const handleNavigateStrategy = useCallback(() => {
    setActiveTab('overview');
    setTimeout(() => {
      const el = document.getElementById('strategy-advisor');
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 50);
  }, []);

  const ariaMood = data?.aria?.mood || 'curious';
  const autonomousActive = Boolean(autonomousMode || ariaCycle?.continuous_active);
  const productionReadiness = data?.production_readiness || null;

  const strategyBlocksAdvancedStart = useMemo(() => {
    if (data?.is_running) return false;
    return Boolean(activeOverviewStrategy?.action) && !allowAdvancedStartOverride;
  }, [data?.is_running, activeOverviewStrategy, allowAdvancedStartOverride]);

  const strategyLockReason = useMemo(() => {
    if (!strategyBlocksAdvancedStart || !activeOverviewStrategy) return '';
    return `Best next step is \"${activeOverviewStrategy.title}\" (Priority #${activeOverviewStrategy.id}). Use Strategy Advisor action or intentionally override advanced setup.`;
  }, [strategyBlocksAdvancedStart, activeOverviewStrategy]);

  const primaryTab = useMemo(() => {
    for (const cat in NAV_CATEGORIES) {
      if (NAV_CATEGORIES[cat].tabs.includes(activeTab)) return cat;
    }
    return 'workbench';
  }, [activeTab]);

  const {
    handleActionComplete,
    handleCycleControl,
    handleFillGapsExperiment,
    handleForceStart,
    handleInvestigate,
    handlePromoteScreening,
    handleQueueInvestigate,
    handleQueueValidate,
    handleRescreen,
    handleRerunExperiment,
    handleSelectCampaign,
    handleStartAutonomous,
    handleStartExperiment,
    handleStopAutonomous,
    handleStopExperiment,
    handleValidate,
  } = useDashboardActions({
    blockedConfig,
    cycleControlBusy,
    eligibilityByResultId,
    emitAutoRepairStarted,
    fetchDashboard,
    investigationQueue,
    overrideIneligibleAlways,
    refreshSharedData,
    setActionError,
    setActionNotice,
    setAutonomousMode,
    setBlockedConfig,
    setCycleControlBusy,
    setSelectedCampaignId,
    setActiveTab,
  });

  return (
    <div className="app">
      <DashboardHeader
        ariaMood={ariaMood}
        autoRefresh={autoRefresh}
        dashboardUpdatedAt={dashboardUpdatedAt}
        fetchDashboard={fetchDashboard}
        isRunning={Boolean(data?.is_running)}
        onAutoRefreshChange={onAutoRefreshChange}
        onOpenDesigner={openDesignerBlank}
        onToggleChat={() => setShowChat((current) => !current)}
        onToggleHelp={() => setShowHelp((current) => !current)}
        onToggleSettings={() => startTransition(() => setShowSettings((current) => !current))}
        showChat={showChat}
        showHelp={showHelp}
        showSettings={showSettings}
      />
      <DashboardNav
        activeTab={activeTab}
        primaryTab={primaryTab}
        setActiveTab={setActiveTab}
        setSelectedExperiment={setSelectedExperiment}
        tabDeltas={tabDeltas}
      />

      <main className="app-main">
        <DashboardStatusBanners
          actionError={actionError}
          actionNotice={actionNotice}
          activeTab={activeTab}
          autonomousActive={autonomousActive}
          blockedConfig={blockedConfig}
          data={data}
          error={error}
          initialLoading={initialLoading}
          investigationQueue={investigationQueue}
          onClearActionError={() => setActionError(null)}
          onClearActionNotice={() => setActionNotice(null)}
          onForceStart={handleForceStart}
          onOpenLiveView={() => setActiveTab('command')}
          onQueueClear={handleQueueClear}
          onQueueInvestigate={handleQueueInvestigate}
          onQueueValidate={handleQueueValidate}
          queueBreakdown={queueBreakdown}
          ariaCycle={ariaCycle}
        />
        <AppTabContent
          activeTab={activeTab}
          apiBase={API_BASE}
          ariaCycle={ariaCycle}
          autonomousActive={autonomousActive}
          centralizedEntries={centralizedEntries}
          centralizedInsights={centralizedInsights}
          comparisonList={comparisonList}
          cycleControlBusy={cycleControlBusy}
          data={data}
          eligibilityByResultId={eligibilityByResultId}
          experimentsHasMore={experimentsHasMore}
          experimentsLoadingMore={experimentsLoadingMore}
          experimentsPageSize={experimentsPageSize}
          handleAddToComparison={handleAddToComparison}
          handleBackFromExperiment={handleBackFromExperiment}
          handleCycleControl={handleCycleControl}
          handleFillGapsExperiment={handleFillGapsExperiment}
          handleHypothesisHandoff={handleHypothesisHandoff}
          handleInvestigate={handleInvestigate}
          handleLoadMoreExperiments={handleLoadMoreExperiments}
          handleNavigateStrategy={handleNavigateStrategy}
          handlePromoteScreening={handlePromoteScreening}
          handleQueueAdd={handleQueueAdd}
          handleQueueRemove={handleQueueRemove}
          handleRemoveFromComparison={handleRemoveFromComparison}
          handleRescreen={handleRescreen}
          handleRerunExperiment={handleRerunExperiment}
          handleSelectCampaign={handleSelectCampaign}
          handleSelectExperiment={handleSelectExperiment}
          handleSelectProgram={handleSelectProgram}
          handleStartAutonomous={handleStartAutonomous}
          handleStartExperiment={handleStartExperiment}
          handleStopAutonomous={handleStopAutonomous}
          handleStopExperiment={handleStopExperiment}
          handleValidate={handleValidate}
          handleViewInLeaderboard={handleViewInLeaderboard}
          leaderboardEntries={leaderboardEntries}
          leaderboardHighlight={leaderboardHighlight}
          learningTrajectory={learningTrajectory}
          onActiveOverviewStrategyChange={setActiveOverviewStrategy}
          onExperimentPageSizeChange={handleExperimentPageSizeChange}
          onHighlightClear={() => setLeaderboardHighlight(null)}
          onOpenDesignerForResult={openDesignerForResult}
          paginatedExperiments={paginatedExperiments}
          productionReadiness={productionReadiness}
          queuedResultIds={investigationQueue.map(item => item.resultId)}
          refreshSharedData={refreshSharedData}
          reportsCampaignsVisible={reportsCampaignsVisible}
          reportsDeferredReady={reportsDeferredReady}
          reportsKnowledgeVisible={reportsKnowledgeVisible}
          selectedCampaignId={selectedCampaignId}
          selectedExperiment={selectedExperiment}
          setActiveTab={setActiveTab}
          setReportsCampaignsVisible={setReportsCampaignsVisible}
          setReportsKnowledgeVisible={setReportsKnowledgeVisible}
        />
      </main>

      <ChatDrawer
        open={showChat}
        onClose={() => setShowChat(false)}
        isRunning={Boolean(data?.is_running)}
        autonomousMode={autonomousActive}
        onAutonomousEnd={() => setAutonomousMode(false)}
        fallback={<LazyFallback />}
        AriaChatPanelComponent={AriaChatPanel}
      />
      <SettingsOverlay
        open={showSettings}
        onClose={() => setShowSettings(false)}
        overrideIneligibleAlways={overrideIneligibleAlways}
        setOverrideIneligibleAlways={setOverrideIneligibleAlways}
        strategyBlocksAdvancedStart={strategyBlocksAdvancedStart}
        strategyLockReason={strategyLockReason}
        onAllowAdvancedStartOverride={() => setAllowAdvancedStartOverride(true)}
        controlPanelProps={{
          isRunning: data?.is_running,
          progress: data?.progress,
          onStart: handleStartExperiment,
          onStop: handleStopExperiment,
          onRestart: () => handleRerunExperiment(data?.recent_experiments?.[0]?.experiment_id),
          restartExperimentId: data?.recent_experiments?.[0]?.experiment_id,
          onRefresh: refreshSharedData,
          autoRecommendation: data?.last_recommendation,
          prefillRequest: controlPanelPrefill,
          onPrefillApplied: () => setControlPanelPrefill(null),
          startLocked: strategyBlocksAdvancedStart,
          startLockReason: strategyLockReason,
        }}
        ControlPanelComponent={ControlPanel}
      />
      <HelpOverlay open={showHelp} onClose={() => setShowHelp(false)} />
      <ProgramDetailOverlay
        resultId={selectedProgram}
        fallback={<LazyFallback />}
        onClose={() => setSelectedProgram(null)}
        onActionComplete={handleActionComplete}
        onSelectExperiment={handleSelectExperiment}
        onViewInLeaderboard={handleViewInLeaderboard}
        onSelectCampaign={handleSelectCampaign}
        onOpenInDesigner={openDesignerForResult}
        onAddToComparison={handleAddToComparison}
        eligibilityByResultId={eligibilityByResultId}
        defaultOverrideIneligible={overrideIneligibleAlways}
      />
      <DesignerDrawerOverlay
        open={designerSession.open}
        resultId={designerSession.resultId}
        readOnly={designerSession.readOnly}
        onClose={closeDesigner}
        fallback={<LazyFallback />}
        ArchitectureDrawerComponent={ArchitectureDrawer}
      />

      <footer className="app-footer">
        <span>HYDRA Architecture Explorer — Program Synthesis Engine</span>
        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
          Keys: 1-7 tabs · ? help · Esc close
        </span>
      </footer>
    </div>
  );
}

export default App;
