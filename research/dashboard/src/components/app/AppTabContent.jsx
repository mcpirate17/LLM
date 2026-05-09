import React, { Suspense } from 'react';
import AblationDiagnostics from '../AblationDiagnostics';
import TemplateSlotObservability from '../TemplateSlotObservability';
import {
  AnalyticsTab,
  LazyFallback,
  LogTab,
} from './AppShellShared';
import CommandWorkbenchTab from './CommandWorkbenchTab';
import ReportsTab from './ReportsTab';
import {
  CompareView,
  ComponentAnalyticsDashboard,
  DecisionTraces,
  Discoveries,
  ExperimentDetail,
  ExperimentList,
  InfrastructureDashboard,
  LearningPanel,
  NativeProfilePanel,
  PerfDashboard,
  ReferenceArchitectures,
} from './lazyComponents';

export default function AppTabContent(props) {
  const {
    activeTab,
    ariaCycle,
    autonomousActive,
    centralizedEntries,
    centralizedInsights,
    comparisonList,
    cycleControlBusy,
    data,
    eligibilityByResultId,
    experimentsHasMore,
    experimentsLoadingMore,
    experimentsPageSize,
    handleAddToComparison,
    handleBackFromExperiment,
    handleCapabilityRank,
    handleFillGapsExperiment,
    handleConfirm,
    handleHypothesisHandoff,
    handleInvestigate,
    handleLoadMoreExperiments,
    handleNavigateStrategy,
    handlePromoteScreening,
    handleQueueAdd,
    handleQueueRemove,
    handleRescreen,
    handleRerunExperiment,
    handleSelectCampaign,
    handleSelectExperiment,
    handleSelectProgram,
    handleStartAutonomous,
    handleStartExperiment,
    handleStopAutonomous,
    handleStopExperiment,
    handleValidate,
    handleViewInLeaderboard,
    leaderboardEntries,
    leaderboardHighlight,
    learningTrajectory,
    onActiveOverviewStrategyChange,
    onExperimentPageSizeChange,
    onHighlightClear,
    onOpenDesignerForResult,
    paginatedExperiments,
    productionReadiness,
    queuedResultIds,
    refreshSharedData,
    reportsCampaignsVisible,
    reportsDeferredReady,
    reportsKnowledgeVisible,
    selectedCampaignId,
    selectedExperiment,
    setActiveTab,
    setReportsCampaignsVisible,
    setReportsKnowledgeVisible,
  } = props;

  if (activeTab === 'command') {
    return (
      <CommandWorkbenchTab
        apiBase={props.apiBase}
        ariaCycle={ariaCycle}
        autonomousActive={autonomousActive}
        cycleControlBusy={cycleControlBusy}
        dashboardData={data}
        leaderboardEntries={leaderboardEntries}
        learningTrajectory={learningTrajectory}
        onCycleControl={props.handleCycleControl}
        onSelectProgram={handleSelectProgram}
        onSetActiveOverviewStrategy={onActiveOverviewStrategyChange}
        onSetActiveTab={setActiveTab}
        onStart={handleStartExperiment}
        onStartAutonomous={handleStartAutonomous}
        onStop={handleStopExperiment}
        onStopAutonomous={handleStopAutonomous}
        productionReadiness={productionReadiness}
      />
    );
  }

  if (activeTab === 'experiments') {
    return (
      <Suspense fallback={<LazyFallback />}>
        <ExperimentList
          experiments={paginatedExperiments}
          onSelectExperiment={handleSelectExperiment}
          onRefresh={refreshSharedData}
          onLoadMore={handleLoadMoreExperiments}
          hasMore={experimentsHasMore}
          loadingMore={experimentsLoadingMore}
          pageSize={experimentsPageSize}
          onPageSizeChange={onExperimentPageSizeChange}
        />
      </Suspense>
    );
  }

  if (activeTab === 'experiment-detail' && selectedExperiment) {
    return (
      <Suspense fallback={<LazyFallback />}>
        <ExperimentDetail
          experimentId={selectedExperiment}
          onBack={handleBackFromExperiment}
          onSelectProgram={handleSelectProgram}
        />
      </Suspense>
    );
  }

  if (activeTab === 'discoveries') {
    return (
      <Suspense fallback={<LazyFallback />}>
        <Discoveries
          onSelectProgram={handleSelectProgram}
          onAddToComparison={handleAddToComparison}
          onRescreen={handleRescreen}
          onPromoteScreening={handlePromoteScreening}
          onInvestigate={handleInvestigate}
          onCapabilityRank={handleCapabilityRank}
          onValidate={handleValidate}
          onConfirm={handleConfirm}
          highlightResultId={leaderboardHighlight}
          onHighlightClear={onHighlightClear}
          onQueueAdd={handleQueueAdd}
          onQueueRemove={handleQueueRemove}
          queuedResultIds={queuedResultIds}
          eligibilityByResultId={eligibilityByResultId}
          onOpenInDesigner={onOpenDesignerForResult}
        />
      </Suspense>
    );
  }

  if (activeTab === 'trends') {
    return (
      <Suspense fallback={<LazyFallback />}>
        <AnalyticsTab
          data={data}
          insights={centralizedInsights}
          leaderboardEntries={leaderboardEntries}
          onSelectExperiment={handleSelectExperiment}
          onSelectProgram={handleSelectProgram}
          onRerunExperiment={handleRerunExperiment}
          onFillGapsExperiment={handleFillGapsExperiment}
          onNavigateStrategy={handleNavigateStrategy}
          onStartExperiment={handleStartExperiment}
          LearningPanelComponent={LearningPanel}
        />
      </Suspense>
    );
  }

  if (activeTab === 'comparison') {
    return (
      <Suspense fallback={<LazyFallback />}>
        <CompareView
          comparisonList={comparisonList}
          onRemoveProgram={props.handleRemoveFromComparison}
          onSelectProgram={handleSelectProgram}
        />
      </Suspense>
    );
  }

  if (activeTab === 'infrastructure') {
    return (
      <Suspense fallback={<LazyFallback />}>
        <InfrastructureDashboard />
      </Suspense>
    );
  }

  if (activeTab === 'ablations') {
    return <AblationDiagnostics />;
  }

  if (activeTab === 'templates') {
    return (
      <div style={{ display: 'grid', gap: 16 }}>
        <div className="card" style={{ padding: 18 }}>
          <div className="card-title" style={{ marginBottom: 8 }}>Template &amp; Slot Observability</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: 0, lineHeight: 1.6 }}>
            Dedicated structural diagnostics for template families, weak slots, routing/MoE fast-lane fairness, and structural trend drift across recent experiments.
          </p>
        </div>
        <TemplateSlotObservability />
      </div>
    );
  }

  if (activeTab === 'components') {
    return (
      <Suspense fallback={<LazyFallback />}>
        <ComponentAnalyticsDashboard />
      </Suspense>
    );
  }

  if (activeTab === 'perf') {
    return (
      <Suspense fallback={<LazyFallback />}>
        <NativeProfilePanel />
        <PerfDashboard />
      </Suspense>
    );
  }

  if (activeTab === 'references') {
    return (
      <Suspense fallback={<LazyFallback />}>
        <ReferenceArchitectures
          leaderboardEntries={leaderboardEntries}
          onSelectProgram={handleSelectProgram}
        />
      </Suspense>
    );
  }

  if (activeTab === 'decisions') {
    return (
      <Suspense fallback={<LazyFallback />}>
        <DecisionTraces />
      </Suspense>
    );
  }

  if (activeTab === 'reports') {
    return (
      <ReportsTab
        eligibilityByResultId={eligibilityByResultId}
        handleHypothesisHandoff={handleHypothesisHandoff}
        handleInvestigate={handleInvestigate}
        handleCapabilityRank={handleCapabilityRank}
        handleConfirm={handleConfirm}
        handleQueueAdd={handleQueueAdd}
        handleQueueRemove={handleQueueRemove}
        handleSelectCampaign={handleSelectCampaign}
        handleSelectExperiment={handleSelectExperiment}
        handleSelectProgram={handleSelectProgram}
        handleValidate={handleValidate}
        onOpenDesignerForResult={onOpenDesignerForResult}
        queuedResultIds={queuedResultIds}
        reportsCampaignsVisible={reportsCampaignsVisible}
        reportsDeferredReady={reportsDeferredReady}
        reportsKnowledgeVisible={reportsKnowledgeVisible}
        selectedCampaignId={selectedCampaignId}
        setReportsCampaignsVisible={setReportsCampaignsVisible}
        setReportsKnowledgeVisible={setReportsKnowledgeVisible}
      />
    );
  }

  if (activeTab === 'log') {
    return (
      <Suspense fallback={<LazyFallback />}>
        <LogTab
          entries={centralizedEntries}
          onSelectExperiment={handleSelectExperiment}
        />
      </Suspense>
    );
  }

  return null;
}
