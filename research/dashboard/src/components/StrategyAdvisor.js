import React from 'react';
import useRenderPerf from '../hooks/useRenderPerf';
import useStrategyData from '../hooks/useStrategyData';
import RecommendationCard from './strategyAdvisor/RecommendationCard';
import StrategyControls from './strategyAdvisor/StrategyControls';
import StrategyList from './strategyAdvisor/StrategyList';

// Re-export so existing consumers (e.g. ActionQueue.js) keep working
export { default as computeStrategy } from './strategyAdvisor/computeStrategy';

function StrategyAdvisor({ dashboardData, onApplyStrategy, onStart, onStop, isRunning, autonomousMode, onStartAutonomous, onStopAutonomous, onStrategyChange, onNavigateEvidence, onOpenAdvancedPanel }) {
  useRenderPerf('StrategyAdvisor');

  const data = useStrategyData({
    dashboardData,
    onApplyStrategy,
    onStart,
    onStop,
    onStartAutonomous,
    onStopAutonomous,
    onStrategyChange,
  });

  if (data.loading && !data.leaderboardEntries?.length) {
    return (
      <div className="card" style={{ gridColumn: '1 / -1', marginBottom: 0 }}>
        <div style={{ fontSize: 13, color: 'var(--text-muted)', padding: 8, textAlign: 'center' }}>
          Loading strategy advisor...
        </div>
      </div>
    );
  }

  return (
    <div className="card strategy-advisor" style={{ gridColumn: '1 / -1', marginBottom: 0 }}>
      <RecommendationCard
        briefing={data.briefing}
        hasBriefing={data.hasBriefing}
        isAiPowered={data.isAiPowered}
        briefingSummary={data.briefingSummary}
        analyzing={data.analyzing}
        strategy={data.strategy}
        suggestedConfig={data.suggestedConfig}
        paramSummary={data.paramSummary}
        actionLabel={data.actionLabel}
        isActionable={data.isActionable}
        mergedDataSources={data.mergedDataSources}
        onNavigateEvidence={onNavigateEvidence}
      />

      <StrategyControls
        isRunning={isRunning}
        autonomousMode={autonomousMode}
        onStopAutonomous={onStopAutonomous}
        isNavigateAction={data.isNavigateAction}
        isActionable={data.isActionable}
        actionLabel={data.actionLabel}
        navigateLabel={data.navigateLabel}
        starting={data.starting}
        startingAutonomous={data.startingAutonomous}
        showLimits={data.showLimits}
        setShowLimits={data.setShowLimits}
        autoMaxExperiments={data.autoMaxExperiments}
        setAutoMaxExperiments={data.setAutoMaxExperiments}
        autoMaxMinutes={data.autoMaxMinutes}
        setAutoMaxMinutes={data.setAutoMaxMinutes}
        onStart={onStart}
        handleStartClick={data.handleStartClick}
        handleStartAutonomous={data.handleStartAutonomous}
        handleNavigateClick={data.handleNavigateClick}
        diagnosing={data.diagnosing}
        diagResult={data.diagResult}
        handleDiagnose={data.handleDiagnose}
        isAiPowered={data.isAiPowered}
        briefing={data.briefing}
        onOpenAdvancedPanel={onOpenAdvancedPanel}
      />

      <StrategyList
        tierSummary={data.strategy.tierSummary}
        learningTrajectory={data.learningTrajectory}
        evidenceItems={data.evidenceItems}
        onNavigateEvidence={onNavigateEvidence}
      />
    </div>
  );
}

export default StrategyAdvisor;
