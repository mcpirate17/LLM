import React from 'react';
import SummaryCards from '../SummaryCards';
import LiveFeed from '../LiveFeed';
import GlobalParetoChart from '../GlobalParetoChart';
import ActionQueue from '../ActionQueue';
import StatusBar from '../StatusBar';
import { QuickAnalyticsPreview } from './AppShellShared';

const COMMAND_NAV_TABS = new Set(['command', 'experiments', 'discoveries', 'trends', 'reports']);
const ACTION_QUEUE_TAB_REMAP = {
  leaderboard: 'discoveries',
  learning: 'trends',
  report: 'reports',
};

export default function CommandWorkbenchTab({
  apiBase,
  ariaCycle,
  autonomousActive,
  cycleControlBusy,
  dashboardData,
  leaderboardEntries,
  learningTrajectory,
  onCycleControl,
  onSelectProgram,
  onSetActiveOverviewStrategy,
  onSetActiveTab,
  onStart,
  onStartAutonomous,
  onStop,
  onStopAutonomous,
  productionReadiness,
}) {
  return (
    <>
      <StatusBar
        aria={dashboardData?.aria}
        isRunning={dashboardData?.is_running}
        progress={dashboardData?.progress}
        ariaCycle={ariaCycle}
        onCycleControl={onCycleControl}
        cycleControlBusy={cycleControlBusy}
        learningTrajectory={learningTrajectory}
        productionReadiness={productionReadiness}
      />
      <ActionQueue
        dashboardData={dashboardData}
        isRunning={dashboardData?.is_running}
        autonomousMode={autonomousActive}
        onStart={onStart}
        onStop={onStop}
        onStartAutonomous={onStartAutonomous}
        onStopAutonomous={onStopAutonomous}
        onStrategyChange={onSetActiveOverviewStrategy}
        onNavigateTab={(tab) => {
          const mapped = ACTION_QUEUE_TAB_REMAP[tab] || tab;
          if (COMMAND_NAV_TABS.has(mapped)) {
            onSetActiveTab(mapped);
          }
        }}
        onSelectProgram={onSelectProgram}
      />
      <div className="overview-grid" style={{ marginTop: 24 }}>
        <div className="overview-left">
          <SummaryCards learningTrend={learningTrajectory} />
          <QuickAnalyticsPreview
            deltas={dashboardData?.deltas}
            learningTrajectory={learningTrajectory}
            summary={dashboardData?.summary}
            onOpenAnalytics={() => onSetActiveTab('trends')}
          />
        </div>
        <div className="overview-right card" style={{ padding: 24 }}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Discovery Frontier</div>
          <div>
            <GlobalParetoChart programs={leaderboardEntries} onSelectProgram={onSelectProgram} onNavigateTab={onSetActiveTab} />
          </div>
          <div style={{ marginTop: 20, borderTop: '1px solid var(--border)', paddingTop: 20 }}>
            <LiveFeed
              apiBase={apiBase}
              experimentId={dashboardData?.progress?.experiment_id || null}
              progress={dashboardData?.progress || null}
            />
          </div>
        </div>
      </div>
    </>
  );
}
