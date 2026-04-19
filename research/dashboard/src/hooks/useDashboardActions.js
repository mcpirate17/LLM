import useExperimentControlActions from './useExperimentControlActions';
import useProgressionActions from './useProgressionActions';

export default function useDashboardActions({
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
}) {
  const experimentActions = useExperimentControlActions({
    blockedConfig,
    cycleControlBusy,
    emitAutoRepairStarted,
    fetchDashboard,
    refreshSharedData,
    setActionError,
    setAutonomousMode,
    setBlockedConfig,
    setCycleControlBusy,
  });

  const progressionActions = useProgressionActions({
    eligibilityByResultId,
    emitAutoRepairStarted,
    fetchDashboard,
    investigationQueue,
    overrideIneligibleAlways,
    setActionError,
    setActionNotice,
    setActiveTab,
    setSelectedCampaignId,
  });

  return {
    ...experimentActions,
    ...progressionActions,
  };
}
