import { useCallback } from 'react';
import { postJson } from '../services/apiService';
import {
  LONG_ACTION_TIMEOUT_MS,
  parseErrorPayload,
  summarizePreflightBlock,
} from './dashboardActionUtils';

export default function useExperimentControlActions({
  blockedConfig,
  cycleControlBusy,
  emitAutoRepairStarted,
  fetchDashboard,
  refreshSharedData,
  setActionError,
  setAutonomousMode,
  setBlockedConfig,
  setCycleControlBusy,
}) {
  const refreshDashboard = useCallback(() => {
    fetchDashboard();
    if (refreshSharedData) {
      refreshSharedData();
    }
  }, [fetchDashboard, refreshSharedData]);

  const handleStartExperiment = useCallback(async (config) => {
    try {
      const res = await postJson('/api/experiments/start', config, { timeoutMs: LONG_ACTION_TIMEOUT_MS });
      if (!res.ok) {
        const err = await parseErrorPayload(res, 'Failed to start experiment');
        const startedRepair = emitAutoRepairStarted(err, 'start_experiment');
        if (startedRepair) {
          const taskId = String(err?.auto_repair_task?.task_id || '').slice(0, 12);
          setActionError(`${err.error || 'Failed to start experiment'} — auto-repair started (${taskId}).`);
        } else if (err?.preflight_blocked) {
          setActionError(summarizePreflightBlock(err, 'Preflight gate blocked launch.'));
          setBlockedConfig(config);
        } else {
          setActionError(err.error || 'Failed to start experiment');
        }
        return { ok: false, ...err };
      }
      setActionError(null);
      setBlockedConfig(null);
      refreshDashboard();
      return { ok: true };
    } catch (err) {
      setActionError(`Failed to start experiment: ${err.message}`);
      return { ok: false, error: err.message };
    }
  }, [emitAutoRepairStarted, refreshDashboard, setActionError, setBlockedConfig]);

  const handleForceStart = useCallback(() => {
    if (blockedConfig) {
      handleStartExperiment({ ...blockedConfig, preflight_override: true });
    }
  }, [blockedConfig, handleStartExperiment]);

  const handleStopExperiment = useCallback(async () => {
    try {
      const res = await postJson('/api/experiments/stop');
      if (!res.ok) {
        const err = await parseErrorPayload(res, 'Failed to stop experiment');
        setActionError(err.error || 'Failed to stop experiment');
        return;
      }
      setActionError(null);
      setAutonomousMode(false);
      refreshDashboard();
    } catch (err) {
      setActionError(`Failed to stop: ${err.message}`);
    }
  }, [refreshDashboard, setActionError, setAutonomousMode]);

  const handleRerunExperiment = useCallback(async (experimentId) => {
    if (!experimentId) {
      setActionError('No recent experiment available to restart');
      return;
    }
    try {
      const res = await postJson(`/api/experiments/${experimentId}/rerun`);
      if (!res.ok) {
        const err = await parseErrorPayload(res, 'Failed to restart experiment');
        setActionError(err.error || 'Failed to restart experiment');
        return;
      }
      setActionError(null);
      fetchDashboard();
    } catch (err) {
      setActionError(`Failed to restart experiment: ${err.message}`);
    }
  }, [fetchDashboard, setActionError]);

  const handleFillGapsExperiment = useCallback(async (experimentId) => {
    if (!experimentId) {
      setActionError('No experiment selected for gap fill');
      return;
    }
    try {
      const res = await postJson(`/api/experiments/${experimentId}/fill-gaps`);
      if (!res.ok) {
        const err = await parseErrorPayload(res, 'Failed to fill metric gaps');
        setActionError(err.error || 'Failed to fill metric gaps');
        return;
      }
      setActionError(null);
      refreshDashboard();
    } catch (err) {
      setActionError(`Failed to fill gaps: ${err.message}`);
    }
  }, [refreshDashboard, setActionError]);

  const handleStartAutonomous = useCallback(async (config) => {
    const payload = {
      mode: 'continuous',
      model_source: 'mixed',
      source: 'action_queue',
      auto_harden: true,
      preflight_override: true,
      enforce_preflight: true,
      ...(config || {}),
    };
    try {
      const res = await postJson('/api/experiments/start', payload);
      if (!res.ok) {
        const err = await parseErrorPayload(res, 'Failed to start autonomous mode');
        const startedRepair = emitAutoRepairStarted(err, 'start_autonomous');
        if (startedRepair) {
          const taskId = String(err?.auto_repair_task?.task_id || '').slice(0, 12);
          setActionError(`${err.error || 'Failed to start autonomous mode'} — auto-repair started (${taskId}).`);
        } else if (err?.preflight_blocked) {
          setActionError(summarizePreflightBlock(err, 'Preflight gate blocked launch.'));
        } else {
          setActionError(err.error || 'Failed to start autonomous mode');
        }
        return;
      }
      setActionError(null);
      setAutonomousMode(true);
      fetchDashboard();
    } catch (err) {
      setActionError(`Failed to start autonomous mode: ${err.message}`);
    }
  }, [emitAutoRepairStarted, fetchDashboard, setActionError, setAutonomousMode]);

  const handleStopAutonomous = useCallback(async () => {
    try {
      setCycleControlBusy(true);
      const res = await postJson('/api/aria/cycle-control', { action: 'pause' });
      const payload = await parseErrorPayload(res, 'Failed to pause autonomous cycle');
      if (!res.ok || payload?.error) {
        throw new Error(payload?.error || 'Failed to pause autonomous cycle');
      }
      try {
        await postJson('/api/experiments/stop');
      } catch {
        // No active experiment.
      }
      setAutonomousMode(false);
      setActionError(null);
      fetchDashboard();
    } catch (err) {
      setActionError(`Failed to stop autonomous loop: ${err.message}`);
    } finally {
      setCycleControlBusy(false);
    }
  }, [fetchDashboard, setActionError, setAutonomousMode, setCycleControlBusy]);

  const handleCycleControl = useCallback(async (action) => {
    if (!action || cycleControlBusy) return;
    setCycleControlBusy(true);
    try {
      const res = await postJson('/api/aria/cycle-control', { action });
      const payload = await parseErrorPayload(res, `Failed to ${action} cycle`);
      if (!res.ok || payload?.error) {
        throw new Error(payload?.error || `Failed to ${action} cycle`);
      }
      if (action === 'start') setAutonomousMode(true);
      if (action === 'pause') setAutonomousMode(false);
      setActionError(null);
      fetchDashboard();
    } catch (err) {
      setActionError(`Cycle control failed: ${err.message}`);
    } finally {
      setCycleControlBusy(false);
    }
  }, [cycleControlBusy, fetchDashboard, setActionError, setAutonomousMode, setCycleControlBusy]);

  return {
    handleCycleControl,
    handleFillGapsExperiment,
    handleForceStart,
    handleRerunExperiment,
    handleStartAutonomous,
    handleStartExperiment,
    handleStopAutonomous,
    handleStopExperiment,
  };
}
