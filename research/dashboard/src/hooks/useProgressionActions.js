import { useCallback, useMemo } from 'react';
import { postJson } from '../services/apiService';
import {
  LONG_ACTION_TIMEOUT_MS,
  buildEligibilityFilter,
  parseErrorPayload,
} from './dashboardActionUtils';

export default function useProgressionActions({
  eligibilityByResultId,
  emitAutoRepairStarted,
  fetchDashboard,
  investigationQueue,
  overrideIneligibleAlways,
  setActionError,
  setActionNotice,
  setActiveTab,
  setSelectedCampaignId,
}) {
  const filterEligibleResultIds = useMemo(
    () => buildEligibilityFilter(eligibilityByResultId),
    [eligibilityByResultId],
  );

  const startProgression = useCallback(async (mode, resultIds) => {
    const label = mode.charAt(0).toUpperCase() + mode.slice(1);
    const eligibility = filterEligibleResultIds(mode, resultIds);
    const rawIds = Array.isArray(resultIds) ? resultIds.filter(Boolean) : [];
    const hasIneligible = rawIds.length > eligibility.eligibleIds.length;
    const shouldForceAll = overrideIneligibleAlways && rawIds.length > 0;

    const startForced = async (ids) => {
      try {
        const res = await postJson('/api/experiments/start', { mode, result_ids: ids, force: true, override_ineligible: true });
        if (!res.ok) {
          const err = await parseErrorPayload(res, `Failed to start forced ${mode}`);
          setActionError(err.error || `Failed to start forced ${mode}`);
          return;
        }
        setActionError(`${label} started with override.`);
        fetchDashboard();
      } catch (err) {
        setActionError(`Failed to start forced ${mode}: ${err.message}`);
      }
    };

    if (!eligibility.ok) {
      if (!rawIds.length) {
        setActionError(eligibility.message);
        return;
      }
      if (!shouldForceAll && !window.confirm(`${eligibility.message}\n\nForce override and start ${mode} anyway?`)) {
        setActionError(eligibility.message);
        return;
      }
      await startForced(rawIds);
      return;
    }

    if (hasIneligible && rawIds.length) {
      const confirmOverride = shouldForceAll || window.confirm(
        `${eligibility.message}\n\nForce override and include the ineligible fingerprint(s) too?`,
      );
      if (confirmOverride) {
        await startForced(rawIds);
        return;
      }
    }

    try {
      const res = await postJson('/api/experiments/start', { mode, result_ids: eligibility.eligibleIds });
      if (!res.ok) {
        const err = await parseErrorPayload(res, `Failed to start ${mode}`);
        const startedRepair = emitAutoRepairStarted(err, `start_${mode}`);
        if (startedRepair) {
          const taskId = String(err?.auto_repair_task?.task_id || '').slice(0, 12);
          setActionError(`${err.error || `Failed to start ${mode}`} — auto-repair started (${taskId}).`);
        } else {
          setActionError(err.error || `Failed to start ${mode}`);
        }
        return;
      }
      setActionError(eligibility.message || null);
      fetchDashboard();
    } catch (err) {
      setActionError(`Failed to start ${mode}: ${err.message}`);
    }
  }, [emitAutoRepairStarted, fetchDashboard, filterEligibleResultIds, overrideIneligibleAlways, setActionError]);

  const handleInvestigate = useCallback((resultIds) => startProgression('investigation', resultIds), [startProgression]);
  const handleValidate = useCallback((resultIds) => startProgression('validation', resultIds), [startProgression]);

  const handleRescreen = useCallback(async (resultId) => {
    if (!resultId) {
      setActionError('Missing result ID for screening replay.');
      return;
    }
    try {
      const res = await postJson(`/api/programs/${resultId}/rescreen`, { fast: true, device: 'cuda' }, { timeoutMs: LONG_ACTION_TIMEOUT_MS });
      const payload = await parseErrorPayload(res, 'Failed to start screening replay');
      if (!res.ok) {
        setActionError(payload?.error || 'Failed to start screening replay');
        return;
      }
      setActionNotice({
        message: `Screening replay started for ${String(resultId).slice(0, 8)} (${String(payload?.experiment_id || '').slice(0, 8)}).`,
        tone: 'info',
        clearOnExperimentId: payload?.experiment_id || null,
      });
      fetchDashboard();
    } catch (err) {
      setActionError(`Failed to start screening replay: ${err.message}`);
    }
  }, [fetchDashboard, setActionError, setActionNotice]);

  const handlePromoteScreening = useCallback(async (resultId) => {
    if (!resultId) {
      setActionError('Missing result ID for screening promotion.');
      return;
    }
    try {
      const res = await postJson(`/api/programs/${resultId}/promote-screening`);
      const payload = await parseErrorPayload(res, 'Failed to promote screening candidate');
      if (!res.ok) {
        setActionError(payload?.error || 'Failed to promote screening candidate');
        return;
      }
      setActionNotice({
        message: `Promoted ${String(resultId).slice(0, 8)} into screened candidates.`,
        tone: 'info',
        clearOnExperimentId: null,
      });
      fetchDashboard();
    } catch (err) {
      setActionError(`Failed to promote screening candidate: ${err.message}`);
    }
  }, [fetchDashboard, setActionError, setActionNotice]);

  const handleRunProductionTemplate = useCallback(async (template) => {
    const payload = template?.start_payload;
    if (!payload || typeof payload !== 'object') {
      setActionError('Invalid production template payload');
      return;
    }
    const templateMode = payload?.mode;
    let nextPayload = payload;
    let eligibilityMessage = null;
    if (templateMode === 'investigation' || templateMode === 'validation') {
      const rawResultIds = Array.isArray(payload.result_ids)
        ? payload.result_ids
        : payload.result_id
          ? [payload.result_id]
          : [];
      const eligibility = filterEligibleResultIds(templateMode, rawResultIds);
      if (!eligibility.ok) {
        setActionError(eligibility.message);
        return;
      }
      const { result_id, ...rest } = payload;
      nextPayload = { ...rest, result_ids: eligibility.eligibleIds };
      eligibilityMessage = eligibility.message || null;
    }
    try {
      const res = await postJson('/api/experiments/start', nextPayload);
      if (!res.ok) {
        const err = await parseErrorPayload(res, 'Failed to run production template');
        const startedRepair = emitAutoRepairStarted(err, 'run_production_template');
        if (startedRepair) {
          const taskId = String(err?.auto_repair_task?.task_id || '').slice(0, 12);
          setActionError(`${err.error || 'Failed to run production template'} — auto-repair started (${taskId}).`);
        } else {
          setActionError(err.error || 'Failed to run production template');
        }
        return;
      }
      setActionError(eligibilityMessage);
      setActiveTab('experiments');
      fetchDashboard();
    } catch (err) {
      setActionError(`Failed to run production template: ${err.message}`);
    }
  }, [emitAutoRepairStarted, fetchDashboard, filterEligibleResultIds, setActionError, setActiveTab]);

  const handleQueueInvestigate = useCallback(() => {
    if (!investigationQueue.length) return;
    const queuedIds = investigationQueue
      .filter((item) => item.intent === 'investigation')
      .map((item) => item.resultId);
    const eligibleIds = queuedIds.filter((resultId) => eligibilityByResultId[resultId]?.investigationEligible);
    if (!eligibleIds.length && !overrideIneligibleAlways) {
      setActionError('No queued investigation candidates are currently eligible.');
      return;
    }
    handleInvestigate(overrideIneligibleAlways ? queuedIds : eligibleIds);
  }, [eligibilityByResultId, handleInvestigate, investigationQueue, overrideIneligibleAlways, setActionError]);

  const handleQueueValidate = useCallback(() => {
    if (!investigationQueue.length) return;
    const queuedIds = investigationQueue
      .filter((item) => item.intent === 'validation')
      .map((item) => item.resultId);
    const eligibleIds = queuedIds.filter((resultId) => eligibilityByResultId[resultId]?.validationEligible);
    if (!eligibleIds.length && !overrideIneligibleAlways) {
      setActionError('No queued validation candidates are currently eligible.');
      return;
    }
    handleValidate(overrideIneligibleAlways ? queuedIds : eligibleIds);
  }, [eligibilityByResultId, handleValidate, investigationQueue, overrideIneligibleAlways, setActionError]);

  const handleActionComplete = useCallback(() => {
    fetchDashboard();
  }, [fetchDashboard]);

  const handleSelectCampaign = useCallback((campaignId) => {
    setSelectedCampaignId(campaignId);
    setActiveTab('reports');
  }, [setActiveTab, setSelectedCampaignId]);

  return {
    handleActionComplete,
    handleInvestigate,
    handlePromoteScreening,
    handleQueueInvestigate,
    handleQueueValidate,
    handleRescreen,
    handleRunProductionTemplate,
    handleSelectCampaign,
    handleValidate,
  };
}
