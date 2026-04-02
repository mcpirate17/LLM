import { useState, useCallback, useEffect, useMemo } from 'react';
import { apiCall } from '../services/apiService';
import useLocalStorage from './useLocalStorage';

const AUTO_REPAIR_SHOW_COMPLETED_KEY = 'aria_auto_repair_show_completed_v1';

function isTerminalAgentStatus(status) {
  const normalized = String(status || '').toLowerCase();
  return normalized === 'completed' || normalized === 'failed';
}

function mergeAutoRepairTask(existing, incoming, fallbackSource = 'start') {
  if (!incoming || !incoming.task_id) return null;
  return {
    ...(existing || {}),
    ...incoming,
    source: incoming.source || existing?.source || fallbackSource,
    status: incoming.status || existing?.status || 'queued',
    updated_at: incoming.updated_at || Date.now() / 1000,
  };
}

export default function useAutoRepair({ pollTick }) {
  const [autoRepairTasks, setAutoRepairTasks] = useState([]);
  const [showCompletedRepairs, setShowCompletedRepairs] = useLocalStorage(AUTO_REPAIR_SHOW_COMPLETED_KEY, false);

  const upsertAutoRepairTask = useCallback((detail, fallbackSource = 'start') => {
    const task = detail?.task;
    if (!task || !task.task_id) {
      return false;
    }

    const nextTask = {
      ...task,
      source: detail?.source || fallbackSource,
      error: detail?.error || '',
      status: task.status || 'queued',
      updated_at: task.updated_at || Date.now() / 1000,
    };

    setAutoRepairTasks((prev) => {
      const idx = prev.findIndex((item) => item.task_id === nextTask.task_id);
      if (idx < 0) {
        return [nextTask, ...prev].slice(0, 8);
      }
      const merged = mergeAutoRepairTask(prev[idx], nextTask, fallbackSource);
      if (!merged) return prev;
      const updated = [...prev];
      updated[idx] = merged;
      return updated;
    });
    return true;
  }, []);

  // Listen for custom auto-repair-started events from other components
  useEffect(() => {
    const onAutoRepairStarted = (event) => {
      const detail = event?.detail || {};
      upsertAutoRepairTask(detail, detail?.source || 'event');
    };

    window.addEventListener('aria-auto-repair-started', onAutoRepairStarted);
    return () => {
      window.removeEventListener('aria-auto-repair-started', onAutoRepairStarted);
    };
  }, [upsertAutoRepairTask]);

  // Poll active (non-terminal) tasks for status updates
  useEffect(() => {
    const activeTaskIds = autoRepairTasks
      .filter((task) => !isTerminalAgentStatus(task?.status))
      .map((task) => task.task_id)
      .filter(Boolean);

    if (!activeTaskIds.length) return;

    Promise.all(activeTaskIds.map(async (taskId) => {
      try {
        const res = await apiCall(`/api/aria/agent/status/${encodeURIComponent(taskId)}`);
        const payload = await res.json();
        if (!res.ok || !payload?.task) return;

        const task = payload.task;
        setAutoRepairTasks((prev) => {
          const idx = prev.findIndex((item) => item.task_id === taskId);
          if (idx < 0) return prev;
          const merged = mergeAutoRepairTask(prev[idx], task, prev[idx]?.source || 'status_poll');
          if (!merged) return prev;
          const updated = [...prev];
          updated[idx] = merged;
          return updated;
        });
      } catch {
        // Ignore transient polling failures.
      }
    }));
  }, [autoRepairTasks, pollTick]);

  const activeAutoRepairTasks = useMemo(
    () => autoRepairTasks.filter((task) => !isTerminalAgentStatus(task?.status)),
    [autoRepairTasks],
  );

  const completedAutoRepairCount = useMemo(
    () => autoRepairTasks.filter((task) => isTerminalAgentStatus(task?.status)).length,
    [autoRepairTasks],
  );

  const visibleAutoRepairTasks = useMemo(() => {
    if (showCompletedRepairs) {
      return autoRepairTasks;
    }
    return activeAutoRepairTasks;
  }, [autoRepairTasks, activeAutoRepairTasks, showCompletedRepairs]);

  const handleResetAutoRepairStripPreferences = useCallback(() => {
    setShowCompletedRepairs(false);
  }, [setShowCompletedRepairs]);

  const emitAutoRepairStarted = useCallback((payload, source = 'start') => {
    const task = payload?.auto_repair_task;
    if (!payload?.auto_repair_started || !task || !task.task_id) {
      return false;
    }
    upsertAutoRepairTask({
      source,
      task,
      error: payload?.error || '',
    }, source);
    try {
      window.dispatchEvent(new CustomEvent('aria-auto-repair-started', {
        detail: {
          source,
          task,
          error: payload?.error || '',
        },
      }));
    } catch {
      // ignore UI event dispatch issues
    }
    return true;
  }, [upsertAutoRepairTask]);

  return {
    autoRepairTasks,
    setAutoRepairTasks,
    showCompletedRepairs,
    setShowCompletedRepairs,
    activeAutoRepairTasks,
    completedAutoRepairCount,
    visibleAutoRepairTasks,
    handleResetAutoRepairStripPreferences,
    emitAutoRepairStarted,
  };
}
