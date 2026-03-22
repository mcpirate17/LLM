import { useState, useEffect, useCallback, useMemo } from 'react';
import { apiCall } from '../services/apiService';
import { useEventBus } from './useEventBus';
import { useAriaData } from './useAriaData';
import computeStrategy from '../components/strategyAdvisor/computeStrategy';

const API_BASE = process.env.REACT_APP_API_URL || '';

const ACTION_CONFIGS = {
  investigate: { suggestedMode: 'investigation', source: 'strategy_advisor', configOverrides: {} },
  continuous: { suggestedMode: 'continuous', source: 'mixed', configOverrides: { model_source: 'mixed' } },
  start_first: { suggestedMode: 'continuous', source: 'mixed', configOverrides: { model_source: 'mixed' } },
  novelty_search: { suggestedMode: 'novelty', source: 'mixed', configOverrides: { model_source: 'mixed' } },
  novelty: { suggestedMode: 'novelty', source: 'mixed', configOverrides: { model_source: 'mixed' } },
  scale_up: { suggestedMode: 'scale_up', source: 'strategy_advisor', configOverrides: {} },
  validate: { suggestedMode: 'validation', source: 'strategy_advisor', configOverrides: {} },
  validation: { suggestedMode: 'validation', source: 'strategy_advisor', configOverrides: {} },
  evolve: { suggestedMode: 'evolve', source: 'mixed', configOverrides: { model_source: 'mixed' } },
  investigation: { suggestedMode: 'investigation', source: 'strategy_advisor', configOverrides: {} },
};

function normalizeSuggestedMode(mode) {
  if (!mode) return null;
  const normalized = String(mode).trim().toLowerCase();
  const aliases = {
    evolution: 'evolve',
    novelty_search: 'novelty',
    investigate: 'investigation',
    validate: 'validation',
    'scale-up': 'scale_up',
  };
  return aliases[normalized] || normalized;
}

function sanitizeBriefingText(text) {
  const raw = String(text || '');
  if (!raw.trim()) return '';
  return raw
    .replace(/```(?!action\b)[\s\S]*?```/gi, '[details sent to local agent]')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function summarizeBriefingText(text, maxChars = 320) {
  const cleaned = sanitizeBriefingText(text);
  if (!cleaned) return cleaned;
  if (cleaned.length <= maxChars) return cleaned;
  return `${cleaned.slice(0, maxChars - 1).trimEnd()}\u2026`;
}

export function fallbackReasonLabel(reason) {
  if (!reason) return 'unknown';
  if (reason === 'llm_not_configured') return 'LLM not configured';
  if (reason === 'llm_unreachable') return 'LLM configured but unreachable';
  if (reason === 'llm_empty_response') return 'LLM returned no briefing text';
  if (String(reason).startsWith('llm_error:')) {
    const detail = String(reason).slice('llm_error:'.length).trim();
    return `LLM error: ${detail || 'unknown'}`;
  }
  return String(reason);
}

/**
 * Extract structured dataSources from briefing evidence.
 */
export function extractBriefingDataSources(evidence) {
  if (!evidence) return [];
  const sources = [];
  if (typeof evidence.learning_trend === 'string' && evidence.learning_trend) {
    sources.push({ metric: 'Learning Trend', value: evidence.learning_trend, threshold: null, comparison: 'status', tab: 'learning' });
  }
  if (typeof evidence.avg_recent_s1_rate === 'number') {
    sources.push({ metric: 'Recent Avg S1 Rate', value: `${(evidence.avg_recent_s1_rate * 100).toFixed(1)}%`, threshold: null, comparison: 'context', tab: 'trends' });
  }
  if (typeof evidence.recent_zero_s1_runs === 'number' && typeof evidence.recent_completed_runs === 'number') {
    sources.push({ metric: 'Zero-Survivor Runs', value: `${evidence.recent_zero_s1_runs}/${evidence.recent_completed_runs}`, threshold: null, comparison: 'ratio', tab: 'trends' });
  }
  if (typeof evidence.recent_cancelled_runs === 'number' && evidence.recent_cancelled_runs > 0) {
    sources.push({ metric: 'Cancelled Runs', value: evidence.recent_cancelled_runs, threshold: 0, comparison: '>', tab: 'experiments' });
  }
  if (evidence.pipeline && typeof evidence.pipeline === 'object') {
    const p = evidence.pipeline;
    sources.push({ metric: 'Pipeline Distribution', value: `S${p.screening || 0} / I${p.investigation || 0} / V${p.validation || 0} / B${p.breakthrough || 0}`, threshold: null, comparison: 'context', tab: 'leaderboard' });
  }
  const sparse = evidence.sparse || null;
  if (sparse && typeof sparse.n_sparse_programs === 'number' && sparse.n_sparse_programs > 0) {
    sources.push({ metric: 'Sparse Program Count', value: sparse.n_sparse_programs, threshold: null, comparison: 'context', tab: 'trends' });
    if (typeof sparse.avg_density_mean === 'number') {
      sources.push({ metric: 'Avg Sparse Density', value: `${(sparse.avg_density_mean * 100).toFixed(1)}%`, threshold: null, comparison: 'context', tab: 'trends' });
    }
  }
  const sc = evidence.sparse_coverage || null;
  if (sc && typeof sc.sparse_share === 'number') {
    const target = typeof sc.target_share === 'number' ? sc.target_share : 0.15;
    sources.push({ metric: 'Sparsity Coverage', value: `${(sc.sparse_share * 100).toFixed(1)}%`, threshold: `${(target * 100).toFixed(0)}%`, comparison: sc.sparse_share < target ? '<' : '>=', tab: 'trends' });
  }
  return sources;
}

/**
 * Custom hook encapsulating all StrategyAdvisor state, data fetching, and action handlers.
 */
export default function useStrategyData({ dashboardData, onApplyStrategy, onStart, onStop, onStartAutonomous, onStopAutonomous, onStrategyChange }) {
  const {
    leaderboardEntries,
    learningTrajectory,
    mathFamilyCoverage,
  } = useAriaData() || {};

  const [briefing, setBriefing] = useState(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [startingAutonomous, setStartingAutonomous] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [showLimits, setShowLimits] = useState(false);
  const [autoMaxExperiments, setAutoMaxExperiments] = useState(20);
  const [autoMaxMinutes, setAutoMaxMinutes] = useState(60);
  const [diagnosing, setDiagnosing] = useState(false);
  const [diagResult, setDiagResult] = useState(null);

  const fetchBriefing = useCallback(async (justCompletedId) => {
    try {
      const briefingUrl = justCompletedId
        ? `${API_BASE}/api/strategy/briefing?just_completed=${encodeURIComponent(justCompletedId)}`
        : `${API_BASE}/api/strategy/briefing`;
      const brRes = await fetch(briefingUrl);
      if (brRes.ok) {
        const brData = await brRes.json();
        if (brData && !brData.error) {
          setBriefing(brData);
        }
      }
    } catch {
      // Silently fail -- strategy will use available data
    }
    setLoading(false);
    setAnalyzing(false);
  }, []);

  // SSE listener: auto-refresh when experiment completes
  useEventBus('experiment_completed', useCallback((data) => {
    setAnalyzing(true);
    const expId = data.experiment_id || null;
    setTimeout(() => fetchBriefing(expId), 2000);
  }, [fetchBriefing]));

  // Re-fetch briefing when LLM is configured
  useEffect(() => {
    const handler = () => {
      setAnalyzing(true);
      setTimeout(() => fetchBriefing(), 500);
    };
    window.addEventListener('llm-configured', handler);
    return () => window.removeEventListener('llm-configured', handler);
  }, [fetchBriefing]);

  useEffect(() => {
    fetchBriefing();
    const interval = setInterval(fetchBriefing, 30000);
    return () => clearInterval(interval);
  }, [fetchBriefing]);

  const normalizedMathCoverage = useMemo(() => Array.isArray(mathFamilyCoverage)
    ? mathFamilyCoverage
    : mathFamilyCoverage?.families || [], [mathFamilyCoverage]);

  const strategy = useMemo(() => computeStrategy(dashboardData, leaderboardEntries, normalizedMathCoverage), [dashboardData, leaderboardEntries, normalizedMathCoverage]);

  useEffect(() => {
    if (onStrategyChange) {
      onStrategyChange(strategy);
    }
  }, [onStrategyChange, strategy]);

  // --- Derived display values ---
  const hasBriefing = briefing && briefing.briefing;
  const isAiPowered = briefing?.ai_powered === true;
  const suggestedConfig = briefing?.suggested_config;
  const briefingSummary = hasBriefing ? summarizeBriefingText(briefing.briefing) : '';

  const briefingAction = briefing?.action;
  const normalizedSuggestedModeVal = normalizeSuggestedMode(suggestedConfig?.mode || briefingAction);
  const actionConfig = suggestedConfig
    ? {
        suggestedMode: normalizedSuggestedModeVal || 'continuous',
        configOverrides: {
          ...suggestedConfig,
          mode: normalizedSuggestedModeVal || 'continuous',
        },
      }
    : (briefingAction && ACTION_CONFIGS[briefingAction]
        ? ACTION_CONFIGS[briefingAction]
        : strategy.action);
  const isNavigateAction = !actionConfig || briefingAction === 'export_breakthrough' || briefingAction === 'monitor_validation';
  const isActionable = Boolean(actionConfig) && !isNavigateAction;
  const actionLabel = briefing?.action_label || strategy.title;

  const buildStartConfig = useCallback(() => {
    if (!actionConfig) return null;
    if (suggestedConfig) {
      const fullConfig = { ...suggestedConfig };
      delete fullConfig.hypothesis;
      delete fullConfig.result_ids;
      return {
        ...fullConfig,
        mode: normalizedSuggestedModeVal || 'continuous',
        model_source: fullConfig.model_source || 'mixed',
        source: 'aria_briefing',
        hypothesis: suggestedConfig.hypothesis || undefined,
        result_ids: suggestedConfig.result_ids || undefined,
      };
    }
    return {
      mode: actionConfig.suggestedMode || 'continuous',
      model_source: actionConfig.configOverrides?.model_source || 'mixed',
      source: 'strategy_advisor',
      ...actionConfig.configOverrides,
    };
  }, [actionConfig, suggestedConfig, normalizedSuggestedModeVal]);

  const paramEntries = suggestedConfig
    ? Object.entries(suggestedConfig).filter(([k, v]) =>
        k !== 'mode' && k !== 'model_source' && k !== 'hypothesis' && v != null)
    : [];
  const paramSummary = paramEntries.slice(0, 5).map(([k, v]) => {
    let display;
    if (typeof v === 'number') display = Number.isInteger(v) ? v : v.toFixed(2);
    else if (typeof v === 'object' && v !== null) display = JSON.stringify(v);
    else display = String(v);
    return `${k.replace(/_/g, ' ')}: ${display}`;
  });

  const handleStartClick = useCallback(async () => {
    const config = buildStartConfig();
    if (!config || !onStart) return;
    setStarting(true);
    try {
      await onStart(config);
    } finally {
      setStarting(false);
    }
  }, [buildStartConfig, onStart]);

  const handleStartAutonomous = useCallback(async () => {
    if (!onStartAutonomous) return;
    setStartingAutonomous(true);
    try {
      await onStartAutonomous({
        mode: 'continuous',
        model_source: 'mixed',
        source: 'autonomous_mode',
        max_experiments: autoMaxExperiments,
        max_time_minutes: autoMaxMinutes,
      });
    } finally {
      setStartingAutonomous(false);
    }
  }, [onStartAutonomous, autoMaxExperiments, autoMaxMinutes]);

  const handleNavigateClick = useCallback(() => {
    if (onApplyStrategy) {
      onApplyStrategy({
        action: briefingAction || strategy.action || null,
        actionLabel,
        source: briefingAction ? 'briefing' : 'strategy',
        strategy,
      });
    }
  }, [onApplyStrategy, briefingAction, strategy, actionLabel]);

  const handleDiagnose = async () => {
    setDiagnosing(true);
    setDiagResult(null);
    try {
      const res = await apiCall('/api/aria/diagnose', { method: 'POST' });
      if (res.ok) {
        const result = await res.json();
        setDiagResult(result);
        if (result.actions_applied && result.actions_applied.length > 0) {
          setTimeout(() => fetchBriefing(), 1500);
        }
      } else {
        setDiagResult({ error: 'Diagnosis request failed' });
      }
    } catch {
      setDiagResult({ error: 'Could not reach server' });
    }
    setDiagnosing(false);
  };

  // Evidence items for display
  const evidence = briefing?.evidence || null;
  const sparseEvidence = evidence?.sparse || briefing?.data?.sparse || null;
  const sparseCoverage = evidence?.sparse_coverage || null;
  const evidenceItems = [];
  if (evidence) {
    if (typeof evidence.learning_trend === 'string' && evidence.learning_trend) {
      evidenceItems.push({ label: `Trend: ${evidence.learning_trend}`, tab: 'learning' });
    }
    if (typeof evidence.avg_recent_s1_rate === 'number') {
      evidenceItems.push({ label: `Recent S1: ${(evidence.avg_recent_s1_rate * 100).toFixed(1)}%`, tab: 'trends' });
    }
    if (typeof evidence.recent_zero_s1_runs === 'number' && typeof evidence.recent_completed_runs === 'number') {
      evidenceItems.push({ label: `Zero-S1 runs: ${evidence.recent_zero_s1_runs}/${evidence.recent_completed_runs}`, tab: 'trends' });
    }
    if (typeof evidence.recent_cancelled_runs === 'number' && evidence.recent_cancelled_runs > 0) {
      evidenceItems.push({ label: `Cancelled runs: ${evidence.recent_cancelled_runs}`, tab: 'experiments' });
    }
    if (evidence.pipeline && typeof evidence.pipeline === 'object') {
      const p = evidence.pipeline;
      evidenceItems.push({ label: `Pipeline: S${p.screening || 0} / I${p.investigation || 0} / V${p.validation || 0} / B${p.breakthrough || 0}`, tab: 'leaderboard' });
    }
    if (sparseEvidence && typeof sparseEvidence.n_sparse_programs === 'number' && sparseEvidence.n_sparse_programs > 0) {
      evidenceItems.push({ label: `Sparse runs: ${sparseEvidence.n_sparse_programs}`, tab: 'trends' });
      if (typeof sparseEvidence.avg_density_mean === 'number') {
        evidenceItems.push({ label: `Sparse density: ${(sparseEvidence.avg_density_mean * 100).toFixed(1)}%`, tab: 'trends' });
      }
      if (typeof sparseEvidence.avg_nm_compliance === 'number') {
        evidenceItems.push({ label: `N:M compliance: ${(sparseEvidence.avg_nm_compliance * 100).toFixed(1)}%`, tab: 'trends' });
      }
      if (Array.isArray(sparseEvidence.top_sparse_ops) && sparseEvidence.top_sparse_ops.length > 0) {
        const topOp = sparseEvidence.top_sparse_ops[0];
        if (topOp?.op_name) {
          evidenceItems.push({ label: `Top sparse op: ${topOp.op_name}`, tab: 'trends' });
        }
      }
    }
    if (sparseCoverage && typeof sparseCoverage.sparse_share === 'number') {
      const targetShare = typeof sparseCoverage.target_share === 'number' ? sparseCoverage.target_share : 0.15;
      evidenceItems.push({ label: `Sparse coverage: ${(sparseCoverage.sparse_share * 100).toFixed(1)}% (target ${(targetShare * 100).toFixed(0)}%)`, tab: 'trends' });
      if (typeof sparseCoverage.sparse_survival_rate === 'number') {
        evidenceItems.push({ label: `Sparse survival: ${(sparseCoverage.sparse_survival_rate * 100).toFixed(1)}%`, tab: 'trends' });
      }
    }
  }

  const briefingDataSources = extractBriefingDataSources(evidence);
  const mergedDataSources = briefingDataSources.length > 0
    ? briefingDataSources
    : (strategy.dataSources || []);

  const navigateLabel = briefingAction === 'export_breakthrough'
    ? 'Export Breakthrough Report'
    : briefingAction === 'monitor_validation'
      ? 'Review Validation Progress'
      : 'Review in Leaderboard';

  return {
    // State
    briefing,
    loading,
    starting,
    startingAutonomous,
    analyzing,
    showLimits,
    setShowLimits,
    autoMaxExperiments,
    setAutoMaxExperiments,
    autoMaxMinutes,
    setAutoMaxMinutes,
    diagnosing,
    diagResult,

    // Computed
    strategy,
    leaderboardEntries,
    learningTrajectory,
    hasBriefing,
    isAiPowered,
    suggestedConfig,
    briefingSummary,
    isNavigateAction,
    isActionable,
    actionLabel,
    paramSummary,
    evidenceItems,
    mergedDataSources,
    navigateLabel,

    // Handlers
    handleStartClick,
    handleStartAutonomous,
    handleNavigateClick,
    handleDiagnose,
  };
}
