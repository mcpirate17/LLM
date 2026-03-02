import { apiCall } from "../services/apiService";
import React, { useState, useEffect, useCallback } from 'react';
import { useEventBus } from '../hooks/useEventBus';
import { useAriaData } from '../hooks/useAriaData';
import useRenderPerf from '../hooks/useRenderPerf';
import { TIER_COLORS } from '../utils/scoringEngine';

const API_BASE = process.env.REACT_APP_API_URL || '';

/**
 * Map briefing action types to experiment start configs.
 */
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

function fallbackReasonLabel(reason) {
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
  if (!cleaned) return '';
  if (cleaned.length <= maxChars) return cleaned;
  return `${cleaned.slice(0, maxChars - 1).trimEnd()}…`;
}

/**
 * Pure deterministic strategy computation.
 * Returns { id, title, rationale, action, tierSummary }.
 */
export function computeStrategy(dashboard, leaderboard, mathCoverage) {
  // Build tier summary from leaderboard
  const entries = Array.isArray(leaderboard) ? leaderboard : [];
  const tierSummary = {
    screening: 0,
    investigation: 0,
    validation: 0,
    breakthrough: 0,
  };

  const breakthroughCandidates = [];
  const validationPassed = [];
  const investigationPassed = [];
  const investigationFailed = [];
  const screeningSurvivors = [];

  const normalizeTier = (entry) => {
    const tier = typeof entry?.tier === 'string' ? entry.tier.toLowerCase() : '';
    if (tier === 'screening' || tier === 'investigation' || tier === 'validation' || tier === 'breakthrough') {
      return tier;
    }
    return null;
  };

  for (const entry of entries) {
    // Skip pinned reference architectures — they are baselines, not discoveries
    const rid = String(entry?.result_id || '').toLowerCase();
    const refName = String(entry?.reference_name || '').trim();
    if (
      entry.is_reference ||
      entry.is_pinned ||
      entry.model_source === 'reference' ||
      refName.length > 0 ||
      rid.startsWith('ref_')
    ) {
      continue;
    }
    const tier = normalizeTier(entry);
    const effectiveTier = tier || 'screening';
    tierSummary[effectiveTier] += 1;
    if (effectiveTier === 'breakthrough') {
      breakthroughCandidates.push(entry);
    } else if (effectiveTier === 'validation' && entry.validation_passed) {
      validationPassed.push(entry);
    } else if (effectiveTier === 'investigation' && entry.investigation_passed) {
      investigationPassed.push(entry);
    } else if (effectiveTier === 'investigation' && !entry.investigation_passed) {
      investigationFailed.push(entry);
    } else if (effectiveTier === 'screening') {
      screeningSurvivors.push(entry);
    }
  }

  const totalExperiments = dashboard?.summary?.total_experiments || 0;
  const totalPrograms = dashboard?.summary?.total_programs_evaluated || 0;
  const stage1Survivors = dashboard?.summary?.stage1_survivors || 0;
  const survivalRate = totalPrograms > 0 ? stage1Survivors / totalPrograms : 0;

  // Check recent experiment history for consecutive zero-survivor runs
  const recentExperiments = Array.isArray(dashboard?.recent_experiments)
    ? dashboard.recent_experiments.slice(0, 3)
    : [];
  const lastThreeZeroSurvivors = recentExperiments.length >= 3 &&
    recentExperiments.every(exp => (exp?.stage1_survivors || exp?.s1_passed || 0) === 0);

  // Math family coverage gaps
  const families = Array.isArray(mathCoverage) ? mathCoverage : [];
  const undertestedFamilies = families.filter(f => (f.tested_share || 0) < 0.05);

  // Priority rules (1-10)
  // Each return includes dataSources: array of { metric, value, threshold, comparison, tab }
  // so the UI can show exactly why this recommendation was triggered.

  // 1. No experiments yet
  if (totalExperiments === 0) {
    return {
      id: 1,
      title: 'Start Mixed Continuous Research',
      rationale: 'No experiments have been run yet. Begin with continuous mixed-source research to establish a baseline of architecture candidates across graph synthesis and morphological box sources.',
      action: { suggestedMode: 'continuous', source: 'mixed', configOverrides: { model_source: 'mixed' } },
      tierSummary,
      dataSources: [
        { metric: 'Total Experiments', value: 0, threshold: 1, comparison: '<', tab: 'experiments' },
      ],
    };
  }

  // 2. Breakthrough candidates exist
  if (breakthroughCandidates.length > 0) {
    return {
      id: 2,
      title: `Export/Publish ${breakthroughCandidates.length} Breakthrough${breakthroughCandidates.length > 1 ? 's' : ''}`,
      rationale: `${breakthroughCandidates.length} candidate${breakthroughCandidates.length > 1 ? 's have' : ' has'} passed validation with high composite scores. Review and export these breakthrough architectures.`,
      action: null, // navigate to leaderboard
      tierSummary,
      dataSources: [
        { metric: 'Breakthrough Candidates', value: breakthroughCandidates.length, threshold: 1, comparison: '>=', tab: 'leaderboard' },
      ],
    };
  }

  // 3. Validation-passed candidates ready for scale-up
  if (validationPassed.length > 0) {
    return {
      id: 3,
      title: `Scale Up ${validationPassed.length} Validated Candidate${validationPassed.length > 1 ? 's' : ''}`,
      rationale: `${validationPassed.length} candidate${validationPassed.length > 1 ? 's have' : ' has'} passed validation. Scale up training to confirm performance at larger dimensions and longer sequences.`,
      action: { suggestedMode: 'scale_up', source: 'strategy_advisor', configOverrides: {} },
      tierSummary,
      dataSources: [
        { metric: 'Validation-Passed Candidates', value: validationPassed.length, threshold: 1, comparison: '>=', tab: 'leaderboard' },
      ],
    };
  }

  // 4. Investigation-passed, not yet validated
  if (investigationPassed.length > 0) {
    return {
      id: 4,
      title: `Run Validation on ${investigationPassed.length} Investigated Candidate${investigationPassed.length > 1 ? 's' : ''}`,
      rationale: `${investigationPassed.length} candidate${investigationPassed.length > 1 ? 's have' : ' has'} passed investigation but not yet been validated. Run multi-seed validation to confirm robustness.`,
      action: { suggestedMode: 'validation', source: 'strategy_advisor', configOverrides: {} },
      tierSummary,
      dataSources: [
        { metric: 'Investigation-Passed Candidates', value: investigationPassed.length, threshold: 1, comparison: '>=', tab: 'leaderboard' },
        { metric: 'Validation-Passed Candidates', value: validationPassed.length, threshold: 0, comparison: '=', tab: 'leaderboard' },
      ],
    };
  }

  // 5. Screening survivors awaiting investigation
  if (screeningSurvivors.length > 0) {
    const sources = [
      { metric: 'Screening Survivors', value: screeningSurvivors.length, threshold: 1, comparison: '>=', tab: 'leaderboard' },
    ];
    if (investigationFailed.length > 0) {
      sources.push({ metric: 'Prior Investigation Failures', value: investigationFailed.length, threshold: null, comparison: 'context', tab: 'leaderboard' });
    }
    return {
      id: 5,
      title: `Investigate ${screeningSurvivors.length} Screening Survivor${screeningSurvivors.length > 1 ? 's' : ''}`,
      rationale: `${screeningSurvivors.length} candidate${screeningSurvivors.length > 1 ? 's' : ''} passed screening and ${screeningSurvivors.length > 1 ? 'are' : 'is'} awaiting investigation.${investigationFailed.length > 0 ? ` (${investigationFailed.length} prior investigation${investigationFailed.length > 1 ? 's' : ''} failed — new candidates may perform better.)` : ''} Run deeper investigation with extended training and multiple training programs.`,
      action: { suggestedMode: 'investigation', source: 'strategy_advisor', configOverrides: {} },
      tierSummary,
      dataSources: sources,
    };
  }

  // 6. All investigations failed
  if (investigationFailed.length > 0 && screeningSurvivors.length === 0 && investigationPassed.length === 0) {
    return {
      id: 6,
      title: 'Find New Candidates (All Investigations Failed)',
      rationale: `${investigationFailed.length} candidate${investigationFailed.length > 1 ? 's were' : ' was'} investigated but ${investigationFailed.length > 1 ? 'none' : 'it did not'} pass${investigationFailed.length === 1 ? '' : 'ed'}. Run more screening experiments to discover new candidates worth investigating.`,
      action: { suggestedMode: 'continuous', source: 'mixed', configOverrides: { model_source: 'mixed' } },
      tierSummary,
      dataSources: [
        { metric: 'Investigation Failures', value: investigationFailed.length, threshold: 0, comparison: '>', tab: 'leaderboard' },
        { metric: 'Screening Survivors', value: 0, threshold: 0, comparison: '=', tab: 'leaderboard' },
        { metric: 'Investigation-Passed', value: 0, threshold: 0, comparison: '=', tab: 'leaderboard' },
      ],
    };
  }

  // 7. Low survival rate
  if (totalExperiments > 10 && survivalRate < 0.01) {
    return {
      id: 7,
      title: 'Try Evolution/Novelty Search',
      rationale: `Survival rate is only ${(survivalRate * 100).toFixed(1)}% across ${totalExperiments} experiments. Population-based search can breed better candidates by combining successful traits.`,
      action: { suggestedMode: 'evolve', source: 'mixed', configOverrides: { model_source: 'mixed' } },
      tierSummary,
      dataSources: [
        { metric: 'S1 Pass Rate', value: `${(survivalRate * 100).toFixed(1)}%`, threshold: '1%', comparison: '<', tab: 'trends' },
        { metric: 'Total Experiments', value: totalExperiments, threshold: 10, comparison: '>', tab: 'experiments' },
      ],
    };
  }

  // 8. Under-tested math families
  if (undertestedFamilies.length > 0) {
    const familyNames = undertestedFamilies.slice(0, 3).map(f => f.family || f.name).join(', ');
    return {
      id: 8,
      title: 'Expand Math Space Coverage',
      rationale: `${undertestedFamilies.length} math ${undertestedFamilies.length === 1 ? 'family is' : 'families are'} under-explored (<5% tested): ${familyNames}. Increase math space weight to diversify architecture search.`,
      action: { suggestedMode: 'continuous', source: 'mixed', configOverrides: { model_source: 'mixed', math_space_weight: 4.0 } },
      tierSummary,
      dataSources: [
        { metric: 'Under-tested Math Families', value: undertestedFamilies.length, threshold: '5% coverage', comparison: '<', tab: 'learning' },
        { metric: 'Families', value: familyNames, threshold: null, comparison: 'context', tab: 'learning' },
      ],
    };
  }

  // 9. Last 3 experiments had zero survivors
  if (lastThreeZeroSurvivors) {
    return {
      id: 9,
      title: 'Novelty Search to Escape Local Minimum',
      rationale: 'The last 3 experiments produced zero survivors each. Novelty search can escape the current search region by rewarding architectural diversity over raw fitness.',
      action: { suggestedMode: 'novelty', source: 'mixed', configOverrides: { model_source: 'mixed' } },
      tierSummary,
      dataSources: [
        { metric: 'Consecutive Zero-Survivor Runs', value: 3, threshold: 3, comparison: '>=', tab: 'trends' },
      ],
    };
  }

  // 10. Default
  return {
    id: 10,
    title: 'Continue Mixed Continuous Research',
    rationale: 'The pipeline is healthy. Continue exploring the architecture space with mixed-source continuous research to find new candidates.',
    action: { suggestedMode: 'continuous', source: 'mixed', configOverrides: { model_source: 'mixed' } },
    tierSummary,
    dataSources: [
      { metric: 'Pipeline Status', value: 'healthy', threshold: null, comparison: 'nominal', tab: 'overview' },
      { metric: 'S1 Pass Rate', value: `${(survivalRate * 100).toFixed(1)}%`, threshold: null, comparison: 'context', tab: 'trends' },
      { metric: 'Total Experiments', value: totalExperiments, threshold: null, comparison: 'context', tab: 'experiments' },
    ],
  };
}

/**
 * Extract structured dataSources from briefing evidence.
 * Converts the server-provided evidence fields into the same
 * { metric, value, threshold, comparison, tab } format used by computeStrategy().
 */
function extractBriefingDataSources(evidence) {
  if (!evidence) return [];
  const sources = [];
  if (typeof evidence.learning_trend === 'string' && evidence.learning_trend) {
    sources.push({
      metric: 'Learning Trend',
      value: evidence.learning_trend,
      threshold: null,
      comparison: 'status',
      tab: 'learning',
    });
  }
  if (typeof evidence.avg_recent_s1_rate === 'number') {
    sources.push({
      metric: 'Recent Avg S1 Rate',
      value: `${(evidence.avg_recent_s1_rate * 100).toFixed(1)}%`,
      threshold: null,
      comparison: 'context',
      tab: 'trends',
    });
  }
  if (typeof evidence.recent_zero_s1_runs === 'number' && typeof evidence.recent_completed_runs === 'number') {
    sources.push({
      metric: 'Zero-Survivor Runs',
      value: `${evidence.recent_zero_s1_runs}/${evidence.recent_completed_runs}`,
      threshold: null,
      comparison: 'ratio',
      tab: 'trends',
    });
  }
  if (typeof evidence.recent_cancelled_runs === 'number' && evidence.recent_cancelled_runs > 0) {
    sources.push({
      metric: 'Cancelled Runs',
      value: evidence.recent_cancelled_runs,
      threshold: 0,
      comparison: '>',
      tab: 'experiments',
    });
  }
  if (evidence.pipeline && typeof evidence.pipeline === 'object') {
    const p = evidence.pipeline;
    sources.push({
      metric: 'Pipeline Distribution',
      value: `S${p.screening || 0} / I${p.investigation || 0} / V${p.validation || 0} / B${p.breakthrough || 0}`,
      threshold: null,
      comparison: 'context',
      tab: 'leaderboard',
    });
  }
  const sparse = evidence.sparse || null;
  if (sparse && typeof sparse.n_sparse_programs === 'number' && sparse.n_sparse_programs > 0) {
    sources.push({
      metric: 'Sparse Program Count',
      value: sparse.n_sparse_programs,
      threshold: null,
      comparison: 'context',
      tab: 'trends',
    });
    if (typeof sparse.avg_density_mean === 'number') {
      sources.push({
        metric: 'Avg Sparse Density',
        value: `${(sparse.avg_density_mean * 100).toFixed(1)}%`,
        threshold: null,
        comparison: 'context',
        tab: 'trends',
      });
    }
  }
  const sc = evidence.sparse_coverage || null;
  if (sc && typeof sc.sparse_share === 'number') {
    const target = typeof sc.target_share === 'number' ? sc.target_share : 0.15;
    sources.push({
      metric: 'Sparsity Coverage',
      value: `${(sc.sparse_share * 100).toFixed(1)}%`,
      threshold: `${(target * 100).toFixed(0)}%`,
      comparison: sc.sparse_share < target ? '<' : '>=',
      tab: 'trends',
    });
  }
  return sources;
}

function StrategyAdvisor({ dashboardData, onApplyStrategy, onStart, onStop, isRunning, autonomousMode, onStartAutonomous, onStopAutonomous, onStrategyChange, onNavigateEvidence, onOpenAdvancedPanel }) {
  useRenderPerf('StrategyAdvisor');

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
      // Silently fail — strategy will use available data
    }
    setLoading(false);
    setAnalyzing(false);
  }, []);

  // SSE listener: auto-refresh when experiment completes (via shared EventBus)
  useEventBus('experiment_completed', useCallback((data) => {
    setAnalyzing(true);
    const expId = data.experiment_id || null;
    setTimeout(() => fetchBriefing(expId), 2000);
  }, [fetchBriefing]));

  // Re-fetch briefing when LLM is configured (dispatched by ControlPanel)
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

  const normalizedMathCoverage = Array.isArray(mathFamilyCoverage)
    ? mathFamilyCoverage
    : mathFamilyCoverage?.families || [];
  const strategy = computeStrategy(dashboardData, leaderboardEntries, normalizedMathCoverage);
  const ts = strategy.tierSummary;

  useEffect(() => {
    if (onStrategyChange) {
      onStrategyChange(strategy);
    }
  }, [onStrategyChange, strategy]);

  if (loading && !leaderboardEntries?.length) {
    return (
      <div className="card" style={{ gridColumn: '1 / -1', marginBottom: 0 }}>
        <div style={{ fontSize: 13, color: 'var(--text-muted)', padding: 8, textAlign: 'center' }}>
          Loading strategy advisor...
        </div>
      </div>
    );
  }

  // --- Determine display content ---
  const hasBriefing = briefing && briefing.briefing;
  const isAiPowered = briefing?.ai_powered === true;
  const suggestedConfig = briefing?.suggested_config;
  const briefingSummary = hasBriefing ? summarizeBriefingText(briefing.briefing) : '';

  // Action button config: prefer suggested_config from briefing, fall back to strategy
  const briefingAction = briefing?.action;
  const normalizedSuggestedMode = normalizeSuggestedMode(suggestedConfig?.mode || briefingAction);
  const actionConfig = suggestedConfig
    ? {
        suggestedMode: normalizedSuggestedMode || 'continuous',
        configOverrides: {
          ...suggestedConfig,
          mode: normalizedSuggestedMode || 'continuous',
        },
      }
    : (briefingAction && ACTION_CONFIGS[briefingAction]
        ? ACTION_CONFIGS[briefingAction]
        : strategy.action);
  const isNavigateAction = !actionConfig || briefingAction === 'export_breakthrough' || briefingAction === 'monitor_validation';
  const isActionable = Boolean(actionConfig) && !isNavigateAction;

  // Action button label
  const actionLabel = briefing?.action_label || strategy.title;

  // Build final config for experiment start
  const buildStartConfig = () => {
    if (!actionConfig) return null;
    if (suggestedConfig) {
      // Preserve all AI-suggested keys so sparse/compression steering knobs are not dropped.
      const fullConfig = { ...suggestedConfig };
      delete fullConfig.hypothesis;
      delete fullConfig.result_ids;
      return {
        ...fullConfig,
        mode: normalizedSuggestedMode || 'continuous',
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
  };

  // Summarize key params for display
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

  const handleStartClick = async () => {
    const config = buildStartConfig();
    if (!config || !onStart) return;
    setStarting(true);
    try {
      await onStart(config);
    } finally {
      setStarting(false);
    }
  };

  const handleStartAutonomous = async () => {
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
  };

  const handleNavigateClick = () => {
    if (onApplyStrategy) {
      onApplyStrategy({
        action: briefingAction || strategy.action || null,
        actionLabel,
        source: briefingAction ? 'briefing' : 'strategy',
        strategy,
      });
    }
  };

  const handleDiagnose = async () => {
    setDiagnosing(true);
    setDiagResult(null);
    try {
      const res = await apiCall(`/api/aria/diagnose`, { method: 'POST' });
      if (res.ok) {
        const result = await res.json();
        setDiagResult(result);
        // Refresh briefing data after fixes applied
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

  // Navigate-action label for breakthrough/validation
  const navigateLabel = briefingAction === 'export_breakthrough'
    ? 'Export Breakthrough Report'
    : briefingAction === 'monitor_validation'
      ? 'Review Validation Progress'
      : 'Review in Leaderboard';

  const executeLabel = 'Execute Recommended Action';
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
      evidenceItems.push({
        label: `Zero-S1 runs: ${evidence.recent_zero_s1_runs}/${evidence.recent_completed_runs}`,
        tab: 'trends',
      });
    }
    if (typeof evidence.recent_cancelled_runs === 'number' && evidence.recent_cancelled_runs > 0) {
      evidenceItems.push({ label: `Cancelled runs: ${evidence.recent_cancelled_runs}`, tab: 'experiments' });
    }
    if (evidence.pipeline && typeof evidence.pipeline === 'object') {
      const p = evidence.pipeline;
      evidenceItems.push({
        label: `Pipeline: S${p.screening || 0} / I${p.investigation || 0} / V${p.validation || 0} / B${p.breakthrough || 0}`,
        tab: 'leaderboard',
      });
    }
    if (sparseEvidence && typeof sparseEvidence.n_sparse_programs === 'number' && sparseEvidence.n_sparse_programs > 0) {
      evidenceItems.push({
        label: `Sparse runs: ${sparseEvidence.n_sparse_programs}`,
        tab: 'trends',
      });
      if (typeof sparseEvidence.avg_density_mean === 'number') {
        evidenceItems.push({
          label: `Sparse density: ${(sparseEvidence.avg_density_mean * 100).toFixed(1)}%`,
          tab: 'trends',
        });
      }
      if (typeof sparseEvidence.avg_nm_compliance === 'number') {
        evidenceItems.push({
          label: `N:M compliance: ${(sparseEvidence.avg_nm_compliance * 100).toFixed(1)}%`,
          tab: 'trends',
        });
      }
      if (Array.isArray(sparseEvidence.top_sparse_ops) && sparseEvidence.top_sparse_ops.length > 0) {
        const topOp = sparseEvidence.top_sparse_ops[0];
        if (topOp?.op_name) {
          evidenceItems.push({
            label: `Top sparse op: ${topOp.op_name}`,
            tab: 'trends',
          });
        }
      }
    }
    if (sparseCoverage && typeof sparseCoverage.sparse_share === 'number') {
      const targetShare = typeof sparseCoverage.target_share === 'number' ? sparseCoverage.target_share : 0.15;
      evidenceItems.push({
        label: `Sparse coverage: ${(sparseCoverage.sparse_share * 100).toFixed(1)}% (target ${(targetShare * 100).toFixed(0)}%)`,
        tab: 'trends',
      });
      if (typeof sparseCoverage.sparse_survival_rate === 'number') {
        evidenceItems.push({
          label: `Sparse survival: ${(sparseCoverage.sparse_survival_rate * 100).toFixed(1)}%`,
          tab: 'trends',
        });
      }
    }
  }

  // Merge data sources: briefing evidence (if AI-powered) takes priority,
  // then fall back to the deterministic strategy's dataSources.
  const briefingDataSources = extractBriefingDataSources(evidence);
  const mergedDataSources = briefingDataSources.length > 0
    ? briefingDataSources
    : (strategy.dataSources || []);

  return (
    <div className="card strategy-advisor" style={{ gridColumn: '1 / -1', marginBottom: 0 }}>
      {/* Aria's Analysis — the main briefing */}
      <div style={{
        padding: '12px 14px',
        marginBottom: 12,
        background: 'var(--bg-tertiary)',
        borderRadius: 6,
        borderLeft: `3px solid ${isAiPowered ? 'var(--accent-purple)' : 'var(--accent-blue)'}`,
        fontSize: 13,
        lineHeight: 1.6,
        color: 'var(--text-secondary)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
          <span style={{
            fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
            letterSpacing: 0.5,
            color: isAiPowered ? 'var(--accent-purple)' : 'var(--accent-blue)',
          }}>
            Aria's Analysis
          </span>
          <span style={{
            fontSize: 9, fontWeight: 600,
            color: isAiPowered ? 'var(--accent-purple)' : 'var(--text-muted)',
            background: isAiPowered
              ? 'rgba(137, 87, 229, 0.12)'
              : 'rgba(128, 128, 128, 0.12)',
            border: `1px solid ${isAiPowered ? 'var(--accent-purple)' : 'var(--text-muted)'}`,
            borderRadius: 4,
            padding: '1px 5px',
          }}>
            {isAiPowered ? 'AI-Powered' : 'Rule-Based'}
          </span>
          {hasBriefing && briefing.data?.learning_trend && briefing.data.learning_trend !== 'insufficient_data' && (
            <TrendChip trend={briefing.data.learning_trend} slope={briefing.data.learning_slope} />
          )}
        </div>
        {analyzing ? (
          <div style={{ color: 'var(--accent-purple)', fontStyle: 'italic' }}>
            Aria is analyzing the latest results...
          </div>
        ) : hasBriefing ? (
          briefingSummary || 'No concise summary available.'
        ) : (
          <span style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>
            No briefing data available. Run an experiment to get started.
          </span>
        )}
        {briefing?.ref_comparison && (
          <div style={{
            marginTop: 6, padding: '6px 10px',
            background: briefing.ref_comparison.beats_all_references
              ? 'rgba(63, 185, 80, 0.12)' : 'rgba(139, 148, 158, 0.08)',
            borderRadius: 6,
            border: briefing.ref_comparison.beats_all_references
              ? '1px solid var(--accent-green)' : '1px solid var(--border)',
            fontSize: 12,
          }}>
            {briefing.ref_comparison.beats_all_references ? (
              <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>
                Synthesized model beats all references by {briefing.ref_comparison.margin_pct}%
                {' '}(score {briefing.ref_comparison.best_synthesized_score?.toFixed(1)} vs best ref {briefing.ref_comparison.best_reference_score?.toFixed(1)})
              </span>
            ) : (
              <span style={{ color: 'var(--text-muted)' }}>
                Best reference: {briefing.ref_comparison.best_reference_score?.toFixed(1)}
                {briefing.ref_comparison.references?.map(r =>
                  <span key={r.name}> | {r.name}: {r.score?.toFixed(1)}</span>
                )}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Suggested experiment + action button */}
      <div className="strategy-content">
        <div className="strategy-header">
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4, flexWrap: 'wrap' }}>
            <DataSourceBadge dataSources={mergedDataSources} onNavigateEvidence={onNavigateEvidence} />
            <span style={{
              fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
              color: isActionable ? 'var(--accent-green)' : 'var(--accent-yellow)',
              background: isActionable ? 'rgba(63, 185, 80, 0.16)' : 'rgba(210, 153, 34, 0.16)',
              border: `1px solid ${isActionable ? 'var(--accent-green)' : 'var(--accent-yellow)'}`,
              borderRadius: 4,
              padding: '1px 5px',
            }}>
              {isActionable ? 'Actionable' : 'Advice only'}
            </span>
            {briefing?.confidence != null && isAiPowered && (
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                Confidence: {(briefing.confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
          <div className="strategy-title">{actionLabel}</div>
          <div className="strategy-rationale">
            {briefing?.action_rationale || strategy.rationale}
          </div>
          {/* Show hypothesis if AI suggested one */}
          {suggestedConfig?.hypothesis && (
            <div style={{
              marginTop: 6, padding: '6px 10px',
              background: 'var(--bg-primary)',
              borderRadius: 4,
              fontSize: 12,
              color: 'var(--text-secondary)',
              fontStyle: 'italic',
              borderLeft: '2px solid var(--accent-purple)',
            }}>
              {suggestedConfig.hypothesis}
            </div>
          )}
          {paramSummary.length > 0 && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
              {paramSummary.map((p, i) => (
                <span key={i} style={{
                  fontSize: 11, padding: '2px 6px', borderRadius: 4,
                  background: 'var(--bg-tertiary)', color: 'var(--text-secondary)',
                }}>{p}</span>
              ))}
            </div>
          )}
        </div>

        <div className="strategy-actions">
          {isRunning && autonomousMode ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'flex-start' }}>
              <div style={{ fontSize: 12, color: 'var(--accent-purple)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>
                <span className="pulse-dot" style={{ background: 'var(--accent-purple)' }}></span>
                Autonomous mode active — Aria is running experiments automatically.
              </div>
              <button
                className="strategy-apply-btn"
                onClick={() => onStopAutonomous && onStopAutonomous()}
                style={{
                  background: 'var(--accent-red, #e74c3c)', color: '#fff', fontWeight: 600,
                  fontSize: 13, padding: '6px 16px',
                }}
              >
                Stop Autonomous Mode
              </button>
            </div>
          ) : isRunning ? (
            <div style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>
              Experiment running — Aria will analyze results when complete.
            </div>
          ) : isNavigateAction ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'flex-start' }}>
              <button
                className="strategy-apply-btn"
                onClick={handleNavigateClick}
                style={{ background: 'var(--accent-green)', color: '#000', fontWeight: 600 }}
              >
                {executeLabel}
              </button>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                Action: {navigateLabel}
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'flex-start' }}>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <button
                  className="strategy-apply-btn"
                  onClick={handleStartClick}
                  disabled={starting || !onStart}
                  style={{
                    background: 'var(--accent-green)', color: '#000', fontWeight: 600,
                    opacity: starting ? 0.7 : 1,
                    fontSize: 14,
                    padding: '8px 20px',
                  }}
                >
                  {starting ? 'Executing...' : executeLabel}
                </button>
                <button
                  className="strategy-apply-btn"
                  onClick={handleStartAutonomous}
                  disabled={startingAutonomous}
                  style={{
                    background: 'var(--accent-purple)', color: '#fff', fontWeight: 600,
                    opacity: startingAutonomous ? 0.7 : 1,
                    fontSize: 13,
                    padding: '8px 16px',
                  }}
                >
                  {startingAutonomous ? 'Starting...' : 'Start Autonomous Mode'}
                </button>
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                Action: {actionLabel}
              </div>
              <div>
                <button
                  type="button"
                  onClick={() => setShowLimits(v => !v)}
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    fontSize: 11, color: 'var(--text-muted)', padding: 0,
                    textDecoration: 'underline',
                  }}
                >
                  {showLimits ? 'Hide limits' : 'Autonomous limits...'}
                </button>
                {showLimits && (
                  <div style={{ display: 'flex', gap: 12, marginTop: 6, alignItems: 'center' }}>
                    <label style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                      Max experiments:
                      <input
                        type="number" min={1} max={100} value={autoMaxExperiments}
                        onChange={e => setAutoMaxExperiments(Math.max(1, Math.min(100, parseInt(e.target.value) || 20)))}
                        style={{
                          width: 48, marginLeft: 4, background: 'var(--bg-primary)',
                          border: '1px solid var(--border)', borderRadius: 4,
                          color: 'var(--text-primary)', fontSize: 11, padding: '2px 4px',
                        }}
                      />
                    </label>
                    <label style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                      Max minutes:
                      <input
                        type="number" min={5} max={480} value={autoMaxMinutes}
                        onChange={e => setAutoMaxMinutes(Math.max(5, Math.min(480, parseInt(e.target.value) || 60)))}
                        style={{
                          width: 48, marginLeft: 4, background: 'var(--bg-primary)',
                          border: '1px solid var(--border)', borderRadius: 4,
                          color: 'var(--text-primary)', fontSize: 11, padding: '2px 4px',
                        }}
                      />
                    </label>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {evidenceItems.length > 0 && (
        <div style={{
          marginTop: 8,
          padding: '8px 10px',
          borderRadius: 6,
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border)',
        }}>
          <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 4 }}>
            Why this was chosen
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {evidenceItems.slice(0, 5).map((item, i) => (
              <button
                key={i}
                type="button"
                onClick={() => {
                  if (onNavigateEvidence && item.tab) {
                    onNavigateEvidence(item.tab);
                  }
                }}
                style={{
                fontSize: 11,
                padding: '2px 6px',
                borderRadius: 4,
                background: 'var(--bg-primary)',
                color: 'var(--text-secondary)',
                border: '1px solid var(--border)',
                cursor: onNavigateEvidence && item.tab ? 'pointer' : 'default',
              }}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Diagnose & Fix button */}
      <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <button
          className="strategy-apply-btn"
          onClick={handleDiagnose}
          disabled={diagnosing}
          style={{
            background: 'var(--accent-yellow)',
            color: '#000',
            fontWeight: 600,
            fontSize: 12,
            padding: '5px 12px',
            opacity: diagnosing ? 0.7 : 1,
          }}
        >
          {diagnosing ? 'Diagnosing...' : 'Diagnose & Fix'}
        </button>
        {diagResult && !diagResult.error && (
          <span style={{ fontSize: 11, color: diagResult.actions_applied?.length > 0 ? 'var(--accent-green)' : 'var(--text-muted)' }}>
            {diagResult.summary}
          </span>
        )}
        {diagResult?.error && (
          <span style={{ fontSize: 11, color: 'var(--accent-red, #e74c3c)' }}>
            {diagResult.error}
          </span>
        )}
      </div>
      {diagResult && diagResult.issues && diagResult.issues.length > 0 && (
        <div style={{
          marginTop: 6,
          padding: '6px 10px',
          borderRadius: 6,
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border)',
          fontSize: 11,
          lineHeight: 1.6,
        }}>
          {diagResult.issues.map((issue, i) => (
            <div key={i} style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
              <span style={{ color: issue.fixed ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                {issue.fixed ? '\u2713' : '\u2022'}
              </span>
              <span style={{ color: 'var(--text-secondary)' }}>
                {issue.issue}
                {issue.fixed && <span style={{ color: 'var(--accent-green)', marginLeft: 4 }}>(fixed)</span>}
                {issue.action_type === 'info' && <span style={{ color: 'var(--text-muted)', marginLeft: 4 }}>(info)</span>}
              </span>
            </div>
          ))}
        </div>
      )}

      {!isAiPowered && (
        <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span>Aria is in rule-based fallback mode ({fallbackReasonLabel(briefing?.fallback_reason)}).</span>
          <button
            className="refresh-btn"
            style={{ fontSize: 10, padding: '2px 8px' }}
            onClick={() => { if (onOpenAdvancedPanel) onOpenAdvancedPanel(); }}
          >
            Configure LLM
          </button>
        </div>
      )}

      <div className="strategy-pipeline" role="list" aria-label="Research pipeline">
        <PipelineBadge label="Screening" count={ts.screening} color={TIER_COLORS.screening} />
        <span className="pipeline-arrow" aria-hidden="true">&rarr;</span>
        <PipelineBadge label="Investigation" count={ts.investigation} color={TIER_COLORS.investigation} />
        <span className="pipeline-arrow" aria-hidden="true">&rarr;</span>
        <PipelineBadge label="Validation" count={ts.validation} color={TIER_COLORS.validation} />
        <span className="pipeline-arrow" aria-hidden="true">&rarr;</span>
        <PipelineBadge label="Breakthrough" count={ts.breakthrough} color={TIER_COLORS.breakthrough} />
        {learningTrajectory && learningTrajectory.trend && learningTrajectory.trend !== 'insufficient_data' && (
          <span className="pipeline-arrow" style={{ marginLeft: 'auto' }} />
        )}
        {learningTrajectory && learningTrajectory.trend && learningTrajectory.trend !== 'insufficient_data' && (
          <LearningTrendBadge trend={learningTrajectory} onNavigate={onNavigateEvidence} />
        )}
      </div>
    </div>
  );
}

export default StrategyAdvisor;

function TrendChip({ trend, slope }) {
  const color = trend === 'improving' ? 'var(--accent-green)'
    : trend === 'declining' ? 'var(--accent-red, #e74c3c)'
    : 'var(--accent-yellow)';
  const arrow = trend === 'improving' ? '\u2191' : trend === 'declining' ? '\u2193' : '\u2192';
  const label = trend === 'improving' ? 'Learning' : trend === 'declining' ? 'Declining' : 'Plateaued';
  return (
    <span style={{
      fontSize: 10, fontWeight: 600, color,
      background: `color-mix(in srgb, ${color} 12%, transparent)`,
      borderRadius: 4, padding: '1px 5px',
    }}>
      {arrow} {label}
      {slope != null && ` (${slope > 0 ? '+' : ''}${(slope * 100).toFixed(2)}%/exp)`}
    </span>
  );
}

function LearningTrendBadge({ trend, onNavigate }) {
  const t = trend.trend;
  const color = t === 'improving' ? 'var(--accent-green)'
    : t === 'declining' ? 'var(--accent-red, #e74c3c)'
    : 'var(--accent-yellow)';
  const arrow = t === 'improving' ? '\u2191' : t === 'declining' ? '\u2193' : '\u2192';
  const label = t === 'improving' ? 'Improving' : t === 'declining' ? 'Declining' : 'Plateaued';
  const slopeStr = trend.slope != null ? `${trend.slope > 0 ? '+' : ''}${(trend.slope * 100).toFixed(2)}%/exp` : '';
  const s1Str = trend.recent_s1_rate != null ? `${(trend.recent_s1_rate * 100).toFixed(1)}% recent S1` : '';
  return (
    <div 
      className="pipeline-badge" 
      onClick={() => onNavigate && onNavigate('learning')}
      style={{ 
        borderColor: color, 
        minWidth: 90,
        cursor: onNavigate ? 'pointer' : 'default'
      }}
      title="View detailed learning trajectory"
    >
      <span style={{ color, fontWeight: 700, fontSize: 13 }}>{arrow} {label}</span>
      <span className="pipeline-label">
        {slopeStr}{slopeStr && s1Str ? ' | ' : ''}{s1Str}
      </span>
    </div>
  );
}

function PipelineBadge({ label, count, color }) {
  return (
    <div className="pipeline-badge" style={{ borderColor: color }}>
      <span className="pipeline-count" style={{ color }}>{count}</span>
      <span className="pipeline-label">{label}</span>
    </div>
  );
}

/**
 * DataSourceBadge — "Recommended Action" label with a hover tooltip
 * showing the specific data points that triggered this recommendation.
 */
function DataSourceBadge({ dataSources, onNavigateEvidence }) {
  const [showTooltip, setShowTooltip] = React.useState(false);
  const hasSources = Array.isArray(dataSources) && dataSources.length > 0;

  const formatComparison = (ds) => {
    if (ds.comparison === 'context' || ds.comparison === 'status' || ds.comparison === 'nominal') {
      return `${ds.metric} = ${ds.value}`;
    }
    if (ds.comparison === 'ratio') {
      return `${ds.metric}: ${ds.value}`;
    }
    if (ds.threshold != null) {
      return `${ds.metric} ${ds.comparison} ${ds.threshold} (actual: ${ds.value})`;
    }
    return `${ds.metric}: ${ds.value}`;
  };

  return (
    <span
      style={{ position: 'relative', display: 'inline-flex' }}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <span style={{
        fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
        letterSpacing: 0.3,
        color: 'var(--accent-green)',
        background: 'rgba(63, 185, 80, 0.16)',
        border: '1px solid var(--accent-green)',
        borderRadius: 4,
        padding: '1px 6px',
        cursor: hasSources ? 'help' : 'default',
      }}>
        Recommended Action
      </span>
      {showTooltip && hasSources && (
        <div style={{
          position: 'absolute',
          top: '100%',
          left: 0,
          marginTop: 6,
          minWidth: 280,
          maxWidth: 380,
          padding: '10px 12px',
          background: 'var(--bg-secondary, #1c2128)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          zIndex: 100,
          fontSize: 11,
          lineHeight: 1.6,
        }}>
          <div style={{
            fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
            color: 'var(--accent-green)', marginBottom: 6,
            letterSpacing: 0.5,
          }}>
            Data Sources
          </div>
          {dataSources.map((ds, i) => (
            <div
              key={i}
              style={{
                display: 'flex', alignItems: 'baseline', gap: 6,
                padding: '2px 0',
                color: 'var(--text-secondary)',
              }}
            >
              <span style={{
                color: ds.comparison === '<' || ds.comparison === '>'
                  ? 'var(--accent-yellow)' : 'var(--text-muted)',
                flexShrink: 0,
              }}>
                {ds.comparison === '<' || ds.comparison === '>' ? '\u26A0' : '\u2022'}
              </span>
              <span style={{ flex: 1 }}>
                {formatComparison(ds)}
              </span>
              {ds.tab && onNavigateEvidence && (
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); onNavigateEvidence(ds.tab); }}
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    fontSize: 10, color: 'var(--accent-blue)',
                    padding: 0, textDecoration: 'underline', flexShrink: 0,
                  }}
                >
                  {ds.tab}
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </span>
  );
}
