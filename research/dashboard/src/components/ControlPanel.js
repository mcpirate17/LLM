import { apiCall } from "../services/apiService";
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useAriaData } from '../hooks/useAriaData';
import {
  TELEMETRY_PRESET_KEYS,
  clampCanaryCooldown,
  inferTelemetryPreset,
  normalizeTelemetryPresetForStorage,
  applyTelemetryPresetSettings,
} from './controlPanelTelemetryPresets';

import { DEFAULT_CONFIG, readCanaryPrefs, writeCanaryPrefs } from '../utils/configDefaults';

// Control Components
import HypothesisCritique from './control/HypothesisCritique';
import RunStatusPanel from './control/RunStatusPanel';
import ModeSelector from './control/ModeSelector';
import SystemStatusSection from './control/SystemStatusSection';
import SearchSpaceConfig from './control/SearchSpaceConfig';
import AdvancedConfigPanels from './control/AdvancedConfigPanels';
import ExperimentActions from './control/ExperimentActions';

/**
 * ControlPanel — Orchestrates experiment launching, configuration,
 * and real-time monitoring of running experiments.
 */
function ControlPanel({
  isRunning,
  progress,
  onStart,
  onStop,
  autoRecommendation,
  prefillRequest,
  onPrefillApplied,
  startLocked = false,
}) {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [hypothesis, setHypothesis] = useState('');
  const [mode, setMode] = useState('single');
  const { summary: liveSummary } = useAriaData() || {};
  const [systemStatus, setSystemStatus] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [loadingRec, setLoadingRec] = useState(false);
  const [actionError, setActionError] = useState('');
  const [blockedConfig, setBlockedConfig] = useState(null);

  // Scale-up state
  const [scaleUpUseTop, setScaleUpUseTop] = useState(true);
  const [scaleUpTopN, setScaleUpTopN] = useState(5);
  const [scaleUpIds, setScaleUpIds] = useState('');
  const [scaleUpSteps, setScaleUpSteps] = useState(5000);
  const [scaleUpBatchSize, setScaleUpBatchSize] = useState(8);
  const [scaleUpSeqLen, setScaleUpSeqLen] = useState(512);
  const [prefillSummary, setPrefillSummary] = useState(null);

  // Canary telemetry preferences (persisted to localStorage)
  const [showCanarySummary, setShowCanarySummary] = useState(() => {
    const prefs = readCanaryPrefs();
    return typeof prefs.showCanarySummary === 'boolean' ? prefs.showCanarySummary : true;
  });
  const [showCanaryRefreshHint, setShowCanaryRefreshHint] = useState(() => {
    const prefs = readCanaryPrefs();
    return typeof prefs.showCanaryRefreshHint === 'boolean' ? prefs.showCanaryRefreshHint : true;
  });
  const [canaryRefreshing, setCanaryRefreshing] = useState(false);
  const [canaryRefreshCooldownS, setCanaryRefreshCooldownS] = useState(0);
  const [canaryCooldownSeconds, setCanaryCooldownSeconds] = useState(() => {
    const prefs = readCanaryPrefs();
    return clampCanaryCooldown(prefs.canaryCooldownSeconds);
  });
  const [nativeTelemetryExpanded, setNativeTelemetryExpanded] = useState(() => {
    const prefs = readCanaryPrefs();
    return typeof prefs.nativeTelemetryExpanded === 'boolean' ? prefs.nativeTelemetryExpanded : true;
  });
  const [canaryTelemetryPreset, setCanaryTelemetryPreset] = useState(() => {
    const prefs = readCanaryPrefs();
    const storedPreset = String(prefs.canaryTelemetryPreset || '').toLowerCase();
    if (TELEMETRY_PRESET_KEYS.includes(storedPreset)) {
      return storedPreset;
    }
    const fallbackCooldown = clampCanaryCooldown(prefs.canaryCooldownSeconds);
    return inferTelemetryPreset({
      showCanarySummary: typeof prefs.showCanarySummary === 'boolean' ? prefs.showCanarySummary : true,
      showCanaryRefreshHint: typeof prefs.showCanaryRefreshHint === 'boolean' ? prefs.showCanaryRefreshHint : true,
      nativeTelemetryExpanded: typeof prefs.nativeTelemetryExpanded === 'boolean' ? prefs.nativeTelemetryExpanded : true,
      canaryCooldownSeconds: fallbackCooldown,
    });
  });
  const [canaryPrefsNotice, setCanaryPrefsNotice] = useState('');

  useEffect(() => {
    writeCanaryPrefs({
      showCanarySummary,
      showCanaryRefreshHint,
      canaryCooldownSeconds: clampCanaryCooldown(canaryCooldownSeconds),
      nativeTelemetryExpanded,
      canaryTelemetryPreset: normalizeTelemetryPresetForStorage(canaryTelemetryPreset),
    });
  }, [showCanarySummary, showCanaryRefreshHint, canaryCooldownSeconds, nativeTelemetryExpanded, canaryTelemetryPreset]);

  useEffect(() => {
    const inferred = inferTelemetryPreset({
      showCanarySummary,
      showCanaryRefreshHint,
      nativeTelemetryExpanded,
      canaryCooldownSeconds,
    });
    setCanaryTelemetryPreset((prev) => (prev === inferred ? prev : inferred));
  }, [showCanarySummary, showCanaryRefreshHint, nativeTelemetryExpanded, canaryCooldownSeconds]);

  useEffect(() => {
    if (canaryRefreshCooldownS <= 0) return;
    const timer = setTimeout(() => {
      setCanaryRefreshCooldownS(prev => Math.max(0, prev - 1));
    }, 1000);
    return () => clearTimeout(timer);
  }, [canaryRefreshCooldownS]);

  useEffect(() => {
    if (!canaryPrefsNotice) return;
    const timer = setTimeout(() => setCanaryPrefsNotice(''), 2200);
    return () => clearTimeout(timer);
  }, [canaryPrefsNotice]);

  // Auto-populate recommendation from completed experiment
  useEffect(() => {
    if (autoRecommendation && !isRunning) {
      setRecommendation(autoRecommendation);
    }
  }, [autoRecommendation, isRunning]);

  // Fetch system status on mount
  useEffect(() => {
    apiCall('/api/system/status')
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setSystemStatus(data); })
      .catch(() => {});
  }, []);

  // Investigation/validation state
  const [investIds, setInvestIds] = useState('');
  const [investUseTop, setInvestUseTop] = useState(true);
  const [investTopN, setInvestTopN] = useState(5);

  useEffect(() => {
    if (!prefillRequest || !prefillRequest.requestedAt) return;
    const validModes = ['single', 'continuous', 'evolve', 'novelty', 'scale_up', 'investigation', 'validation'];
    const suggestedMode = validModes.includes(prefillRequest.suggestedMode)
      ? prefillRequest.suggestedMode
      : 'single';
    setMode(suggestedMode);

    const objectiveText = typeof prefillRequest.objective === 'string' ? prefillRequest.objective.trim() : '';
    const hypothesisText = typeof prefillRequest.hypothesis === 'string' ? prefillRequest.hypothesis.trim() : '';
    const mergedHypothesis = [
      objectiveText ? `Objective: ${objectiveText}` : '',
      hypothesisText ? `Hypothesis: ${hypothesisText}` : '',
    ].filter(Boolean).join(' | ');
    if (mergedHypothesis) {
      setHypothesis(mergedHypothesis);
    }
    if (suggestedMode === 'investigation' || suggestedMode === 'validation') {
      setInvestUseTop(true);
    }

    if (prefillRequest.configOverrides && typeof prefillRequest.configOverrides === 'object') {
      setConfig(prev => ({ ...prev, ...prefillRequest.configOverrides }));
    }

    setPrefillSummary({
      mode: suggestedMode,
      source: prefillRequest.source || 'campaign',
      campaignTitle: prefillRequest.campaignTitle || null,
      objective: objectiveText || null,
      hypothesis: hypothesisText || null,
    });
    if (onPrefillApplied) onPrefillApplied();
  }, [prefillRequest, onPrefillApplied]);

  const handleStart = async (overrideParams = {}) => {
    setActionError('');
    setBlockedConfig(null);
    let finalConfig = {
      ...config,
      mode,
      hypothesis: hypothesis || undefined,
      ...overrideParams
    };

    // Only send non-default category weights
    if (finalConfig.category_weights && typeof finalConfig.category_weights === 'object') {
      const nonDefault = {};
      Object.entries(finalConfig.category_weights).forEach(([k, v]) => {
        if (v !== 1.0) nonDefault[k] = v;
      });
      finalConfig.category_weights = Object.keys(nonDefault).length > 0 ? nonDefault : undefined;
    }

    // Convert string-based op control fields to proper types
    if (typeof finalConfig.excluded_ops === 'string') {
      const ops = finalConfig.excluded_ops.split(',').map(s => s.trim()).filter(Boolean);
      finalConfig.excluded_ops = ops.length > 0 ? ops : undefined;
    }
    if (typeof finalConfig.op_weights === 'string') {
      const parsed = {};
      finalConfig.op_weights.split(',').map(s => s.trim()).filter(Boolean).forEach(pair => {
        const [op, w] = pair.split(':').map(s => s.trim());
        if (op && w && !isNaN(parseFloat(w))) parsed[op] = parseFloat(w);
      });
      finalConfig.op_weights = Object.keys(parsed).length > 0 ? parsed : undefined;
    }

    const processStart = async (fullPayload) => {
      try {
        const result = await onStart(fullPayload);
        if (result && !result.ok) {
          if (result.preflight_blocked) {
            setBlockedConfig(fullPayload);
            setActionError('Preflight gate blocked launch. You can Force Start to override.');
          }
        }
      } catch (e) {
        setActionError('Failed to start: ' + e.message);
      }
    };

    if (mode === 'investigation') {
      if (investUseTop) {
        try {
          const r = await apiCall(`/api/programs?n=${investTopN}&sort=loss_ratio`);
          const programs = await r.json();
          const ids = programs
            .filter(p => p.stage1_passed)
            .map(p => p.result_id)
            .slice(0, investTopN);
          if (ids.length === 0) {
            setActionError('No Stage 1 survivors found to investigate.');
            return;
          }
          await processStart({ ...finalConfig, result_ids: ids });
        } catch (e) {
          setActionError('Failed to fetch programs: ' + e.message);
        }
      } else {
        const ids = investIds.split(',').map(s => s.trim()).filter(Boolean);
        if (ids.length === 0) {
          setActionError('Please enter at least one result ID.');
          return;
        }
        await processStart({ ...finalConfig, result_ids: ids });
      }
      return;
    }
    if (mode === 'validation') {
      if (investUseTop) {
        try {
          const r = await apiCall(`/api/leaderboard?tier=investigation&sort=composite_score&limit=${investTopN}`);
          const data = await r.json();
          const ids = (data.entries || [])
            .filter(e => e.investigation_passed)
            .map(e => e.result_id)
            .slice(0, investTopN);
          if (ids.length === 0) {
            setActionError('No investigation survivors found to validate.');
            return;
          }
          await processStart({ ...finalConfig, result_ids: ids });
        } catch (e) {
          setActionError('Failed to fetch leaderboard: ' + e.message);
        }
      } else {
        const ids = investIds.split(',').map(s => s.trim()).filter(Boolean);
        if (ids.length === 0) {
          setActionError('Please enter at least one result ID.');
          return;
        }
        await processStart({ ...finalConfig, result_ids: ids });
      }
      return;
    }
    if (mode === 'scale_up') {
      const scaleUpPayload = {
        ...finalConfig,
        scale_up_steps: scaleUpSteps,
        scale_up_batch_size: scaleUpBatchSize,
        scale_up_seq_len: scaleUpSeqLen,
      };
      if (scaleUpUseTop) {
        try {
          const r = await apiCall(`/api/programs?n=${scaleUpTopN}&sort=loss_ratio`);
          const programs = await r.json();
          const ids = programs
            .filter(p => p.stage1_passed)
            .map(p => p.result_id)
            .slice(0, scaleUpTopN);
          if (ids.length === 0) {
            setActionError('No Stage 1 survivors found to scale up.');
            return;
          }
          await processStart({ ...scaleUpPayload, result_ids: ids });
        } catch (e) {
          setActionError('Failed to fetch top programs: ' + e.message);
        }
      } else {
        const ids = scaleUpIds.split(',').map(s => s.trim()).filter(Boolean);
        if (ids.length === 0) {
          setActionError('Please enter at least one result ID.');
          return;
        }
        await processStart({ ...scaleUpPayload, result_ids: ids });
      }
      return;
    }

    await processStart(finalConfig);
  };

  const updateConfig = useCallback((key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  }, []);

  const handleAskAria = useCallback(async () => {
    setLoadingRec(true);
    setRecommendation(null);
    try {
      const res = await apiCall('/api/aria/recommendation', { timeoutMs: 60000 });
      if (res.ok) {
        setRecommendation(await res.json());
      }
    } catch (e) {
      setRecommendation({ reasoning: 'Failed to get recommendation: ' + e.message });
    }
    setLoadingRec(false);
  }, []);

  const applyRecommendation = useCallback(() => {
    if (recommendation?.config) {
      setConfig(prev => {
        const next = { ...prev, ...recommendation.config };
        if (recommendation.config.category_weights && prev.category_weights) {
          next.category_weights = { ...prev.category_weights, ...recommendation.config.category_weights };
        }
        if (Array.isArray(next.excluded_ops)) {
          next.excluded_ops = next.excluded_ops.join(', ');
        }
        if (typeof next.op_weights === 'object' && next.op_weights !== null && !Array.isArray(next.op_weights)) {
          next.op_weights = Object.entries(next.op_weights).map(([k, v]) => `${k}:${v}`).join(', ');
        }
        return next;
      });
      setRecommendation(null);
    }
  }, [recommendation]);

  const handleCategoryWeightChange = useCallback((cat, val) => {
    setConfig(prev => ({ ...prev, category_weights: { ...prev.category_weights, [cat]: val } }));
  }, []);

  const isEvolutionMode = mode === 'evolve' || mode === 'novelty';

  const programTotal = progress?.total_programs || 0;
  const programCurrent = progress?.current_program || 0;
  const generationTotal = progress?.total_generations || 0;
  const generationCurrent = progress?.current_generation || 0;
  const isGenerationProgress = generationTotal > 0;

  const pct = isGenerationProgress
    ? (generationTotal > 0 ? Math.round((generationCurrent / generationTotal) * 100) : 0)
    : (programTotal > 0 ? Math.round((programCurrent / programTotal) * 100) : 0);

  const programProgressText = useMemo(() => {
    if (programTotal > 0) return `${programCurrent} / ${programTotal} programs (${pct}%)`;
    if (programCurrent > 0) return `${programCurrent} / ? programs (in progress)`;
    if (['investigating', 'validating', 'scale_up'].includes(String(progress?.status || '').toLowerCase())) return '0 / ? programs (initializing)';
    if (String(progress?.status || '').toLowerCase() === 'resuming') return 'Resuming experiment state...';
    return 'Initializing experiment...';
  }, [programTotal, programCurrent, pct, progress?.status]);

  return (
    <div className="card control-panel">
      <div className="card-title">Experiment Control</div>

      {!isRunning && (
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
          Generate and test random computation graphs as potential replacements for transformer attention layers.
        </p>
      )}

      {actionError && (
        <div style={{ marginBottom: 12, padding: '8px 10px', borderRadius: 6, border: '1px solid var(--accent-red)', background: 'rgba(248, 81, 73, 0.1)', color: 'var(--accent-red)', fontSize: 12 }}>
          {actionError}
        </div>
      )}

      {prefillSummary && !isRunning && (
        <div className="info-banner">
          <div>
            Prefill applied from {prefillSummary.source.replace('_', ' ')}: mode set to{' '}
            <strong>{prefillSummary.mode}</strong>.
          </div>
          {prefillSummary.objective && (
            <div><strong>Objective:</strong> {prefillSummary.objective}</div>
          )}
          <button className="info-banner-dismiss" onClick={() => setPrefillSummary(null)}>
            Dismiss
          </button>
        </div>
      )}

      <RunStatusPanel
        isRunning={isRunning}
        progress={progress}
        onStop={onStop}
        programProgressText={programProgressText}
        pct={pct}
        isGenerationProgress={isGenerationProgress}
        mode={mode}
      />

      {!isRunning ? (
        <>
          <SystemStatusSection
            systemStatus={systemStatus}
            onSystemStatusUpdate={setSystemStatus}
          />

          <ModeSelector selectedMode={mode} onModeChange={setMode} disabled={isRunning} />

          <div className="control-row">
            <label className="control-label" style={{ fontWeight: 700, color: 'var(--text-primary)' }}>Research Hypothesis</label>
            <input
              className="control-input"
              type="text"
              value={hypothesis}
              onChange={(e) => setHypothesis(e.target.value)}
              placeholder="Let Aria formulate one automatically based on evidence..."
              style={{ borderRadius: 8, padding: '10px 12px' }}
            />
          </div>

          <SearchSpaceConfig
            config={config}
            updateConfig={updateConfig}
            isEvolutionMode={isEvolutionMode}
          />

          <AdvancedConfigPanels
            config={config}
            updateConfig={updateConfig}
            onCategoryWeightChange={handleCategoryWeightChange}
          />

          <ExperimentActions
            mode={mode}
            loadingRec={loadingRec}
            onAskAria={handleAskAria}
            recommendation={recommendation}
            onApplyRecommendation={applyRecommendation}
            onStart={() => handleStart()}
            startLocked={startLocked}
            blockedConfig={blockedConfig}
            onForceStart={() => handleStart({ preflight_override: true })}
          />
        </>
      ) : (
        <div className="experiment-progress">
          <div className="progress-header">
            <span>{progress?.status || 'Running'}</span>
            <span style={{ opacity: 0.6 }}>{progress?.experiment_id?.slice(0, 8)}</span>
          </div>
          {progress?.hypothesis_critique && <HypothesisCritique critique={progress.hypothesis_critique} />}
        </div>
      )}
    </div>
  );
}

export default ControlPanel;
