import React, { useState, useEffect, useCallback } from 'react';

const API_BASE = process.env.REACT_APP_API_URL || '';

const DEFAULT_CONFIG = {
  n_programs: 50,
  model_dim: 256,
  n_layers: 4,
  vocab_size: 32000,
  max_seq_len: 256,
  device: 'cuda',
  stage1_steps: 500,
  stage1_lr: 0.0003,
  max_depth: 10,
  max_ops: 16,
  math_space_weight: 2.0,
  residual_prob: 0.7,
  max_experiments: 100,
  max_time_minutes: 0,
  max_cost_dollars: 0,
  // Evolution/novelty
  population_size: 50,
  n_generations: 20,
  tournament_size: 5,
  mutation_rate: 0.7,
  crossover_rate: 0.3,
  elitism: 5,
  novelty_weight: 0.5,
  fitness_weight: 0.5,
  archive_size: 200,
  k_nearest: 15,
  archive_threshold: 0.3,
  // Automation
  auto_scale_up: true,
  auto_scale_up_min_survivors: 3,
  auto_scale_up_top_n: 5,
  auto_report: true,
  auto_report_every_n: 5,
  // Model source
  model_source: 'mixed',
  morph_ratio: 0.5,
  // Training programs
  use_synthesized_training: false,
  n_training_programs: 3,
  // Auto-escalation
  auto_investigate: true,
  auto_investigate_min_survivors: 1,
  auto_investigate_top_n: 5,
  auto_validate: true,
  auto_validate_min_robustness: 0.5,
  auto_validate_top_n: 3,
  // Investigation/validation
  investigation_steps: 2500,
  investigation_batch_size: 4,
  validation_steps: 10000,
  validation_batch_size: 8,
  validation_seq_len: 512,
  validation_n_seeds: 3,
};

const CRITIQUE_VERDICT_STYLES = {
  proceed: { color: 'var(--accent-green)', label: 'Proceed', icon: '\u2714' },
  caution: { color: 'var(--accent-yellow)', label: 'Caution', icon: '\u26A0' },
  revise: { color: 'var(--accent-red)', label: 'Revise', icon: '\u2718' },
};

const CRITIQUE_GATE_STYLES = {
  pass: { color: 'var(--accent-green)', bg: 'rgba(63, 185, 80, 0.18)', label: 'Pass' },
  warn: { color: 'var(--accent-yellow)', bg: 'rgba(210, 153, 34, 0.18)', label: 'Warn' },
  fail: { color: 'var(--accent-red)', bg: 'rgba(248, 81, 73, 0.18)', label: 'Fail' },
};

function HypothesisCritique({ critique }) {
  const style = CRITIQUE_VERDICT_STYLES[critique.verdict] || CRITIQUE_VERDICT_STYLES.caution;
  const gate = typeof critique.gate === 'string' ? critique.gate : (critique.verdict === 'proceed' ? 'pass' : critique.verdict === 'revise' ? 'fail' : 'warn');
  const gateStyle = CRITIQUE_GATE_STYLES[gate] || CRITIQUE_GATE_STYLES.warn;
  const checks = Array.isArray(critique.checks) ? critique.checks : [];
  const hasConcerns = critique.concerns && critique.concerns.length > 0;
  const hasSuggestions = critique.suggestions && critique.suggestions.length > 0;

  return (
    <div style={{
      marginBottom: 10,
      padding: '8px 10px',
      borderRadius: 6,
      border: `1px solid ${style.color}`,
      background: `${style.color}11`,
      fontSize: 12,
      lineHeight: 1.5,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: hasConcerns || hasSuggestions ? 6 : 0 }}>
        <span style={{ fontSize: 14 }}>{style.icon}</span>
        <strong style={{ color: style.color }}>Hypothesis Review: {style.label}</strong>
        <span style={{
          marginLeft: 8,
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: 0.3,
          textTransform: 'uppercase',
          color: gateStyle.color,
          background: gateStyle.bg,
          border: `1px solid ${gateStyle.color}`,
          borderRadius: 4,
          padding: '1px 6px',
        }}>
          Gate: {gateStyle.label}
        </span>
        {critique.confidence != null && (
          <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 'auto' }}>
            confidence {(critique.confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>
      {checks.length > 0 && (
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 6,
          marginBottom: hasConcerns || hasSuggestions ? 6 : 0,
          paddingLeft: 20,
        }}>
          {checks.map((check, idx) => {
            const checkStyle = CRITIQUE_GATE_STYLES[check?.status] || CRITIQUE_GATE_STYLES.warn;
            const label = check?.label || check?.key || `Check ${idx + 1}`;
            return (
              <span
                key={`${label}-${idx}`}
                style={{
                  fontSize: 10,
                  color: checkStyle.color,
                  background: checkStyle.bg,
                  border: `1px solid ${checkStyle.color}`,
                  borderRadius: 4,
                  padding: '1px 6px',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 4,
                }}
              >
                <strong>{checkStyle.label}</strong>
                <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
              </span>
            );
          })}
        </div>
      )}
      {hasConcerns && (
        <div style={{ marginBottom: hasSuggestions ? 4 : 0 }}>
          {critique.concerns.map((c, i) => (
            <div key={i} style={{ color: 'var(--text-secondary)', paddingLeft: 20 }}>
              &bull; {c}
            </div>
          ))}
        </div>
      )}
      {hasSuggestions && (
        <div>
          {critique.suggestions.map((s, i) => (
            <div key={i} style={{ color: 'var(--text-muted)', paddingLeft: 20, fontStyle: 'italic' }}>
              &rarr; {s}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ControlPanel({
  isRunning,
  progress,
  onStart,
  onStop,
  onRefresh,
  autoRecommendation,
  prefillRequest,
  onPrefillApplied,
  displayMode = 'novice',
  startLocked = false,
  startLockReason = '',
}) {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [hypothesis, setHypothesis] = useState('');
  const [mode, setMode] = useState('single');
  const [showAdvanced, setShowAdvanced] = useState(displayMode === 'expert');
  const [systemStatus, setSystemStatus] = useState(null);
  const [validating, setValidating] = useState(false);
  const [validationResult, setValidationResult] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [loadingRec, setLoadingRec] = useState(false);
  const [showLlmConfig, setShowLlmConfig] = useState(false);
  const [llmConfig, setLlmConfig] = useState(null);
  const [llmForm, setLlmForm] = useState({ backend: '', api_key: '', model: '', host: '' });
  const [llmSaving, setLlmSaving] = useState(false);
  const [llmMessage, setLlmMessage] = useState('');
  const [actionError, setActionError] = useState('');
  // Scale-up state
  const [scaleUpUseTop, setScaleUpUseTop] = useState(true);
  const [scaleUpTopN, setScaleUpTopN] = useState(5);
  const [scaleUpIds, setScaleUpIds] = useState('');
  const [scaleUpSteps, setScaleUpSteps] = useState(5000);
  const [scaleUpBatchSize, setScaleUpBatchSize] = useState(8);
  const [scaleUpSeqLen, setScaleUpSeqLen] = useState(512);
  const [prefillSummary, setPrefillSummary] = useState(null);

  // Auto-populate recommendation from completed experiment
  useEffect(() => {
    if (autoRecommendation && !isRunning) {
      setRecommendation(autoRecommendation);
    }
  }, [autoRecommendation, isRunning]);

  useEffect(() => {
    setShowAdvanced(displayMode === 'expert');
  }, [displayMode]);

  // Fetch system status and LLM config on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/system/status`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setSystemStatus(data); })
      .catch(() => {});
    fetch(`${API_BASE}/api/llm/config`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setLlmConfig(data); })
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

  const handleStart = () => {
    setActionError('');
    if (mode === 'investigation') {
      const payload = {
        ...config,
        mode: 'investigation',
        hypothesis: hypothesis || undefined,
      };
      if (investUseTop) {
        fetch(`${API_BASE}/api/programs?n=${investTopN}&sort=loss_ratio`)
          .then(r => r.json())
          .then(programs => {
            const ids = programs
              .filter(p => p.stage1_passed)
              .map(p => p.result_id)
              .slice(0, investTopN);
            if (ids.length === 0) {
              setActionError('No Stage 1 survivors found to investigate.');
              return;
            }
            onStart({ ...payload, result_ids: ids });
          })
          .catch(e => setActionError('Failed to fetch programs: ' + e.message));
      } else {
        const ids = investIds.split(',').map(s => s.trim()).filter(Boolean);
        if (ids.length === 0) {
          setActionError('Please enter at least one result ID.');
          return;
        }
        onStart({ ...payload, result_ids: ids });
      }
      return;
    }
    if (mode === 'validation') {
      const payload = {
        ...config,
        mode: 'validation',
        hypothesis: hypothesis || undefined,
      };
      if (investUseTop) {
        fetch(`${API_BASE}/api/leaderboard?tier=investigation&sort=composite_score&limit=${investTopN}`)
          .then(r => r.json())
          .then(data => {
            const ids = (data.entries || [])
              .filter(e => e.investigation_passed)
              .map(e => e.result_id)
              .slice(0, investTopN);
            if (ids.length === 0) {
              setActionError('No investigation survivors found to validate.');
              return;
            }
            onStart({ ...payload, result_ids: ids });
          })
          .catch(e => setActionError('Failed to fetch leaderboard: ' + e.message));
      } else {
        const ids = investIds.split(',').map(s => s.trim()).filter(Boolean);
        if (ids.length === 0) {
          setActionError('Please enter at least one result ID.');
          return;
        }
        onStart({ ...payload, result_ids: ids });
      }
      return;
    }
    if (mode === 'scale_up') {
      const payload = {
        ...config,
        mode: 'scale_up',
        hypothesis: hypothesis || undefined,
        scale_up_steps: scaleUpSteps,
        scale_up_batch_size: scaleUpBatchSize,
        scale_up_seq_len: scaleUpSeqLen,
      };
      if (scaleUpUseTop) {
        // Fetch top N programs and use their IDs
        fetch(`${API_BASE}/api/programs?n=${scaleUpTopN}&sort=loss_ratio`)
          .then(r => r.json())
          .then(programs => {
            const ids = programs
              .filter(p => p.stage1_passed)
              .map(p => p.result_id)
              .slice(0, scaleUpTopN);
            if (ids.length === 0) {
              setActionError('No Stage 1 survivors found to scale up.');
              return;
            }
            onStart({ ...payload, result_ids: ids });
          })
          .catch(e => setActionError('Failed to fetch top programs: ' + e.message));
      } else {
        const ids = scaleUpIds.split(',').map(s => s.trim()).filter(Boolean);
        if (ids.length === 0) {
          setActionError('Please enter at least one result ID.');
          return;
        }
        onStart({ ...payload, result_ids: ids });
      }
      return;
    }
    onStart({
      ...config,
      mode,
      hypothesis: hypothesis || undefined,
    });
  };

  const updateConfig = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const handleValidate = useCallback(async () => {
    setValidating(true);
    setValidationResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n: 5 }),
      });
      if (res.ok) {
        setValidationResult(await res.json());
      }
    } catch (e) {
      setValidationResult({ healthy: false, errors: [e.message] });
    }
    setValidating(false);
  }, []);

  const handleAskAria = useCallback(async () => {
    setLoadingRec(true);
    setRecommendation(null);
    try {
      const res = await fetch(`${API_BASE}/api/aria/recommendation`);
      if (res.ok) {
        setRecommendation(await res.json());
      }
    } catch (e) {
      setRecommendation({ reasoning: 'Failed to get recommendation: ' + e.message });
    }
    setLoadingRec(false);
  }, []);

  const applyRecommendation = () => {
    if (recommendation?.config) {
      setConfig(prev => ({ ...prev, ...recommendation.config }));
      setRecommendation(null);
    }
  };

  const handleLlmSave = useCallback(async () => {
    if (!llmForm.backend) return;
    setLlmSaving(true);
    setLlmMessage('');
    try {
      const res = await fetch(`${API_BASE}/api/llm/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(llmForm),
      });
      const data = await res.json();
      if (res.ok) {
        setLlmConfig(data.config);
        if (data.warning) {
          setLlmMessage(`Warning: ${data.warning}`);
        } else {
          setLlmMessage('LLM configured and verified successfully');
        }
        setLlmForm({ backend: '', api_key: '', model: '', host: '' });
        // Notify other components (e.g. StrategyAdvisor) that LLM is now available
        window.dispatchEvent(new CustomEvent('llm-configured'));
        // Refresh system status
        fetch(`${API_BASE}/api/system/status`)
          .then(r => r.ok ? r.json() : null)
          .then(d => { if (d) setSystemStatus(d); })
          .catch(() => {});
      } else {
        setLlmMessage(data.error || 'Configuration failed');
      }
    } catch (e) {
      setLlmMessage('Error: ' + e.message);
    }
    setLlmSaving(false);
  }, [llmForm]);

  const isEvolutionMode = mode === 'evolve' || mode === 'novelty';
  const isScaleUpMode = mode === 'scale_up';
  const generationTotal = progress?.total_generations || 0;
  const generationCurrent = progress?.current_generation || 0;
  const programTotal = progress?.total_programs || 0;
  const programCurrent = progress?.current_program || 0;
  const progressStatus = String(progress?.status || '').toLowerCase();
  const isGenerationProgress = generationTotal > 0;

  const pct = isGenerationProgress
    ? (generationTotal > 0
        ? Math.round((generationCurrent / generationTotal) * 100)
        : 0)
    : (programTotal > 0
        ? Math.round((programCurrent / programTotal) * 100)
        : 0);

  const programProgressText = (() => {
    if (programTotal > 0) {
      return `${programCurrent} / ${programTotal} programs (${pct}%)`;
    }
    if (programCurrent > 0) {
      return `${programCurrent} / ? programs (in progress)`;
    }

    if (['investigating', 'validating', 'scale_up'].includes(progressStatus)) {
      return '0 / ? programs (initializing)';
    }

    if (progressStatus === 'resuming') {
      return 'Resuming experiment state...';
    }

    return 'Initializing experiment...';
  })();

  return (
    <div className="card control-panel">
      <div className="card-title">Experiment Control</div>
      {!isRunning && (
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
          Generate and test random computation graphs as potential replacements for transformer
          attention layers. Single mode runs one batch. Continuous mode keeps running and adapts
          strategy between experiments. Evolution and Novelty modes use population-based search
          to breed better architectures over generations.
        </p>
      )}
      {actionError && (
        <div style={{
          marginBottom: 12,
          padding: '8px 10px',
          borderRadius: 6,
          border: '1px solid var(--accent-red)',
          background: 'rgba(248, 81, 73, 0.1)',
          color: 'var(--accent-red)',
          fontSize: 12,
        }}>
          {actionError}
        </div>
      )}

      {prefillSummary && !isRunning && (
        <div style={{
          marginBottom: 12,
          padding: '8px 10px',
          borderRadius: 6,
          border: '1px solid var(--accent-blue)',
          background: 'rgba(88, 166, 255, 0.12)',
          color: 'var(--text-secondary)',
          fontSize: 12,
          lineHeight: 1.5,
        }}>
          <div>
            Prefill applied from {prefillSummary.source.replace('_', ' ')}
            {prefillSummary.campaignTitle ? ` (${prefillSummary.campaignTitle})` : ''}
            : mode set to <strong>{prefillSummary.mode}</strong>.
          </div>
          {prefillSummary.objective && (
            <div><strong>Objective:</strong> {prefillSummary.objective}</div>
          )}
          {prefillSummary.hypothesis && (
            <div><strong>Hypothesis:</strong> {prefillSummary.hypothesis}</div>
          )}
          <button
            className="refresh-btn"
            style={{ fontSize: 10, padding: '2px 8px', marginTop: 6 }}
            onClick={() => setPrefillSummary(null)}
          >
            Dismiss
          </button>
        </div>
      )}

      {/* System Status */}
      {systemStatus && !isRunning && (
        <div className="system-status-badges">
          <span className={`sys-badge ${systemStatus.cuda?.available ? 'pass' : 'fail'}`}>
            {systemStatus.cuda?.available
              ? `CUDA: ${systemStatus.cuda.device_name || 'GPU'}`
              : 'CPU Only'}
          </span>
          <span className={`sys-badge ${systemStatus.llm?.available ? 'pass' : 'warn'}`}>
            {systemStatus.llm?.available
              ? `LLM: ${systemStatus.llm.backend}`
              : 'No LLM'}
          </span>
          <span className="sys-badge info">
            DB: {systemStatus.database?.total_experiments || 0} exp
          </span>
        </div>
      )}

      {!isRunning ? (
        <>
          {/* LLM Configuration */}
          <button
            className="advanced-toggle"
            onClick={() => setShowLlmConfig(!showLlmConfig)}
          >
            {showLlmConfig ? '\u25BC' : '\u25B6'} LLM Configuration
            {llmConfig?.available && (
              <span className="llm-active-indicator"> ({llmConfig.backend})</span>
            )}
          </button>

          {showLlmConfig && (
            <div className="llm-config-section">
              {llmConfig?.available && (
                <div className="llm-current">
                  <span className="sys-badge pass">Active: {llmConfig.backend}</span>
                  {llmConfig.model && <span className="sys-badge info">{llmConfig.model}</span>}
                  {llmConfig.api_key_set && <span className="sys-badge info">Key: {llmConfig.api_key_hint}</span>}
                </div>
              )}
              <div className="config-grid">
                <div className="config-item">
                  <label>Backend</label>
                  <select
                    value={llmForm.backend}
                    onChange={(e) => setLlmForm(prev => ({ ...prev, backend: e.target.value }))}
                  >
                    <option value="">Select...</option>
                    <option value="anthropic">Anthropic (Claude)</option>
                    <option value="openai">OpenAI</option>
                    <option value="ollama">Ollama (Local)</option>
                  </select>
                </div>
                {(llmForm.backend === 'anthropic' || llmForm.backend === 'openai') && (
                  <div className="config-item" style={{ gridColumn: '1 / -1' }}>
                    <label>API Key</label>
                    <input
                      type="password"
                      value={llmForm.api_key}
                      onChange={(e) => setLlmForm(prev => ({ ...prev, api_key: e.target.value }))}
                      placeholder={llmForm.backend === 'anthropic' ? 'sk-ant-...' : 'sk-...'}
                    />
                  </div>
                )}
                {llmForm.backend === 'ollama' && (
                  <div className="config-item">
                    <label>Host URL</label>
                    <input
                      type="text"
                      value={llmForm.host}
                      onChange={(e) => setLlmForm(prev => ({ ...prev, host: e.target.value }))}
                      placeholder="http://localhost:11434"
                    />
                  </div>
                )}
                {llmForm.backend && (
                  <div className="config-item">
                    <label>Model (optional)</label>
                    <input
                      type="text"
                      value={llmForm.model}
                      onChange={(e) => setLlmForm(prev => ({ ...prev, model: e.target.value }))}
                      placeholder={
                        llmForm.backend === 'anthropic' ? 'claude-sonnet-4-5-20250929'
                        : llmForm.backend === 'openai' ? 'gpt-4o'
                        : 'llama3'
                      }
                    />
                  </div>
                )}
              </div>
              {llmForm.backend && (
                <button
                  className="validate-btn"
                  onClick={handleLlmSave}
                  disabled={llmSaving || (!llmForm.api_key && llmForm.backend !== 'ollama')}
                  style={{ marginTop: 8 }}
                >
                  {llmSaving ? 'Configuring...' : 'Configure LLM'}
                </button>
              )}
              {llmMessage && (
                <div className={`llm-message ${llmMessage.includes('success') ? 'pass' : 'fail'}`}>
                  {llmMessage}
                </div>
              )}
            </div>
          )}

          {/* Mode selector */}
          <div className="control-row">
            <label className="control-label">Mode</label>
            <div className="mode-selector">
              <button
                className={`mode-btn ${mode === 'single' ? 'active' : ''}`}
                onClick={() => setMode('single')}
              >
                Single Experiment
              </button>
              <button
                className={`mode-btn ${mode === 'continuous' ? 'active' : ''}`}
                onClick={() => setMode('continuous')}
              >
                Continuous Research
              </button>
              <button
                className={`mode-btn ${mode === 'evolve' ? 'active' : ''}`}
                onClick={() => setMode('evolve')}
              >
                Evolution Search
              </button>
              <button
                className={`mode-btn ${mode === 'novelty' ? 'active' : ''}`}
                onClick={() => setMode('novelty')}
              >
                Novelty Search
              </button>
              <button
                className={`mode-btn ${mode === 'scale_up' ? 'active' : ''}`}
                onClick={() => setMode('scale_up')}
              >
                Scale Up
              </button>
              <button
                className={`mode-btn ${mode === 'investigation' ? 'active' : ''}`}
                onClick={() => setMode('investigation')}
              >
                Investigation
              </button>
              <button
                className={`mode-btn ${mode === 'validation' ? 'active' : ''}`}
                onClick={() => setMode('validation')}
              >
                Validation
              </button>
            </div>
          </div>

          {/* Model Source selector (for single/continuous modes) */}
          {(mode === 'single' || mode === 'continuous') && (
            <div className="control-row">
              <label className="control-label">Model Source</label>
              <div className="mode-selector">
                <button
                  className={`mode-btn ${config.model_source === 'graph_synthesis' ? 'active' : ''}`}
                  onClick={() => updateConfig('model_source', 'graph_synthesis')}
                >
                  Graph Synthesis
                </button>
                <button
                  className={`mode-btn ${config.model_source === 'morphological_box' ? 'active' : ''}`}
                  onClick={() => updateConfig('model_source', 'morphological_box')}
                >
                  Morphological Box
                </button>
                <button
                  className={`mode-btn ${config.model_source === 'mixed' ? 'active' : ''}`}
                  onClick={() => updateConfig('model_source', 'mixed')}
                >
                  Mixed
                </button>
              </div>
              {config.model_source === 'mixed' && (
                <div className="config-grid" style={{ marginTop: 4 }}>
                  <div className="config-item">
                    <label>Morph Ratio</label>
                    <input
                      type="number" min="0" max="1" step="0.1"
                      value={config.morph_ratio}
                      onChange={(e) => updateConfig('morph_ratio', parseFloat(e.target.value) || 0.5)}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Use Synthesized Training toggle */}
          {(mode === 'single' || mode === 'continuous') && (
            <div className="control-row">
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
                <input
                  type="checkbox"
                  checked={config.use_synthesized_training}
                  onChange={(e) => updateConfig('use_synthesized_training', e.target.checked)}
                />
                Use Synthesized Training Programs
              </label>
              {config.use_synthesized_training && (
                <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 8 }}>
                  Candidates trained with random loss/optimizer/curriculum instead of AdamW + cross-entropy
                </span>
              )}
            </div>
          )}

          {/* Hypothesis input */}
          {(mode === 'single' || isEvolutionMode || isScaleUpMode) && (
            <div className="control-row">
              <label className="control-label">Hypothesis (optional)</label>
              <input
                className="control-input"
                type="text"
                value={hypothesis}
                onChange={(e) => setHypothesis(e.target.value)}
                placeholder="Let Aria formulate one automatically..."
              />
            </div>
          )}

          {/* Core config */}
          <div className="config-grid">
            {!isEvolutionMode && (
              <div className="config-item">
                <label>Programs</label>
                <input
                  type="number" min="5" max="500" step="5"
                  value={config.n_programs}
                  onChange={(e) => updateConfig('n_programs', parseInt(e.target.value) || 50)}
                />
              </div>
            )}
            <div className="config-item">
              <label>Dimension</label>
              <select
                value={config.model_dim}
                onChange={(e) => updateConfig('model_dim', parseInt(e.target.value))}
              >
                <option value={64}>64</option>
                <option value={128}>128</option>
                <option value={256}>256</option>
                <option value={512}>512</option>
              </select>
            </div>
            <div className="config-item">
              <label>Layers</label>
              <input
                type="number" min="1" max="12" step="1"
                value={config.n_layers}
                onChange={(e) => updateConfig('n_layers', parseInt(e.target.value) || 4)}
              />
            </div>
            <div className="config-item">
              <label>Device</label>
              <select
                value={config.device}
                onChange={(e) => updateConfig('device', e.target.value)}
              >
                <option value="cuda">CUDA (GPU)</option>
                <option value="cpu">CPU</option>
              </select>
            </div>
          </div>

          {/* Evolution/Novelty config */}
          {isEvolutionMode && (
            <div className="config-grid">
              <div className="config-item">
                <label>Population Size</label>
                <input
                  type="number" min="10" max="200" step="10"
                  value={config.population_size}
                  onChange={(e) => updateConfig('population_size', parseInt(e.target.value) || 50)}
                />
              </div>
              <div className="config-item">
                <label>Generations</label>
                <input
                  type="number" min="5" max="100" step="5"
                  value={config.n_generations}
                  onChange={(e) => updateConfig('n_generations', parseInt(e.target.value) || 20)}
                />
              </div>
              <div className="config-item">
                <label>Tournament Size</label>
                <input
                  type="number" min="2" max="10" step="1"
                  value={config.tournament_size}
                  onChange={(e) => updateConfig('tournament_size', parseInt(e.target.value) || 5)}
                />
              </div>
              <div className="config-item">
                <label>Mutation Rate</label>
                <input
                  type="number" min="0" max="1" step="0.1"
                  value={config.mutation_rate}
                  onChange={(e) => updateConfig('mutation_rate', parseFloat(e.target.value) || 0.7)}
                />
              </div>
              {mode === 'novelty' && (
                <>
                  <div className="config-item">
                    <label>Archive Size</label>
                    <input
                      type="number" min="50" max="1000" step="50"
                      value={config.archive_size}
                      onChange={(e) => updateConfig('archive_size', parseInt(e.target.value) || 200)}
                    />
                  </div>
                  <div className="config-item">
                    <label>K-Nearest</label>
                    <input
                      type="number" min="5" max="50" step="5"
                      value={config.k_nearest}
                      onChange={(e) => updateConfig('k_nearest', parseInt(e.target.value) || 15)}
                    />
                  </div>
                  <div className="config-item">
                    <label>Novelty Weight</label>
                    <input
                      type="number" min="0" max="1" step="0.1"
                      value={config.novelty_weight}
                      onChange={(e) => updateConfig('novelty_weight', parseFloat(e.target.value) || 0.5)}
                    />
                  </div>
                </>
              )}
            </div>
          )}

          {/* Scale-up config */}
          {mode === 'scale_up' && (
            <div className="config-grid" style={{ marginTop: 8 }}>
              <div className="config-item" style={{ gridColumn: '1 / -1' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input
                    type="checkbox"
                    checked={scaleUpUseTop}
                    onChange={(e) => setScaleUpUseTop(e.target.checked)}
                  />
                  Use top N survivors (by loss ratio)
                </label>
              </div>
              {scaleUpUseTop ? (
                <div className="config-item">
                  <label>Top N</label>
                  <input
                    type="number" min="1" max="20" step="1"
                    value={scaleUpTopN}
                    onChange={(e) => setScaleUpTopN(parseInt(e.target.value) || 5)}
                  />
                </div>
              ) : (
                <div className="config-item" style={{ gridColumn: '1 / -1' }}>
                  <label>Result IDs (comma-separated)</label>
                  <input
                    type="text"
                    value={scaleUpIds}
                    onChange={(e) => setScaleUpIds(e.target.value)}
                    placeholder="result-id-1, result-id-2, ..."
                    className="control-input"
                  />
                </div>
              )}
              <div className="config-item">
                <label>Steps</label>
                <input
                  type="number" min="1000" max="50000" step="1000"
                  value={scaleUpSteps}
                  onChange={(e) => setScaleUpSteps(parseInt(e.target.value) || 5000)}
                />
              </div>
              <div className="config-item">
                <label>Batch Size</label>
                <input
                  type="number" min="4" max="16" step="1"
                  value={scaleUpBatchSize}
                  onChange={(e) => setScaleUpBatchSize(parseInt(e.target.value) || 8)}
                />
              </div>
              <div className="config-item">
                <label>Seq Length</label>
                <input
                  type="number" min="256" max="1024" step="128"
                  value={scaleUpSeqLen}
                  onChange={(e) => setScaleUpSeqLen(parseInt(e.target.value) || 512)}
                />
              </div>
            </div>
          )}

          {/* Investigation/Validation config */}
          {(mode === 'investigation' || mode === 'validation') && (
            <div className="config-grid" style={{ marginTop: 8 }}>
              <div className="config-item" style={{ gridColumn: '1 / -1' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input
                    type="checkbox"
                    checked={investUseTop}
                    onChange={(e) => setInvestUseTop(e.target.checked)}
                  />
                  Use top N {mode === 'investigation' ? 'S1 survivors' : 'investigation survivors'}
                </label>
              </div>
              {investUseTop ? (
                <div className="config-item">
                  <label>Top N</label>
                  <input
                    type="number" min="1" max="20" step="1"
                    value={investTopN}
                    onChange={(e) => setInvestTopN(parseInt(e.target.value) || 5)}
                  />
                </div>
              ) : (
                <div className="config-item" style={{ gridColumn: '1 / -1' }}>
                  <label>Result IDs (comma-separated)</label>
                  <input
                    type="text"
                    value={investIds}
                    onChange={(e) => setInvestIds(e.target.value)}
                    placeholder="result-id-1, result-id-2, ..."
                    className="control-input"
                  />
                </div>
              )}
              {mode === 'investigation' && (
                <>
                  <div className="config-item">
                    <label>Steps</label>
                    <input
                      type="number" min="500" max="10000" step="500"
                      value={config.investigation_steps}
                      onChange={(e) => updateConfig('investigation_steps', parseInt(e.target.value) || 2500)}
                    />
                  </div>
                  <div className="config-item">
                    <label>Training Programs</label>
                    <input
                      type="number" min="1" max="10" step="1"
                      value={config.n_training_programs}
                      onChange={(e) => updateConfig('n_training_programs', parseInt(e.target.value) || 3)}
                    />
                  </div>
                </>
              )}
              {mode === 'validation' && (
                <>
                  <div className="config-item">
                    <label>Steps</label>
                    <input
                      type="number" min="1000" max="50000" step="1000"
                      value={config.validation_steps}
                      onChange={(e) => updateConfig('validation_steps', parseInt(e.target.value) || 10000)}
                    />
                  </div>
                  <div className="config-item">
                    <label>Seeds</label>
                    <input
                      type="number" min="1" max="10" step="1"
                      value={config.validation_n_seeds}
                      onChange={(e) => updateConfig('validation_n_seeds', parseInt(e.target.value) || 3)}
                    />
                  </div>
                  <div className="config-item">
                    <label>Batch Size</label>
                    <input
                      type="number" min="4" max="16" step="1"
                      value={config.validation_batch_size}
                      onChange={(e) => updateConfig('validation_batch_size', parseInt(e.target.value) || 8)}
                    />
                  </div>
                </>
              )}
            </div>
          )}

          {/* Advanced toggle */}
          <button
            className="advanced-toggle"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? '\u25BC' : '\u25B6'} Advanced Settings
          </button>

          {showAdvanced && (
            <div className="config-grid">
              <div className="config-item">
                <label>Stage 1 Steps</label>
                <input
                  type="number" min="50" max="2000" step="50"
                  value={config.stage1_steps}
                  onChange={(e) => updateConfig('stage1_steps', parseInt(e.target.value) || 500)}
                />
              </div>
              <div className="config-item">
                <label>Max Depth</label>
                <input
                  type="number" min="4" max="20" step="1"
                  value={config.max_depth}
                  onChange={(e) => updateConfig('max_depth', parseInt(e.target.value) || 10)}
                />
              </div>
              <div className="config-item">
                <label>Max Ops</label>
                <input
                  type="number" min="4" max="32" step="1"
                  value={config.max_ops}
                  onChange={(e) => updateConfig('max_ops', parseInt(e.target.value) || 16)}
                />
              </div>
              <div className="config-item">
                <label>Math Weight</label>
                <input
                  type="number" min="0" max="10" step="0.5"
                  value={config.math_space_weight}
                  onChange={(e) => updateConfig('math_space_weight', parseFloat(e.target.value) || 2.0)}
                />
              </div>
              <div className="config-item">
                <label>Residual Prob</label>
                <input
                  type="number" min="0" max="1" step="0.1"
                  value={config.residual_prob}
                  onChange={(e) => updateConfig('residual_prob', parseFloat(e.target.value) || 0.7)}
                />
              </div>
              <div className="config-item">
                <label>Vocab Size</label>
                <select
                  value={config.vocab_size}
                  onChange={(e) => updateConfig('vocab_size', parseInt(e.target.value))}
                >
                  <option value={1000}>1K (fast)</option>
                  <option value={8000}>8K</option>
                  <option value={32000}>32K</option>
                </select>
              </div>
              {mode === 'continuous' && (
                <>
                  <div className="config-item">
                    <label>Max Experiments</label>
                    <input
                      type="number" min="1" max="1000" step="1"
                      value={config.max_experiments}
                      onChange={(e) => updateConfig('max_experiments', parseInt(e.target.value) || 100)}
                    />
                  </div>
                  <div className="config-item">
                    <label>Time Limit (min)</label>
                    <input
                      type="number" min="0" max="1440" step="5"
                      value={config.max_time_minutes}
                      onChange={(e) => updateConfig('max_time_minutes', parseInt(e.target.value) || 0)}
                      placeholder="0 = no limit"
                    />
                  </div>
                  <div className="config-item">
                    <label>Cost Limit ($)</label>
                    <input
                      type="number" min="0" max="100" step="0.5"
                      value={config.max_cost_dollars}
                      onChange={(e) => updateConfig('max_cost_dollars', parseFloat(e.target.value) || 0)}
                      placeholder="0 = no limit"
                    />
                  </div>
                </>
              )}
              {/* Automation settings (always visible in advanced) */}
              <div className="config-item" style={{ gridColumn: '1 / -1', borderTop: '1px solid var(--border)', paddingTop: 8, marginTop: 4 }}>
                <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Automation</label>
              </div>
              <div className="config-item" style={{ gridColumn: '1 / -1' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input
                    type="checkbox"
                    checked={config.auto_scale_up}
                    onChange={(e) => updateConfig('auto_scale_up', e.target.checked)}
                  />
                  Auto scale-up when S1 survivors found
                </label>
              </div>
              {config.auto_scale_up && (
                <>
                  <div className="config-item">
                    <label>Min Survivors</label>
                    <input
                      type="number" min="1" max="20" step="1"
                      value={config.auto_scale_up_min_survivors}
                      onChange={(e) => updateConfig('auto_scale_up_min_survivors', parseInt(e.target.value) || 3)}
                    />
                  </div>
                  <div className="config-item">
                    <label>Top N to Scale</label>
                    <input
                      type="number" min="1" max="20" step="1"
                      value={config.auto_scale_up_top_n}
                      onChange={(e) => updateConfig('auto_scale_up_top_n', parseInt(e.target.value) || 5)}
                    />
                  </div>
                </>
              )}
              <div className="config-item" style={{ gridColumn: '1 / -1' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input
                    type="checkbox"
                    checked={config.auto_report}
                    onChange={(e) => updateConfig('auto_report', e.target.checked)}
                  />
                  Auto-generate research reports
                </label>
              </div>
              {config.auto_report && mode === 'continuous' && (
                <div className="config-item">
                  <label>Report Every N Exp</label>
                  <input
                    type="number" min="1" max="50" step="1"
                    value={config.auto_report_every_n}
                    onChange={(e) => updateConfig('auto_report_every_n', parseInt(e.target.value) || 5)}
                  />
                </div>
              )}
              {/* Auto-escalation settings */}
              <div className="config-item" style={{ gridColumn: '1 / -1', borderTop: '1px solid var(--border)', paddingTop: 8, marginTop: 4 }}>
                <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Auto-Escalation Pipeline</label>
              </div>
              <div className="config-item" style={{ gridColumn: '1 / -1' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input
                    type="checkbox"
                    checked={config.auto_investigate}
                    onChange={(e) => updateConfig('auto_investigate', e.target.checked)}
                  />
                  Auto-investigate when screening survivors found
                </label>
              </div>
              {config.auto_investigate && (
                <>
                  <div className="config-item">
                    <label>Min Survivors</label>
                    <input
                      type="number" min="1" max="20" step="1"
                      value={config.auto_investigate_min_survivors}
                      onChange={(e) => updateConfig('auto_investigate_min_survivors', parseInt(e.target.value) || 2)}
                    />
                  </div>
                  <div className="config-item">
                    <label>Top N to Investigate</label>
                    <input
                      type="number" min="1" max="20" step="1"
                      value={config.auto_investigate_top_n}
                      onChange={(e) => updateConfig('auto_investigate_top_n', parseInt(e.target.value) || 5)}
                    />
                  </div>
                </>
              )}
              <div className="config-item" style={{ gridColumn: '1 / -1' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input
                    type="checkbox"
                    checked={config.auto_validate}
                    onChange={(e) => updateConfig('auto_validate', e.target.checked)}
                  />
                  Auto-validate when investigation passes
                </label>
              </div>
              {config.auto_validate && (
                <>
                  <div className="config-item">
                    <label>Min Robustness</label>
                    <input
                      type="number" min="0" max="1" step="0.1"
                      value={config.auto_validate_min_robustness}
                      onChange={(e) => updateConfig('auto_validate_min_robustness', parseFloat(e.target.value) || 0.5)}
                    />
                  </div>
                  <div className="config-item">
                    <label>Top N to Validate</label>
                    <input
                      type="number" min="1" max="10" step="1"
                      value={config.auto_validate_top_n}
                      onChange={(e) => updateConfig('auto_validate_top_n', parseInt(e.target.value) || 3)}
                    />
                  </div>
                </>
              )}
            </div>
          )}

          {/* Validate + Ask Aria row */}
          <div className="control-actions-row">
            <button
              className="validate-btn"
              onClick={handleValidate}
              disabled={validating}
            >
              {validating ? 'Validating...' : 'Validate Pipeline'}
            </button>
            <button
              className="ask-aria-btn"
              onClick={handleAskAria}
              disabled={loadingRec}
            >
              {loadingRec ? 'Thinking...' : 'Ask Aria'}
            </button>
          </div>

          {/* Validation result */}
          {validationResult && (
            <div className={`validation-result ${validationResult.healthy ? 'pass' : 'fail'}`}>
              <strong>{validationResult.healthy ? 'Pipeline Healthy' : 'Pipeline Issues'}</strong>
              <span> — Generated: {validationResult.generated}, Compiled: {validationResult.compiled}, S0: {validationResult.passed_s0}</span>
              {validationResult.errors?.length > 0 && (
                <div className="validation-errors">
                  {validationResult.errors.slice(0, 3).map((err, i) => (
                    <div key={i} className="validation-error">{err}</div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Aria recommendation */}
          {recommendation && (
            <div className="recommendation-section">
              <div className="recommendation-header">
                <strong>Aria's Recommendation</strong>
                {recommendation.confidence != null && (
                  <span className="rec-confidence">
                    Confidence: {(recommendation.confidence * 100).toFixed(0)}%
                  </span>
                )}
              </div>
              <p className="recommendation-reasoning">{recommendation.reasoning}</p>
              {recommendation.config && Object.keys(recommendation.config).length > 0 && (
                <>
                  <div className="recommendation-config">
                    {Object.entries(recommendation.config).map(([k, v]) => (
                      <span key={k} className="rec-param">{k}: {v}</span>
                    ))}
                  </div>
                  <button className="apply-rec-btn" onClick={applyRecommendation}>
                    Apply Suggestion
                  </button>
                </>
              )}
            </div>
          )}

          {/* Start button */}
          {startLocked && (
            <div style={{
              marginTop: 6,
              marginBottom: 8,
              padding: '8px 10px',
              borderRadius: 6,
              border: '1px solid var(--accent-yellow)',
              background: 'rgba(210, 153, 34, 0.12)',
              color: 'var(--text-secondary)',
              fontSize: 12,
              lineHeight: 1.5,
            }}>
              {startLockReason || 'Start is locked by Strategy Advisor. Use the primary recommendation or override from Overview advanced setup.'}
            </div>
          )}
          <button className="start-btn" onClick={handleStart} disabled={startLocked}>
            {mode === 'continuous' ? 'Start Continuous Research'
              : mode === 'evolve' ? 'Start Evolution Search'
              : mode === 'novelty' ? 'Start Novelty Search'
              : mode === 'scale_up' ? 'Start Scale-Up Validation'
              : mode === 'investigation' ? 'Start Investigation'
              : mode === 'validation' ? 'Start Validation'
              : 'Run Experiment'}
          </button>
        </>
      ) : (
        /* Running state - show progress */
        <div className="experiment-progress">
          <div className="progress-header">
            <span className="progress-status">
              <span className="pulse-dot"></span>
              {progress?.status || 'Running'}
            </span>
            <span className="progress-id">{progress?.experiment_id?.slice(0, 8)}</span>
          </div>

          {/* Hypothesis critique */}
          {progress?.hypothesis_critique && (
            <HypothesisCritique critique={progress.hypothesis_critique} />
          )}

          {/* Progress bar */}
          <div className="progress-bar-container">
            <div className="progress-bar" style={{ width: `${pct}%` }}></div>
            <span className="progress-text">
              {isGenerationProgress
                ? `Gen ${progress?.current_generation || 0} / ${progress?.total_generations || 0} (${pct}%)`
                : programProgressText
              }
            </span>
          </div>

          {/* Evolution-specific stats */}
          {isGenerationProgress ? (
            <div className="live-stats">
              <div className="live-stat">
                <span className="live-stat-label">Best Fitness</span>
                <span className="live-stat-value green">
                  {progress?.best_fitness != null ? progress.best_fitness.toFixed(3) : '—'}
                </span>
              </div>
              <div className="live-stat">
                <span className="live-stat-label">Avg Fitness</span>
                <span className="live-stat-value blue">
                  {progress?.avg_fitness != null ? progress.avg_fitness.toFixed(3) : '—'}
                </span>
              </div>
              <div className="live-stat">
                <span className="live-stat-label">Stage 1</span>
                <span className="live-stat-value purple">{progress?.stage1_passed || 0}</span>
              </div>
              {progress?.archive_size > 0 && (
                <div className="live-stat">
                  <span className="live-stat-label">Archive</span>
                  <span className="live-stat-value yellow">{progress.archive_size}</span>
                </div>
              )}
            </div>
          ) : (
            <>
              {/* Stage indicator */}
              <div className="stage-indicator">
                <span className="stage-label">Current stage:</span>
                <span className="badge running">{progress?.current_stage || '...'}</span>
                {progress?.current_fingerprint && (
                  <span className="fingerprint-label">{progress.current_fingerprint}</span>
                )}
              </div>

              {/* Live stats */}
              <div className="live-stats">
                <div className="live-stat">
                  <span className="live-stat-label">Stage 0</span>
                  <span className="live-stat-value green">{progress?.stage0_passed || 0}</span>
                </div>
                <div className="live-stat">
                  <span className="live-stat-label">Stage 0.5</span>
                  <span className="live-stat-value blue">{progress?.stage05_passed || 0}</span>
                </div>
                <div className="live-stat">
                  <span className="live-stat-label">Stage 1</span>
                  <span className="live-stat-value purple">{progress?.stage1_passed || 0}</span>
                </div>
                <div className="live-stat">
                  <span className="live-stat-label">Novel</span>
                  <span className="live-stat-value yellow">{progress?.novel_count || 0}</span>
                </div>
              </div>
            </>
          )}

          {/* Best scores */}
          {(progress?.best_loss_ratio || progress?.best_novelty) && (
            <div className="best-scores">
              {progress.best_loss_ratio && (
                <span>Best loss ratio: <strong>{progress.best_loss_ratio.toFixed(4)}</strong></span>
              )}
              {progress.best_novelty && (
                <span>Best novelty: <strong>{progress.best_novelty.toFixed(3)}</strong></span>
              )}
            </div>
          )}

          {/* Elapsed time + cost */}
          <div className="elapsed-time">
            Elapsed: {formatElapsed(progress?.elapsed_seconds || 0)}
            {progress?.estimated_cost > 0 && (
              <span style={{ marginLeft: 12 }}>
                Est. cost: <strong>${progress.estimated_cost.toFixed(2)}</strong>
              </span>
            )}
            {progress?.total_tokens > 0 && (
              <span style={{ marginLeft: 12, color: 'var(--text-muted)' }}>
                ({(progress.total_tokens / 1000).toFixed(0)}K tokens)
              </span>
            )}
          </div>

          {/* Aria's message */}
          {progress?.aria_message && (
            <div className="aria-live-message">
              "{progress.aria_message}"
            </div>
          )}

          {/* Stop button */}
          <button className="stop-btn" onClick={onStop}>
            Stop Experiment
          </button>
        </div>
      )}
    </div>
  );
}

function formatElapsed(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  if (mins < 60) return `${mins}m ${secs}s`;
  const hrs = Math.floor(mins / 60);
  return `${hrs}h ${mins % 60}m`;
}

export default ControlPanel;
