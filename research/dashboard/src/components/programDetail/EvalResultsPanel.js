import React from 'react';
import { apiCall, postJson } from '../../services/apiService';

const LONG_ACTION_TIMEOUT_MS = 120000;

function EvalResultsPanel({
  program, leaderboardEntry, resultId, dispatch, state,
  onActionComplete, onClose,
  canInvestigate, canValidate, canConfirm, alreadyInvestigated, alreadyValidated,
}) {
  const {
    scaleUpOpen, scaleUpConfig, scaleUpStarting,
    manualRunOpen, manualRunStarting, manualRunConfig,
    backfillRunning, backfillResult, lossBackfillRunning, lossBackfillResult,
    actionStarting, actionError, overrideIneligible,
  } = state;

  const fmt = (v, d = 4) => v != null ? Number(v).toFixed(d) : '--';

  const investigateDisabled = actionStarting === 'investigate';
  const validateDisabled = actionStarting === 'validate';
  const confirmDisabled = scaleUpStarting || actionStarting === 'confirmation';
  const investigateTitle = canInvestigate
    ? 'Deep study with multiple training programs'
    : 'Deep study with multiple training programs. This run will use override if needed.';
  const validateTitle = canValidate
    ? 'Publication-grade multi-seed validation'
    : 'Publication-grade multi-seed validation. This run will use override if needed.';

  if (!program.stage1_passed) return null;

  return (
    <>
      {/* Champion Confirmation */}
      <div style={{
        padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6,
        border: '1px solid var(--border)',
      }}>
        {!scaleUpOpen ? (
          <button
            className="start-btn"
            onClick={() => dispatch({ type: 'SET_MODAL', payload: { scaleUpOpen: true } })}
            disabled={!canConfirm && !overrideIneligible}
            style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(255, 184, 108, 0.14)', border: '1px solid rgba(255, 184, 108, 0.45)', color: 'var(--score-elite)', opacity: (!canConfirm && !overrideIneligible) ? 0.55 : 1 }}
            title={canConfirm ? 'Run post-validation champion confirmation' : 'Confirmation requires a passed validation result'}
          >
            Champion Confirmation
          </button>
        ) : (
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--text-secondary)' }}>
              Champion Confirmation Configuration
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 8 }}>
              <div>
                <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Steps</label>
                <input type="number" min="1000" max="50000" step="1000"
                  value={scaleUpConfig.steps}
                  onChange={e => dispatch({ type: 'SET_MODAL', payload: { scaleUpConfig: { ...scaleUpConfig, steps: parseInt(e.target.value) || 40000 } } })}
                  style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                />
              </div>
              <div>
                <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Batch Size</label>
                <input type="number" min="4" max="16" step="1"
                  value={scaleUpConfig.batch_size}
                  onChange={e => dispatch({ type: 'SET_MODAL', payload: { scaleUpConfig: { ...scaleUpConfig, batch_size: parseInt(e.target.value) || 8 } } })}
                  style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                />
              </div>
              <div>
                <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Seq Length</label>
                <input type="number" min="256" max="1024" step="128"
                  value={scaleUpConfig.seq_len}
                  onChange={e => dispatch({ type: 'SET_MODAL', payload: { scaleUpConfig: { ...scaleUpConfig, seq_len: parseInt(e.target.value) || 512 } } })}
                  style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                />
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                className="start-btn"
                disabled={confirmDisabled}
                onClick={async () => {
                  dispatch({ type: 'SET_MODAL', payload: { scaleUpStarting: true } });
                  try {
                    const forceRun = Boolean(overrideIneligible || !canConfirm);
                    dispatch({ type: 'SET_ACTION', payload: { starting: 'confirmation', error: null } });
                    const res = await postJson('/api/experiments/start', {
                      mode: 'confirmation',
                      result_ids: [resultId],
                      scale_up_steps: scaleUpConfig.steps,
                      scale_up_batch_size: scaleUpConfig.batch_size,
                      scale_up_seq_len: scaleUpConfig.seq_len,
                      force: forceRun,
                      override_ineligible: forceRun,
                      preflight_override: true,
                      enforce_preflight: true,
                    }, { timeoutMs: LONG_ACTION_TIMEOUT_MS });
                    if (!res.ok) {
                      const err = await res.json();
                      dispatch({ type: 'SET_ACTION', payload: { starting: null, error: err.error || 'Failed to start champion confirmation' } });
                    } else {
                      dispatch({ type: 'SET_ACTION', payload: { starting: null, error: null } });
                      dispatch({ type: 'SET_MODAL', payload: { scaleUpOpen: false } });
                      if (onActionComplete) onActionComplete();
                      onClose();
                    }
                  } catch (e) {
                    dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
                  }
                  dispatch({ type: 'SET_MODAL', payload: { scaleUpStarting: false } });
                }}
                style={{ padding: '6px 16px', fontSize: 12 }}
              >
                {scaleUpStarting ? 'Starting...' : 'Start Confirmation'}
              </button>
              <button
                className="refresh-btn"
                onClick={() => dispatch({ type: 'SET_MODAL', payload: { scaleUpOpen: false } })}
                style={{ padding: '6px 12px', fontSize: 12 }}
              >
                Cancel
              </button>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
              Post-validation run: {scaleUpConfig.steps} steps with batch={scaleUpConfig.batch_size}, seq={scaleUpConfig.seq_len}
            </div>
          </div>
        )}
      </div>
      {/* Manual Run */}
      <div style={{
        padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6,
        border: '1px solid var(--border)',
      }}>
        {!manualRunOpen ? (
          <button
            className="start-btn"
            onClick={() => dispatch({ type: 'SET_MODAL', payload: { manualRunOpen: true } })}
            style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(210, 153, 34, 0.15)', border: '1px solid rgba(210, 153, 34, 0.4)', color: 'var(--accent-yellow)' }}
          >
            Manual Training Run
          </button>
        ) : (
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--text-secondary)' }}>
              Manual Training Configuration
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 8, marginBottom: 8 }}>
              <div>
                <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Steps</label>
                <input type="number" min="500" max="50000" step="500"
                  value={manualRunConfig.steps}
                  onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, steps: parseInt(e.target.value) || 2500 } } })}
                  style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                />
              </div>
              <div>
                <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Batch Size</label>
                <input type="number" min="1" max="32" step="1"
                  value={manualRunConfig.batch_size}
                  onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, batch_size: parseInt(e.target.value) || 4 } } })}
                  style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                />
              </div>
              <div>
                <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Seq Length</label>
                <input type="number" min="64" max="2048" step="64"
                  value={manualRunConfig.seq_len}
                  onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, seq_len: parseInt(e.target.value) || 256 } } })}
                  style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                />
              </div>
              <div>
                <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Training Programs</label>
                <input type="number" min="1" max="10" step="1"
                  value={manualRunConfig.n_training_programs}
                  onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, n_training_programs: parseInt(e.target.value) || 3 } } })}
                  style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 8 }}>
              <div>
                <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Data Source</label>
                <select
                  value={manualRunConfig.data_source}
                  onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, data_source: e.target.value } } })}
                  style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                >
                  <option value="corpus">Corpus</option>
                  <option value="random">Random</option>
                  <option value="huggingface">HuggingFace</option>
                </select>
              </div>
              <div>
                <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Tokenizer</label>
                <select
                  value={manualRunConfig.tokenizer}
                  onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, tokenizer: e.target.value } } })}
                  style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                >
                  <option value="byte">Byte (1 byte = 1 token)</option>
                  <option value="tiktoken">BPE / tiktoken (GPT-2, ~4x context)</option>
                  <option value="whitespace">Whitespace hash</option>
                </select>
              </div>
              {manualRunConfig.data_source === 'huggingface' && (
                <>
                  <div>
                    <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>HF Dataset</label>
                    <input type="text"
                      value={manualRunConfig.hf_dataset}
                      onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, hf_dataset: e.target.value } } })}
                      placeholder="roneneldan/TinyStories"
                      style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                    />
                  </div>
                  <div>
                    <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>HF Subset</label>
                    <input type="text"
                      value={manualRunConfig.hf_subset}
                      onChange={e => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, hf_subset: e.target.value } } })}
                      placeholder="(optional)"
                      style={{ width: '100%', padding: '4px 6px', fontSize: 12 }}
                    />
                  </div>
                </>
              )}
            </div>
            <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
              <button className="refresh-btn" style={{ padding: '3px 8px', fontSize: 11 }}
                onClick={() => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, steps: 1000, batch_size: 4, n_training_programs: 1, seq_len: 256 } } })}>
                Quick
              </button>
              <button className="refresh-btn" style={{ padding: '3px 8px', fontSize: 11 }}
                onClick={() => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, steps: 2500, batch_size: 4, n_training_programs: 3, seq_len: 256 } } })}>
                Standard
              </button>
              <button className="refresh-btn" style={{ padding: '3px 8px', fontSize: 11 }}
                onClick={() => dispatch({ type: 'SET_MODAL', payload: { manualRunConfig: { ...manualRunConfig, steps: 5000, batch_size: 8, n_training_programs: 5, seq_len: 512 } } })}>
                Deep
              </button>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                className="start-btn"
                disabled={manualRunStarting}
                onClick={async () => {
                  dispatch({ type: 'SET_MODAL', payload: { manualRunStarting: true } });
                  try {
                    dispatch({ type: 'SET_ACTION', payload: { starting: null, error: null } });
                    const body = {
                      mode: 'investigation',
                      force: true,
                      result_ids: [resultId],
                      n_training_programs: manualRunConfig.n_training_programs,
                      investigation_steps: manualRunConfig.steps,
                      investigation_batch_size: manualRunConfig.batch_size,
                      max_seq_len: manualRunConfig.seq_len,
                      data_mode: manualRunConfig.data_source,
                      preflight_override: true,
                      enforce_preflight: true,
                    };
                    if (manualRunConfig.tokenizer && manualRunConfig.tokenizer !== 'byte') {
                      body.tokenizer_mode = manualRunConfig.tokenizer;
                    }
                    if (manualRunConfig.data_source === 'huggingface') {
                      body.hf_dataset = manualRunConfig.hf_dataset;
                      body.hf_subset = manualRunConfig.hf_subset;
                    }
                    const res = await postJson('/api/experiments/start', body, { timeoutMs: LONG_ACTION_TIMEOUT_MS });
                    if (!res.ok) {
                      const err = await res.json();
                      dispatch({ type: 'SET_ACTION', payload: { starting: null, error: err.error || 'Failed to start manual run' } });
                    } else {
                      dispatch({ type: 'SET_MODAL', payload: { manualRunOpen: false } });
                      if (onActionComplete) onActionComplete();
                      onClose();
                    }
                  } catch (e) {
                    dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
                  }
                  dispatch({ type: 'SET_MODAL', payload: { manualRunStarting: false } });
                }}
                style={{ padding: '6px 16px', fontSize: 12 }}
              >
                {manualRunStarting ? 'Starting...' : 'Launch Manual Run'}
              </button>
              <button
                className="refresh-btn"
                onClick={() => dispatch({ type: 'SET_MODAL', payload: { manualRunOpen: false } })}
                style={{ padding: '6px 12px', fontSize: 12 }}
              >
                Cancel
              </button>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
              {manualRunConfig.n_training_programs} program(s), {manualRunConfig.steps} steps, batch={manualRunConfig.batch_size}, seq={manualRunConfig.seq_len}, data={manualRunConfig.data_source}, tok={manualRunConfig.tokenizer}
              {manualRunConfig.data_source === 'huggingface' && manualRunConfig.hf_dataset && ` (${manualRunConfig.hf_dataset})`}
            </div>
          </div>
        )}
      </div>
      {/* Backfill Missing Metrics */}
      <BackfillSection
        program={program}
        leaderboardEntry={leaderboardEntry}
        resultId={resultId}
        dispatch={dispatch}
        backfillRunning={backfillRunning}
        backfillResult={backfillResult}
        lossBackfillRunning={lossBackfillRunning}
        lossBackfillResult={lossBackfillResult}
        fmt={fmt}
      />
      {/* Investigate / Validate */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <input
            type="checkbox"
            checked={overrideIneligible}
            onChange={(e) => dispatch({ type: 'SET_UI', payload: { overrideIneligible: Boolean(e.target.checked) } })}
          />
          Override ineligible guardrails
        </label>
        <button
          className="start-btn"
          disabled={investigateDisabled}
          onClick={async () => {
            const forceRun = Boolean(overrideIneligible || !canInvestigate);
            dispatch({ type: 'SET_ACTION', payload: { starting: 'investigate', error: null } });
            try {
              const res = await postJson('/api/experiments/start', {
                mode: 'investigation',
                result_ids: [resultId],
                force: forceRun,
                override_ineligible: forceRun,
                preflight_override: true,
                enforce_preflight: true,
              }, { timeoutMs: LONG_ACTION_TIMEOUT_MS });
              if (!res.ok) {
                const err = await res.json();
                dispatch({ type: 'SET_ACTION', payload: { starting: null, error: err.error || 'Failed to start investigation' } });
              } else {
                dispatch({ type: 'SET_ACTION', payload: { starting: null, error: null } });
                if (onActionComplete) onActionComplete();
                onClose();
              }
            } catch (e) {
              dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
            }
          }}
          style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(63, 185, 80, 0.15)', border: '1px solid rgba(63, 185, 80, 0.4)', color: 'var(--accent-green)' }}
          title={investigateTitle}
        >
          {actionStarting === 'investigate' ? 'Starting...' : 'Investigate'}
        </button>
        {alreadyInvestigated && !overrideIneligible && (
          <span style={{
            fontSize: 11,
            padding: '4px 8px',
            borderRadius: 4,
            background: 'rgba(210,153,34,0.12)',
            color: 'var(--accent-yellow)',
          }} title="Candidate already has investigation evidence; wait for changed conditions before re-investigating">
            Already investigated
          </span>
        )}
        <button
          className="start-btn"
          disabled={validateDisabled}
          onClick={async () => {
            const forceRun = Boolean(overrideIneligible || !canValidate);
            dispatch({ type: 'SET_ACTION', payload: { starting: 'validate', error: null } });
            try {
              const res = await postJson('/api/experiments/start', {
                mode: 'validation',
                result_ids: [resultId],
                force: forceRun,
                override_ineligible: forceRun,
                preflight_override: true,
                enforce_preflight: true,
              }, { timeoutMs: LONG_ACTION_TIMEOUT_MS });
              if (!res.ok) {
                const err = await res.json();
                dispatch({ type: 'SET_ACTION', payload: { starting: null, error: err.error || 'Failed to start validation' } });
              } else {
                dispatch({ type: 'SET_ACTION', payload: { starting: null, error: null } });
                if (onActionComplete) onActionComplete();
                onClose();
              }
            } catch (e) {
              dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
            }
          }}
          style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(188, 140, 255, 0.15)', border: '1px solid rgba(188, 140, 255, 0.4)', color: 'var(--accent-purple)' }}
          title={validateTitle}
        >
          {actionStarting === 'validate' ? 'Starting...' : 'Validate'}
        </button>
        {alreadyValidated && !overrideIneligible && (
          <span style={{
            fontSize: 11,
            padding: '4px 8px',
            borderRadius: 4,
            background: 'rgba(88,166,255,0.12)',
            color: 'var(--accent-blue)',
          }} title="Candidate already has validation evidence. Enable override to rerun validation.">
            Already validated
          </span>
        )}
        {actionError && (
          <span style={{ fontSize: 11, color: 'var(--accent-red)', alignSelf: 'center' }}>
            {actionError}
          </span>
        )}
      </div>
    </>
  );
}

function BackfillSection({ program, leaderboardEntry, resultId, dispatch, backfillRunning, backfillResult, lossBackfillRunning, lossBackfillResult }) {
  const trustLabel = String(program?.trust_label || leaderboardEntry?.trust_label || '').trim().toLowerCase();
  const canPromoteScreening = Boolean(resultId) && trustLabel !== 'candidate_screening' && trustLabel !== 'candidate_grade' && trustLabel !== 'reference';
  const metrics = [
    { key: 'novelty_score', label: 'Novelty' },
    { key: 'fp_jacobian_spectral_norm', label: 'Spectral Norm' },
    { key: 'fp_interaction_locality', label: 'Locality' },
    { key: 'fp_interaction_sparsity', label: 'Sparsity' },
    { key: 'fp_isotropy', label: 'Isotropy' },
    { key: 'fp_rank_ratio', label: 'Rank Ratio' },
    { key: 'fp_sensitivity_uniformity', label: 'Sensitivity' },
  ];
  const missing = metrics.filter(m => program[m.key] == null);
  const lbMissing = leaderboardEntry ? [
    { key: 'robustness_noise_score', label: 'Noise Robustness' },
    { key: 'quant_int8_retention', label: 'INT8 Quantization' },
    { key: 'init_sensitivity_std', label: 'Init Sensitivity' },
    { key: 'param_efficiency', label: 'Param Efficiency' },
  ].filter(m => leaderboardEntry[m.key] == null) : [];
  const allMissing = [...missing, ...lbMissing];
  return (
    <div style={{
      padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6,
      border: '1px solid var(--border)',
    }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
        {allMissing.length > 0
          ? `Missing: ${allMissing.map(m => m.label).join(', ')}`
          : 'On-demand repair tools. Use these to rescreen, recompute metrics, or recover missing losses for this row.'}
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        {canPromoteScreening && (
          <button
            className="start-btn"
            onClick={async () => {
              dispatch({ type: 'SET_BACKFILL', payload: { backfillRunning: true, backfillResult: null } });
              try {
                dispatch({ type: 'SET_ACTION', payload: { starting: null, error: null } });
                const res = await postJson(`/api/programs/${resultId}/promote-screening`);
                const data = await res.json().catch(() => ({}));
                if (!res.ok) {
                  dispatch({ type: 'SET_ACTION', payload: { starting: null, error: data.error || 'Promote to screening failed' } });
                  dispatch({ type: 'SET_BACKFILL', payload: { backfillRunning: false, backfillResult: { status: 'error' } } });
                } else {
                  dispatch({
                    type: 'SET_BACKFILL',
                    payload: {
                      backfillRunning: false,
                      backfillResult: { status: 'ok', mode: 'promote_screening' },
                    }
                  });
                }
              } catch (e) {
                dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
                dispatch({ type: 'SET_BACKFILL', payload: { backfillRunning: false, backfillResult: { status: 'error' } } });
              }
            }}
            style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(63, 185, 80, 0.15)', border: '1px solid rgba(63, 185, 80, 0.4)', color: 'var(--accent-green)' }}
          >
            {backfillRunning ? 'Promoting...' : 'Promote to Screening'}
          </button>
        )}
        <button
          className="start-btn"
          onClick={async () => {
            dispatch({ type: 'SET_BACKFILL', payload: { backfillRunning: true, backfillResult: null } });
            try {
              dispatch({ type: 'SET_ACTION', payload: { starting: null, error: null } });
              const res = await postJson(`/api/programs/${resultId}/rescreen`, { device: 'cuda', fast: true, repeat_per_source: 1 }, { timeoutMs: LONG_ACTION_TIMEOUT_MS });
              if (!res.ok) {
                const err = await res.json();
                dispatch({ type: 'SET_ACTION', payload: { starting: null, error: err.error || 'Rescreen failed' } });
                dispatch({ type: 'SET_BACKFILL', payload: { backfillRunning: false, backfillResult: { status: 'error' } } });
              } else {
                const data = await res.json();
                dispatch({
                  type: 'SET_BACKFILL',
                  payload: {
                    backfillRunning: false,
                    backfillResult: { status: 'ok', mode: 'rescreen', experiment_id: data.experiment_id },
                  }
                });
              }
            } catch (e) {
              dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
              dispatch({ type: 'SET_BACKFILL', payload: { backfillRunning: false, backfillResult: { status: 'error' } } });
            }
          }}
          style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(88, 166, 255, 0.15)', border: '1px solid rgba(88, 166, 255, 0.4)', color: 'var(--accent-blue)' }}
        >
          {backfillRunning ? 'Starting...' : 'Rescreen'}
        </button>
        <button
          className="start-btn"
          disabled={backfillRunning}
          onClick={async () => {
            dispatch({ type: 'SET_BACKFILL', payload: { backfillRunning: true, backfillResult: null } });
            try {
              dispatch({ type: 'SET_ACTION', payload: { starting: null, error: null } });
              const res = await postJson(`/api/programs/${resultId}/backfill-metrics`, { device: 'cpu' }, { timeoutMs: LONG_ACTION_TIMEOUT_MS });
              if (!res.ok) {
                const err = await res.json();
                dispatch({ type: 'SET_ACTION', payload: { starting: null, error: err.error || 'Backfill failed' } });
                dispatch({ type: 'SET_BACKFILL', payload: { backfillRunning: false, backfillResult: { status: 'error' } } });
              } else {
                const data = await res.json();
                dispatch({ type: 'SET_BACKFILL', payload: { backfillRunning: false, backfillResult: data.backfill || { status: 'ok' } } });
              }
            } catch (e) {
              dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
              dispatch({ type: 'SET_BACKFILL', payload: { backfillRunning: false, backfillResult: { status: 'error' } } });
            }
          }}
          style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(139, 92, 246, 0.15)', border: '1px solid rgba(139, 92, 246, 0.4)', color: '#a78bfa' }}
        >
          {backfillRunning ? 'Computing...' : 'Recompute Missing Metrics'}
        </button>
        {backfillResult && backfillResult.status === 'ok' && (
          <span style={{ fontSize: 11, color: 'var(--accent-green)' }}>
            {backfillResult.mode === 'promote_screening'
              ? 'Promoted to screening candidate pool'
              : backfillResult.mode === 'rescreen'
              ? `Rescreen queued${backfillResult.experiment_id ? ` (${String(backfillResult.experiment_id).slice(0, 8)})` : ''}`
              : 'Done — reload to see updates'}
          </span>
        )}
        {backfillResult && backfillResult.status === 'error' && (
          <span style={{ fontSize: 11, color: 'var(--accent-red)' }}>Failed</span>
        )}
      </div>
      <div style={{ marginTop: 8 }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
          Loss repair:{' '}
          {[
            program.discovery_loss_ratio == null && 'Discovery',
            program.validation_loss_ratio == null && 'Validation',
          ].filter(Boolean).join(', ') || 'recompute available even if ratios already exist'}
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button
            className="start-btn"
            disabled={lossBackfillRunning}
            onClick={async () => {
              dispatch({ type: 'SET_BACKFILL', payload: { lossBackfillRunning: true, lossBackfillResult: null } });
              try {
                dispatch({ type: 'SET_ACTION', payload: { starting: null, error: null } });
                const res = await postJson(`/api/programs/${resultId}/backfill-loss`, { device: 'cpu' }, { timeoutMs: LONG_ACTION_TIMEOUT_MS });
                if (!res.ok) {
                  const err = await res.json();
                  dispatch({ type: 'SET_ACTION', payload: { starting: null, error: err.error || 'Loss backfill failed' } });
                  dispatch({ type: 'SET_BACKFILL', payload: { lossBackfillRunning: false, lossBackfillResult: { status: 'error' } } });
                } else {
                  const data = await res.json();
                  dispatch({ type: 'SET_BACKFILL', payload: { lossBackfillRunning: false, lossBackfillResult: data.updates || { status: 'ok' } } });
                }
              } catch (e) {
                dispatch({ type: 'SET_ACTION', payload: { starting: null, error: 'Error: ' + e.message } });
                dispatch({ type: 'SET_BACKFILL', payload: { lossBackfillRunning: false, lossBackfillResult: { status: 'error' } } });
              }
            }}
            style={{ padding: '6px 16px', fontSize: 12, background: 'rgba(139, 92, 246, 0.15)', border: '1px solid rgba(139, 92, 246, 0.4)', color: '#a78bfa' }}
          >
            {lossBackfillRunning ? 'Evaluating...' : 'Compute Discovery & Validation Loss'}
          </button>
          {lossBackfillResult && !lossBackfillResult.status && (
            <span style={{ fontSize: 11, color: 'var(--accent-green)' }}>
              {lossBackfillResult.discovery_loss != null && `D: ${Number(lossBackfillResult.discovery_loss).toFixed(4)}`}
              {lossBackfillResult.discovery_loss_ratio != null && ` (LR ${Number(lossBackfillResult.discovery_loss_ratio).toFixed(4)})`}
              {(lossBackfillResult.discovery_loss != null || lossBackfillResult.discovery_loss_ratio != null) && (lossBackfillResult.validation_loss != null || lossBackfillResult.validation_loss_ratio != null) && ' | '}
              {lossBackfillResult.validation_loss != null && `V: ${Number(lossBackfillResult.validation_loss).toFixed(4)}`}
              {lossBackfillResult.validation_loss_ratio != null && ` (LR ${Number(lossBackfillResult.validation_loss_ratio).toFixed(4)})`}
            </span>
          )}
          {lossBackfillResult && lossBackfillResult.status === 'error' && (
            <span style={{ fontSize: 11, color: 'var(--accent-red)' }}>Failed</span>
          )}
        </div>
      </div>
    </div>
  );
}

export default React.memo(EvalResultsPanel);
