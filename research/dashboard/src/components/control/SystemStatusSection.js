import React, { useState, useCallback } from 'react';
import { apiCall, postJson } from '../../services/apiService';
import ConfigField from './ConfigField';

/**
 * System status badges (CUDA/LLM) and collapsible LLM configuration form.
 * Manages its own LLM form state to avoid polluting the parent.
 */
function SystemStatusSection({ systemStatus, onSystemStatusUpdate }) {
  const [showLlmConfig, setShowLlmConfig] = useState(false);
  const [llmConfig, setLlmConfig] = useState(null);
  const [llmForm, setLlmForm] = useState({ backend: '', api_key: '', model: '', host: '' });
  const [llmSaving, setLlmSaving] = useState(false);
  const [llmMessage, setLlmMessage] = useState('');

  // Fetch LLM config on first open
  const handleToggleConfig = useCallback(() => {
    if (!showLlmConfig) {
      if (!llmConfig) {
        apiCall('/api/llm/config')
          .then(r => r.ok ? r.json() : null)
          .then(data => {
            if (data) {
              setLlmConfig(data);
              setLlmForm({
                backend: data.backend || '',
                api_key: '',
                model: data.model || '',
                host: data.host || '',
              });
            }
          })
          .catch(() => {});
      } else {
        setLlmForm({
          backend: llmConfig.backend || '',
          api_key: '',
          model: llmConfig.model || '',
          host: llmConfig.host || '',
        });
      }
    }
    setShowLlmConfig(prev => !prev);
  }, [showLlmConfig, llmConfig]);

  const handleLlmSave = useCallback(async () => {
    if (!llmForm.backend) return;
    setLlmSaving(true);
    setLlmMessage('');
    try {
      const res = await postJson('/api/llm/config', llmForm);
      const data = await res.json();
      if (res.ok) {
        setLlmConfig(data.config);
        setLlmMessage(data.warning ? `Warning: ${data.warning}` : 'LLM configured and verified successfully');
        setLlmForm(prev => ({ ...prev, api_key: '' }));
        window.dispatchEvent(new CustomEvent('llm-configured'));
        apiCall('/api/system/status')
          .then(r => r.ok ? r.json() : null)
          .then(d => { if (d) onSystemStatusUpdate(d); })
          .catch(() => {});
      } else {
        setLlmMessage(data.error || 'Configuration failed');
      }
    } catch (e) {
      setLlmMessage('Error: ' + e.message);
    }
    setLlmSaving(false);
  }, [llmForm, onSystemStatusUpdate]);

  return (
    <>
      <div className="system-status-badges">
        <span className={`sys-badge ${systemStatus?.cuda?.available ? 'pass' : 'fail'}`}>
          {systemStatus?.cuda?.available ? `CUDA: ${systemStatus.cuda.device_name || 'GPU'}` : 'CPU Only'}
        </span>
        <span className={`sys-badge ${systemStatus?.llm?.available ? 'pass' : 'warn'}`}>
          {systemStatus?.llm?.available ? `LLM: ${systemStatus.llm.backend}` : 'No LLM'}
        </span>
        <button className="sys-badge info" onClick={handleToggleConfig}>
          {showLlmConfig ? 'Hide Config' : 'Configure LLM'}
        </button>
      </div>

      {systemStatus?.ml_influence && (
        <div className="system-status-badges" style={{ marginTop: 8, flexWrap: 'wrap' }}>
          <span className={`sys-badge ${systemStatus.ml_influence.defaults.gbm_prescreener_enabled ? 'warn' : 'pass'}`}>
            {systemStatus.ml_influence.defaults.gbm_prescreener_enabled ? 'ML Gate: Active' : 'ML Gate: Monitor Only'}
          </span>
          <span className={`sys-badge ${systemStatus.ml_influence.defaults.use_learned_candidate_weights ? 'warn' : 'pass'}`}>
            {systemStatus.ml_influence.defaults.use_learned_candidate_weights ? 'Candidate Weights: Active' : 'Candidate Weights: Off'}
          </span>
          <span className={`sys-badge ${systemStatus.ml_influence.defaults.use_screening_signal_weights ? 'warn' : 'pass'}`}>
            {systemStatus.ml_influence.defaults.use_screening_signal_weights ? 'Signal Weights: Active' : 'Signal Weights: Off'}
          </span>
          <span className={`sys-badge ${systemStatus.ml_influence.defaults.use_learned_grammar_weights ? 'warn' : 'pass'}`}>
            {systemStatus.ml_influence.defaults.use_learned_grammar_weights ? 'Grammar Weights: Active' : 'Grammar Weights: Off'}
          </span>
        </div>
      )}

      {showLlmConfig && (
        <div className="llm-config-section">
          <div className="config-grid">
            <ConfigField label="Backend">
              <select value={llmForm.backend} onChange={(e) => setLlmForm(prev => ({ ...prev, backend: e.target.value }))}>
                <option value="">Select...</option>
                <option value="anthropic">Anthropic (Claude)</option>
                <option value="openai">OpenAI</option>
                <option value="ollama">Ollama (Local)</option>
              </select>
            </ConfigField>
            {(llmForm.backend === 'anthropic' || llmForm.backend === 'openai') && (
              <>
                <ConfigField label={llmConfig?.api_key_set ? `API Key (saved: ${llmConfig.api_key_hint})` : 'API Key'}>
                  <input
                    type="text"
                    name="llm_api_key_field"
                    autoComplete="off"
                    value={llmForm.api_key}
                    placeholder={llmConfig?.api_key_set ? 'Leave blank to keep current key' : 'sk-ant-...'}
                    onChange={(e) => setLlmForm(prev => ({ ...prev, api_key: e.target.value }))}
                    style={llmConfig?.api_key_set && !llmForm.api_key ? { borderColor: 'var(--accent)', opacity: 0.8 } : {}}
                  />
                </ConfigField>
                <ConfigField label="Model">
                  <input
                    type="text"
                    value={llmForm.model}
                    placeholder={llmForm.backend === 'anthropic' ? "claude-sonnet-4-5-20250929" : "gpt-4o"}
                    onChange={(e) => setLlmForm(prev => ({ ...prev, model: e.target.value }))}
                  />
                </ConfigField>
              </>
            )}
            {llmForm.backend === 'ollama' && (
              <>
                <ConfigField label="Model">
                  <input
                    type="text"
                    value={llmForm.model}
                    placeholder="e.g. qwen2.5-coder:7b"
                    onChange={(e) => setLlmForm(prev => ({ ...prev, model: e.target.value }))}
                  />
                </ConfigField>
                <ConfigField label="Host">
                  <input
                    type="text"
                    value={llmForm.host}
                    placeholder="http://localhost:11434"
                    onChange={(e) => setLlmForm(prev => ({ ...prev, host: e.target.value }))}
                  />
                </ConfigField>
              </>
            )}
          </div>
          {llmForm.backend && (
            <button className="validate-btn" onClick={handleLlmSave} disabled={llmSaving} style={{ marginTop: 8 }}>
              {llmSaving ? 'Configuring...' : 'Configure LLM'}
            </button>
          )}
          {llmMessage && <div className={`llm-message ${llmMessage.includes('success') ? 'pass' : 'fail'}`}>{llmMessage}</div>}
        </div>
      )}
    </>
  );
}

export default React.memo(SystemStatusSection);
