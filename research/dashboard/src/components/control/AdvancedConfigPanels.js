import React, { useEffect, useState } from 'react';
import ConfigField from './ConfigField';
import CategoryWeightsControl from './CategoryWeightsControl';
import { apiCall, postJson } from '../../services/apiService';

const detailPanelStyle = { background: 'rgba(255,255,255,0.01)', padding: 16, borderRadius: 8, border: '1px solid var(--border)' };
const accentSummary = { fontSize: 13, fontWeight: 600, color: 'var(--accent-blue)', cursor: 'pointer', padding: '4px 0' };
const mutedSummary = { fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', cursor: 'pointer', padding: '4px 0' };

/**
 * Composite scoring version selector. Reads the active version from
 * /api/scoring/version on mount and POSTs changes back. Stays adjacent to
 * the Capability-First toggle because v8.1 is the companion rebalance.
 */
function ScoringVersionSelector() {
  const [version, setVersion] = useState(null);
  const [supported, setSupported] = useState([]);
  const [status, setStatus] = useState('');

  useEffect(() => {
    apiCall('/api/scoring/version')
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (d && d.version) {
          setVersion(d.version);
          setSupported(d.supported || []);
        }
      })
      .catch(() => {});
  }, []);

  const onChange = (e) => {
    const next = e.target.value;
    setStatus('saving…');
    postJson('/api/scoring/version', { version: next })
      .then(r => r.ok ? r.json() : Promise.reject(r))
      .then(d => {
        setVersion(d.version);
        setStatus(`now ${d.version}`);
        setTimeout(() => setStatus(''), 2000);
      })
      .catch(() => setStatus('save failed'));
  };

  if (!version) return null;

  return (
    <div style={{ marginTop: 10, padding: '8px 10px', background: 'rgba(134,193,165,0.04)', borderRadius: 6, border: '1px dashed rgba(134,193,165,0.25)' }}>
      <ConfigField label="Composite Scoring Version (v8 = default, v8.1 = binding-favoring rebalance)">
        <select value={version} onChange={onChange} style={{ maxWidth: 180 }}>
          {(supported.length > 0 ? supported : ['v7', 'v8', 'v8.1']).map(v => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
      </ConfigField>
      {status && (
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>{status}</div>
      )}
      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.5 }}>
        v8.1: all-binding-signals-below penalty tightens from ×0.80 to ×0.50; +15% boost when 0.4·ar + 0.3·ind + 0.3·bind ≥ 0.05. Historical rows are not rescored.
      </div>
    </div>
  );
}

/**
 * Four collapsible `<details>` panels for advanced experiment configuration:
 * search space weights, structural constraints, automation pipeline, and limits.
 */
function AdvancedConfigPanels({ config, updateConfig, onCategoryWeightChange }) {
  return (
    <>
      <details style={{ marginTop: 16 }}>
        <summary style={accentSummary}>Advanced Search Space Weights</summary>
        <div style={{ marginTop: 12, ...detailPanelStyle }}>
          <CategoryWeightsControl
            weights={config.category_weights}
            onChange={onCategoryWeightChange}
          />
          <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: '1fr', gap: 12 }}>
            <ConfigField label="Excluded Ops (comma-separated op names to never generate)">
              <input type="text" value={config.excluded_ops} onChange={(e) => updateConfig('excluded_ops', e.target.value)} placeholder="e.g. spectral_filter, tropical_gate" />
            </ConfigField>
            <ConfigField label="Op Weights (bias generation toward/away from specific ops: >1 = more likely, <1 = less likely, 0.1 = nearly suppressed)">
              <textarea
                rows={3}
                style={{ width: '100%', fontFamily: 'monospace', fontSize: 12, resize: 'vertical' }}
                value={config.op_weights}
                onChange={(e) => updateConfig('op_weights', e.target.value)}
                placeholder="e.g. route_lanes:2.5, split2:0.3, moe_topk:0.5, sparse_threshold:2.2"
              />
            </ConfigField>
          </div>
        </div>
      </details>

      <details style={{ marginTop: 8 }}>
        <summary style={mutedSummary}>Structural & Grammar Constraints</summary>
        <div className="config-grid" style={{ marginTop: 12, ...detailPanelStyle }}>
          <ConfigField label="Max Depth">
            <input type="number" value={config.max_depth} onChange={(e) => updateConfig('max_depth', parseInt(e.target.value))} />
          </ConfigField>
          <ConfigField label="Max Ops">
            <input type="number" value={config.max_ops} onChange={(e) => updateConfig('max_ops', parseInt(e.target.value))} />
          </ConfigField>
          <ConfigField label="Residual Prob">
            <input type="number" step="0.1" value={config.residual_prob} onChange={(e) => updateConfig('residual_prob', parseFloat(e.target.value))} />
          </ConfigField>
          <ConfigField label="Math Space W">
            <input type="number" step="0.5" value={config.math_space_weight} onChange={(e) => updateConfig('math_space_weight', parseFloat(e.target.value))} />
          </ConfigField>
          <ConfigField label="Grammar Split">
            <input type="number" step="0.05" value={config.grammar_split_prob} onChange={(e) => updateConfig('grammar_split_prob', parseFloat(e.target.value))} />
          </ConfigField>
          <ConfigField label="Grammar Merge">
            <input type="number" step="0.05" value={config.grammar_merge_prob} onChange={(e) => updateConfig('grammar_merge_prob', parseFloat(e.target.value))} />
          </ConfigField>
          <ConfigField label="Min Splits">
            <input type="number" value={config.min_splits} onChange={(e) => updateConfig('min_splits', parseInt(e.target.value))} />
          </ConfigField>
          <ConfigField label="3-Way Prob">
            <input type="number" step="0.05" value={config.three_way_split_prob} onChange={(e) => updateConfig('three_way_split_prob', parseFloat(e.target.value))} />
          </ConfigField>
          <ConfigField label="Branch Depth">
            <input type="number" value={config.branch_depth} onChange={(e) => updateConfig('branch_depth', parseInt(e.target.value))} />
          </ConfigField>
          <ConfigField label="Max Recursion">
            <input type="number" value={config.max_recursion_depth} onChange={(e) => updateConfig('max_recursion_depth', parseInt(e.target.value))} />
          </ConfigField>
        </div>
        <div style={{ marginTop: 12, padding: 12, background: 'rgba(99,179,237,0.06)', borderRadius: 6, border: '1px solid rgba(99,179,237,0.2)' }}>
          <ConfigField label="Evolution Exploit Mode (mutate near top-K winners instead of random; forces routing ops)">
            <input type="checkbox" checked={config.exploit_mode} onChange={(e) => updateConfig('exploit_mode', e.target.checked)} />
          </ConfigField>
          {config.exploit_mode && (
            <div className="config-grid" style={{ marginTop: 8 }}>
              <ConfigField label="Exploit Prob (fraction of offspring guided by archive)">
                <input type="number" step="0.05" min="0" max="1" value={config.exploit_prob} onChange={(e) => updateConfig('exploit_prob', parseFloat(e.target.value))} />
              </ConfigField>
              <ConfigField label="Local Mutation Prob (single-op swap for top-K parents)">
                <input type="number" step="0.05" min="0" max="1" value={config.local_mutation_prob} onChange={(e) => updateConfig('local_mutation_prob', parseFloat(e.target.value))} />
              </ConfigField>
              <ConfigField label="Exploit Top-K (how many top individuals to exploit around)">
                <input type="number" min="1" max="20" value={config.exploit_top_k} onChange={(e) => updateConfig('exploit_top_k', parseInt(e.target.value))} />
              </ConfigField>
            </div>
          )}
        </div>
        <div style={{ marginTop: 12, padding: 12, background: 'rgba(134,193,165,0.06)', borderRadius: 6, border: '1px solid rgba(134,193,165,0.25)' }}>
          <ConfigField label="Capability-First Mode (trunk+sidecar graphs, promotes role-slot templates, enables gate8_retrieval_dead; overrides exploit/routing_first preset)">
            <input type="checkbox" checked={!!config._capability_first_mode} onChange={(e) => updateConfig('_capability_first_mode', e.target.checked)} />
          </ConfigField>
          {config._capability_first_mode && (
            <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
              Samples from <code>typed_slot_memory_block</code>, <code>sparse_relation_graph_block</code>, <code>token_program_interpreter_block</code>, and the three <code>*_retrieval_v2</code> variants. Boosts retrieval ops (matmul, outer_product, gather_topk, cosine_similarity, token_type_classifier, entropy_score) and rejects graphs with no content-addressed op at screening. Pair with scoring version v8.1 below for the capability-first composite rebalance.
            </div>
          )}
          <ScoringVersionSelector />
        </div>
        <div style={{ marginTop: 12, padding: 12, background: 'rgba(247,178,103,0.06)', borderRadius: 6, border: '1px solid rgba(247,178,103,0.25)' }}>
          <ConfigField label="Dynamically Learned Slots (narrows multi-class slots to empirical pass-cohort motif classes; falls back to hardcoded tuples when meta DB lacks data)">
            <input type="checkbox" checked={!!config.use_derived_slot_classes} onChange={(e) => updateConfig('use_derived_slot_classes', e.target.checked)} />
          </ConfigField>
          {config.use_derived_slot_classes && (
            <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
              At synth-init, the picker consults <code>research/meta_analysis.db</code> for per-(template, slot_index) pass-cohort motif-class qualifying sets (n≥10, class_pass_rate≥0.40). When data exists, the slot's allow-list is replaced with the qualifying classes; otherwise the existing hardcoded tuple is used. Templates with strong cohort signal (latent_attn_*, attn_*) get tightened slots; templates without it (residual_block, transformer_block) are unchanged. See <code>research/synthesis/_slot_constraints_loader.py</code>.
            </div>
          )}
        </div>
      </details>

      <details style={{ marginTop: 8 }}>
        <summary style={mutedSummary}>Automation & Discovery Pipeline</summary>
        <div className="config-grid" style={{ marginTop: 12, ...detailPanelStyle }}>
          <ConfigField label="Auto Scale Up">
            <input type="checkbox" checked={config.auto_scale_up} onChange={(e) => updateConfig('auto_scale_up', e.target.checked)} />
          </ConfigField>
          <ConfigField label="Auto Investigate">
            <input type="checkbox" checked={config.auto_investigate} onChange={(e) => updateConfig('auto_investigate', e.target.checked)} />
          </ConfigField>
          <ConfigField label="Auto Validate">
            <input type="checkbox" checked={config.auto_validate} onChange={(e) => updateConfig('auto_validate', e.target.checked)} />
          </ConfigField>
          <ConfigField label="Auto Report">
            <input type="checkbox" checked={config.auto_report} onChange={(e) => updateConfig('auto_report', e.target.checked)} />
          </ConfigField>
          <ConfigField label="Pruning Base.">
            <input type="checkbox" checked={config.one_shot_pruning_baseline} onChange={(e) => updateConfig('one_shot_pruning_baseline', e.target.checked)} />
          </ConfigField>
          {config.one_shot_pruning_baseline && (
            <>
              <ConfigField label="Pruning Spar.">
                <input type="number" step="0.05" value={config.one_shot_pruning_sparsity} onChange={(e) => updateConfig('one_shot_pruning_sparsity', parseFloat(e.target.value))} />
              </ConfigField>
              <ConfigField label="Pruning Meth.">
                <select value={config.one_shot_pruning_method} onChange={(e) => updateConfig('one_shot_pruning_method', e.target.value)}>
                  <option value="wanda">Wanda</option>
                  <option value="sparsegpt">SparseGPT</option>
                </select>
              </ConfigField>
            </>
          )}
        </div>
        <div className="config-grid" style={{ marginTop: 12, ...detailPanelStyle }}>
          <ConfigField label="Invest. Steps">
            <input type="number" value={config.investigation_steps} onChange={(e) => updateConfig('investigation_steps', parseInt(e.target.value))} />
          </ConfigField>
          <ConfigField label="Valid. Steps">
            <input type="number" value={config.validation_steps} onChange={(e) => updateConfig('validation_steps', parseInt(e.target.value))} />
          </ConfigField>
        </div>
      </details>

      <details style={{ marginTop: 8 }}>
        <summary style={mutedSummary}>Experiment Limits & Source</summary>
        <div className="config-grid" style={{ marginTop: 12, ...detailPanelStyle }}>
          <ConfigField label="Max Experim.">
            <input type="number" value={config.max_experiments} onChange={(e) => updateConfig('max_experiments', parseInt(e.target.value))} />
          </ConfigField>
          <ConfigField label="Max Time (m)">
            <input type="number" value={config.max_time_minutes} onChange={(e) => updateConfig('max_time_minutes', parseInt(e.target.value))} />
          </ConfigField>
          <ConfigField label="Model Source">
            <select value={config.model_source} onChange={(e) => updateConfig('model_source', e.target.value)}>
              <option value="synthesized">Synthesized</option>
              <option value="morphed">Morphed</option>
              <option value="mixed">Mixed</option>
            </select>
          </ConfigField>
          <ConfigField label="Morph Ratio">
            <input type="number" step="0.1" value={config.morph_ratio} onChange={(e) => updateConfig('morph_ratio', parseFloat(e.target.value))} />
          </ConfigField>
        </div>
      </details>
    </>
  );
}

export default React.memo(AdvancedConfigPanels);
