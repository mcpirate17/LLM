import React from 'react';
import ConfigField from './ConfigField';
import CategoryWeightsControl from './CategoryWeightsControl';

const detailPanelStyle = { background: 'rgba(255,255,255,0.01)', padding: 16, borderRadius: 8, border: '1px solid var(--border)' };
const accentSummary = { fontSize: 13, fontWeight: 600, color: 'var(--accent-blue)', cursor: 'pointer', padding: '4px 0' };
const mutedSummary = { fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', cursor: 'pointer', padding: '4px 0' };

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
