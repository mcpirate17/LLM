import React, { useState, useEffect } from 'react';
import ConfigField from './ConfigField';
import { apiCall } from '../../services/apiService';

/**
 * Architecture search space, training parameters, and evolutionary strategy config grids.
 */
function SearchSpaceConfig({ config, updateConfig, isEvolutionMode }) {
  const gridStyle = { background: 'rgba(255,255,255,0.02)', padding: 16, borderRadius: 8, border: '1px solid var(--border)' };
  const [templateNames, setTemplateNames] = useState([]);

  useEffect(() => {
    apiCall('/api/template-names').then(r => r.ok ? r.json() : { names: [] })
      .then(d => setTemplateNames(d.names || []))
      .catch(() => {});
  }, []);

  return (
    <>
      <div className="section-subheader">Architecture Search Space</div>
      <div className="config-grid" style={gridStyle}>
        {!isEvolutionMode && (
          <ConfigField label="Programs">
            <input type="number" min="5" max="500" value={config.n_programs} onChange={(e) => updateConfig('n_programs', parseInt(e.target.value))} />
          </ConfigField>
        )}
        <ConfigField label="Dimension">
          <select value={config.model_dim} onChange={(e) => updateConfig('model_dim', parseInt(e.target.value))}>
            {[64, 128, 256, 512].map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </ConfigField>
        <ConfigField label="Layers">
          <input type="number" min="1" max="12" value={config.n_layers} onChange={(e) => updateConfig('n_layers', parseInt(e.target.value))} />
        </ConfigField>
        <ConfigField label="Force Template" title="Force all generated graphs to use this template. Leave blank for normal weighted selection.">
          <select
            value={config.forced_template || ''}
            onChange={(e) => updateConfig('forced_template', e.target.value || null)}
            style={{ maxWidth: 220 }}
          >
            <option value="">-- auto --</option>
            {templateNames.map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </ConfigField>
      </div>

      <div className="section-subheader">Training Parameters (Stage 1)</div>
      <div className="config-grid" style={gridStyle}>
        <ConfigField label="Steps">
          <input type="number" value={config.stage1_steps} onChange={(e) => updateConfig('stage1_steps', parseInt(e.target.value))} />
        </ConfigField>
        <ConfigField label="LR">
          <input type="number" step="0.0001" value={config.stage1_lr} onChange={(e) => updateConfig('stage1_lr', parseFloat(e.target.value))} />
        </ConfigField>
      </div>

      {isEvolutionMode && (
        <>
          <div className="section-subheader">Evolutionary Strategy</div>
          <div className="config-grid" style={gridStyle}>
            <ConfigField label="Population">
              <input type="number" value={config.population_size} onChange={(e) => updateConfig('population_size', parseInt(e.target.value))} />
            </ConfigField>
            <ConfigField label="Generations">
              <input type="number" value={config.n_generations} onChange={(e) => updateConfig('n_generations', parseInt(e.target.value))} />
            </ConfigField>
            <ConfigField label="Tourn Size">
              <input type="number" value={config.tournament_size} onChange={(e) => updateConfig('tournament_size', parseInt(e.target.value))} />
            </ConfigField>
            <ConfigField label="Mutation Rate">
              <input type="number" step="0.1" value={config.mutation_rate} onChange={(e) => updateConfig('mutation_rate', parseFloat(e.target.value))} />
            </ConfigField>
          </div>
        </>
      )}
    </>
  );
}

export default React.memo(SearchSpaceConfig);
