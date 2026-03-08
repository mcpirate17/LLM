import React from 'react';
import AriaAvatar from './AriaAvatar';
import '../styles/Header.css';

const Header = ({ 
  workflowMeta, 
  saveState, 
  onSave, 
  onExport,
  onImport, 
  onAskAria, 
  onShowShortcuts, 
  runStatus,
  onValidate,
  onCompile,
  onRun
}) => {
  return (
    <header className="app-header">
      <div className="header-left">
        <div className="logo">
          <AriaAvatar size={32} mood={runStatus.phase === 'running' ? 'busy' : 'neutral'} />
          <h1>Aria Designer</h1>
        </div>
        <div className="workflow-info">
          <span className="workflow-name">{workflowMeta.name || 'Untitled Workflow'}</span>
          {saveState.version && (
            <span className="workflow-version">v{saveState.version}</span>
          )}
        </div>
      </div>
      
      <div className="header-center">
        <div className="workflow-steps toolbar-group workflow">
          <button 
            className={`step-btn step-state-${runStatus.phase === 'validated' ? 'pass' : runStatus.phase === 'failed' ? 'fail' : 'idle'} ${runStatus.phase === 'validating' ? 'busy' : ''}`}
            onClick={onValidate}
            title="Step 1: Validate"
          >
            Step 1: Validate
          </button>
          <button 
            className={`step-btn step-state-${runStatus.phase === 'compiled' ? 'pass' : 'idle'} ${runStatus.phase === 'compiling' ? 'busy' : ''}`}
            onClick={onCompile}
            title="Step 2: Compile"
          >
            Step 2: Compile
          </button>
          <button 
            className={`step-btn step-state-${runStatus.phase === 'success' ? 'pass' : 'idle'} ${runStatus.phase === 'running' ? 'busy' : ''}`}
            onClick={onRun}
            title="Step 3: Test"
          >
            Step 3: Test
          </button>
        </div>
        <div className="run-status-msg">
          {runStatus.message}
        </div>
      </div>

      <div className="header-right actions">
        <div className="toolbar-group library">
          <select
            className="example-select"
            onChange={(e) => {
              if (e.target.value) {
                window.dispatchEvent(new CustomEvent('load-example', { detail: e.target.value }));
              }
            }}
          >
            <option value="">Load Example...</option>
            <option value="/examples/simple_linear.json">Simple Linear</option>
            <option value="/examples/tropical_attention.json">Tropical Attention</option>
            <option value="/examples/tropical_block.json">Tropical Block</option>
            <option value="/examples/transformer_mini.json">Transformer Mini</option>
            <option value="/examples/ssm_stack.json">SSM Stack</option>
            <option value="/examples/hybrid_attn_ssm_moe.json">Hybrid Stack</option>
          </select>
        </div>
        <div className="toolbar-group files">
          <button className="btn-secondary" onClick={onImport}>Import</button>
          <button className="btn-secondary" onClick={onExport}>Export</button>
          <button className="btn-primary" onClick={onSave}>Save</button>
        </div>
        <div className="toolbar-group ai">
          <button className="btn-aria" onClick={onAskAria}>Ask Aria</button>
        </div>
        <button className="btn-icon" onClick={onShowShortcuts} title="Keyboard Shortcuts">⌨️</button>
      </div>
    </header>
  );
};

export default Header;
