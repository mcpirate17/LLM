import React from 'react';

const EmptyState = ({ onLoadTemplate }) => {
  return (
    <div className="empty-state">
      <div className="empty-state-content">
        <div className="empty-state-icon">AR</div>
        <h2>Start a New Model Graph</h2>
        <p>Drag components from the palette or launch a starter workflow to begin composing your architecture.</p>
        
        <div className="empty-state-actions">
          <button className="primary" type="button" onClick={() => onLoadTemplate('/examples/transformer_mini.json')}>
            Transformer Mini
          </button>
          <button type="button" onClick={() => onLoadTemplate('/examples/simple_linear.json')}>
            Simple Linear
          </button>
          <button type="button" onClick={() => onLoadTemplate('/examples/hybrid_attn_ssm_moe.json')}>
            Hybrid Stack
          </button>
        </div>

        <div className="empty-state-help">
          <h3>Build Flow</h3>
          <ul>
            <li><strong>1. Validate:</strong> detect cycles and incompatible ports.</li>
            <li><strong>2. Compile:</strong> generate a runnable module.</li>
            <li><strong>3. Run:</strong> execute forward pass and inspect metrics.</li>
            <li><strong>4. Iterate:</strong> apply Aria proposals and compare outcomes.</li>
          </ul>
          <div className="empty-state-shortcuts">
            <span><strong>?</strong> shortcuts</span>
            <span><strong>Del</strong> delete selection</span>
            <span><strong>Ctrl/Cmd+Z</strong> undo</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmptyState;
