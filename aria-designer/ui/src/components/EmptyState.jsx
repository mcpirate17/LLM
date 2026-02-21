import React from 'react';

const EmptyState = ({ onLoadTemplate }) => {
  return (
    <div className="empty-state">
      <div className="empty-state-content">
        <div className="empty-state-icon">✨</div>
        <h2>Welcome to Aria Designer</h2>
        <p>Get started by dragging components from the sidebar or loading a starter template.</p>
        
        <div className="empty-state-actions">
          <button className="primary" onClick={() => onLoadTemplate('tpl_mlp')}>
            Load Standard MLP
          </button>
          <button onClick={() => onLoadTemplate('tpl_linear')}>
            Simple Linear
          </button>
        </div>

        <div className="empty-state-help">
          <h3>Quick Tips</h3>
          <ul>
            <li><strong>Validate:</strong> Check graph for cycles and type errors.</li>
            <li><strong>Compile:</strong> Build a runnable PyTorch module.</li>
            <li><strong>Run:</strong> Execute a forward pass with metrics.</li>
            <li><strong>Proposals:</strong> Review Aria's co-design suggestions.</li>
          </ul>
          <p className="muted">Press <strong>?</strong> for keyboard shortcuts.</p>
        </div>
      </div>
    </div>
  );
};

export default EmptyState;
