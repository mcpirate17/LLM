import React from 'react';

const PerformanceInfo = ({ performance }) => {
  if (!performance) return null;

  return (
    <div className="perf-info">
      {performance.has_params && (
        <div className="perf-row">
          <span className="perf-label">Params</span>
          <span className="perf-value">{performance.param_formula}</span>
        </div>
      )}
      {performance.flops_formula && (
        <div className="perf-row">
          <span className="perf-label">FLOPs</span>
          <span className="perf-value">{performance.flops_formula}</span>
        </div>
      )}
      {performance.numerically_risky && (
        <div className="perf-row warn">Numerically risky</div>
      )}
      {performance.preserves_gradient === false && (
        <div className="perf-row warn">May block gradients</div>
      )}
    </div>
  );
};

export default PerformanceInfo;
