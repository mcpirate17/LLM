import React from 'react';
import { formatNum } from '../../utils/format';

const MetricGrid = ({ sandboxMetrics, profilingMetrics }) => {
  if (!sandboxMetrics) return null;

  return (
    <div className="metrics-grid">
      <div className="stat">
        <div className="stat-val">{formatNum(sandboxMetrics.param_count)}</div>
        <div className="stat-label">Params</div>
      </div>
      <div className="stat">
        <div className="stat-val">{formatNum(profilingMetrics?.total_flops_per_token)}</div>
        <div className="stat-label">FLOPs/tok</div>
      </div>
      <div className="stat">
        <div className="stat-val">{
          sandboxMetrics.peak_memory_mb
            ? formatNum(sandboxMetrics.peak_memory_mb) + 'MB'
            : profilingMetrics?.total_memory_bytes
              ? formatNum(profilingMetrics.total_memory_bytes / (1024 * 1024)) + 'MB'
              : '-'
        }</div>
        <div className="stat-label">Memory</div>
      </div>
      <div className="stat">
        <div className="stat-val">{sandboxMetrics.forward_ms?.toFixed(1)}</div>
        <div className="stat-label">Fwd ms</div>
      </div>
      <div className="stat">
        <div className="stat-val">{sandboxMetrics.backward_ms?.toFixed(1)}</div>
        <div className="stat-label">Bwd ms</div>
      </div>
      <div className="stat">
        <div className="stat-val">{sandboxMetrics.stability_score?.toFixed(2)}</div>
        <div className="stat-label">Stability</div>
      </div>
    </div>
  );
};

export default MetricGrid;
