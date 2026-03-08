import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { formatNum } from '../../utils/format';

function ProgressBar({ value, color = 'var(--accent)' }) {
  const pct = Math.max(0, Math.min(100, (value || 0) * 100));
  return (
    <div className="fp-bar-bg">
      <div className="fp-bar" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

const CompressionEfficiency = ({ compressionMetrics }) => {
  if (!compressionMetrics) return null;

  return (
    <>
      <div className="efficiency-score-badge">
        <div className="eff-score-value">{Math.round((compressionMetrics.efficiency_score || 0) * 100)}</div>
        <div className="eff-score-label">Efficiency</div>
      </div>

      <div className="compression-breakdown">
        {[
          { label: 'Prune Tol.', val: compressionMetrics.pruning_tolerance, color: '#24d1a0' },
          { label: 'Compression', val: Math.min((compressionMetrics.compression_ratio || 1) / 4, 1), color: '#17a3ff' },
          { label: 'Sparse Ops', val: compressionMetrics.sparse_op_coverage, color: '#a060ff' },
          { label: 'Mem Eff.', val: compressionMetrics.memory_efficiency_score, color: '#f0a020' },
        ].map(({ label, val, color }) => (
          <div className="fp-row" key={label}>
            <span className="fp-label">{label}</span>
            <ProgressBar value={val} color={color} />
            <span className="fp-val">{val != null ? (val * 100).toFixed(0) + '%' : '-'}</span>
          </div>
        ))}
      </div>

      {compressionMetrics.pruning_curve?.length > 0 && (
        <div className="chart-container" style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 4 }}>Pruning Curve</div>
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={compressionMetrics.pruning_curve} margin={{ left: 10, right: 10, top: 4, bottom: 4 }}>
              <XAxis dataKey="sparsity" tick={{ fill: '#8fa8c2', fontSize: 10 }} tickFormatter={v => (v * 100) + '%'} />
              <YAxis tick={{ fill: '#8fa8c2', fontSize: 10 }} domain={[0, 'auto']} tickFormatter={v => v.toFixed(1) + 'x'} />
              <Tooltip
                contentStyle={{ background: '#101b2b', border: '1px solid #1f3147', borderRadius: 6, fontSize: 12 }}
                formatter={(v) => v.toFixed(3) + 'x'}
                labelFormatter={(v) => 'Sparsity: ' + (v * 100) + '%'}
              />
              <Line type="monotone" dataKey="loss_ratio" stroke="#a060ff" strokeWidth={2} dot={{ r: 3, fill: '#a060ff' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="metrics-grid" style={{ marginTop: 8 }}>
        <div className="stat">
          <div className="stat-val">{formatNum(compressionMetrics.compression_ratio?.toFixed(2))}x</div>
          <div className="stat-label">Compression</div>
        </div>
        <div className="stat">
          <div className="stat-val">{compressionMetrics.sparse_ops || 0}</div>
          <div className="stat-label">Sparse Ops</div>
        </div>
        <div className="stat">
          <div className="stat-val">{compressionMetrics.theoretical_size_int8_mb?.toFixed(1)}MB</div>
          <div className="stat-label">INT8 Size</div>
        </div>
        <div className="stat">
          <div className="stat-val">{compressionMetrics.theoretical_size_int4_mb?.toFixed(1)}MB</div>
          <div className="stat-label">INT4 Size</div>
        </div>
      </div>

      {compressionMetrics.sparse_op_names?.length > 0 && (
        <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {compressionMetrics.sparse_op_names.map(name => (
            <span key={name} className="sparse-op-badge">{name}</span>
          ))}
        </div>
      )}
    </>
  );
};

export default CompressionEfficiency;
