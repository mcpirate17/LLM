import React from 'react';

export function SparsityDiagnostics({ program }) {
  const report = program.sparsity_report_json_parsed;
  const ratio = program.sparsity_ratio;
  const deadCount = program.dead_neuron_count;

  if (!report && ratio == null) return null;

  return (
    <div className="card" style={{ padding: 12, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <div className="card-title" style={{ margin: 0 }}>Activation Sparsity</div>
        {report?.max_layer_collapse > 0.9 && (
          <span className="badge badge-error">COLLAPSED</span>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Overall Sparsity</div>
          <div style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)' }}>
            {ratio != null ? `${(ratio * 100).toFixed(1)}%` : 'N/A'}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Dead Neurons</div>
          <div style={{ fontSize: 18, fontWeight: 600, color: deadCount > 0 ? 'var(--accent-yellow)' : 'var(--text-primary)' }}>
            {deadCount != null ? deadCount.toLocaleString() : 'N/A'}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Max Collapse</div>
          <div style={{ fontSize: 18, fontWeight: 600, color: (report?.max_layer_collapse > 0.5) ? 'var(--accent-red)' : 'var(--text-primary)' }}>
            {report?.max_layer_collapse != null ? `${(report.max_layer_collapse * 100).toFixed(1)}%` : 'N/A'}
          </div>
        </div>
      </div>
      
      {report?.layers && (
        <div style={{ display: 'flex', gap: 2, height: 24, alignItems: 'flex-end' }}>
          {report.layers.map((l, i) => (
            <div 
              key={i}
              title={`Layer ${i}: ${(l.sparsity * 100).toFixed(1)}% sparse`}
              style={{
                flex: 1,
                height: `${l.sparsity * 100}%`,
                background: l.sparsity > 0.8 ? 'var(--accent-red)' : 'var(--accent-blue)',
                opacity: 0.7,
                borderRadius: '1px 1px 0 0'
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default React.memo(SparsityDiagnostics);
