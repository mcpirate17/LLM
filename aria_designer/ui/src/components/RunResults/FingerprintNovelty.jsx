import React from 'react';

function ProgressBar({ value, color = 'var(--accent)' }) {
  const pct = Math.max(0, Math.min(100, (value || 0) * 100));
  return (
    <div className="fp-bar-bg">
      <div className="fp-bar" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

const FingerprintNovelty = ({ fingerprintMetrics, noveltyMetrics }) => {
  return (
    <>
      {fingerprintMetrics && !fingerprintMetrics.skipped && (
        <div className="fingerprint-section">
          <div className="fingerprint-grid">
            {[
              { label: 'Transformer', val: fingerprintMetrics.cka_vs_transformer, color: '#17a3ff' },
              { label: 'SSM', val: fingerprintMetrics.cka_vs_ssm, color: '#a060ff' },
              { label: 'Conv', val: fingerprintMetrics.cka_vs_conv, color: '#ff6090' },
              { label: 'Locality', val: fingerprintMetrics.locality, color: '#24d1a0' },
              { label: 'Sparsity', val: fingerprintMetrics.sparsity, color: '#f0a020' },
              { label: 'Isotropy', val: fingerprintMetrics.isotropy, color: '#20c0f0' },
            ].map(({ label, val, color }) => (
              <div className="fp-row" key={label}>
                <span className="fp-label">{label}</span>
                <ProgressBar value={val} color={color} />
                <span className="fp-val">{val != null ? val.toFixed(2) : '-'}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {noveltyMetrics && !noveltyMetrics.skipped && (
        <div className="novelty-section" style={{ marginTop: 12 }}>
          <div className="novelty-bars">
            {[
              { label: 'Structural', val: noveltyMetrics.structural_novelty },
              { label: 'Behavioral', val: noveltyMetrics.behavioral_novelty },
              { label: 'Overall', val: noveltyMetrics.overall_novelty },
            ].map(({ label, val }) => (
              <div className="novelty-row" key={label}>
                <span className="nov-label">{label}</span>
                <ProgressBar value={val} color="var(--accent)" />
                <span className="fp-val">{val != null ? val.toFixed(2) : '-'}</span>
              </div>
            ))}
            {noveltyMetrics.most_similar_to && (
              <div style={{ fontSize: 11, color: 'var(--muted)', paddingLeft: 88, marginTop: 4 }}>
                Most similar to: <strong style={{ color: 'var(--text)' }}>{noveltyMetrics.most_similar_to}</strong>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
};

export default FingerprintNovelty;
